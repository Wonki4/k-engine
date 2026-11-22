from __future__ import annotations

import itertools
from collections.abc import Sequence as Seq
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from kengine.cache import BlockPool, BlockTable, PrefixCache
from kengine.config import EngineConfig, ModelConfig
from kengine.loader import load_tokenizer, resolve_model_dir
from kengine.model import Llama, PagedBatch
from kengine.sampler import sample
from kengine.scheduler import Scheduler
from kengine.sequence import SamplingParams, Sequence


@dataclass(frozen=True)
class StepOutput:
    seq_id: int
    token_id: int
    finish_reason: str | None  # "stop", "length", or None while still generating


def build_batch(
    items: Seq[tuple[BlockTable, Seq[int]]], device: torch.device | str = "cpu"
) -> PagedBatch:
    """Lay out one forward pass: for each (table, new_tokens), feed the new tokens on top of
    the ``table.num_tokens`` already cached. Blocks for the new tokens must be reserved."""
    B = len(items)
    cached = [table.num_tokens for table, _ in items]
    new = [len(tokens) for _, tokens in items]
    T, S = max(new), max(c + n for c, n in zip(cached, new, strict=True))

    tokens = torch.zeros(B, T, dtype=torch.long)
    read_slots = torch.zeros(B, S, dtype=torch.long)
    write_slots: list[int] = []
    for b, (table, new_tokens) in enumerate(items):
        slots = table.slots(0, cached[b] + new[b])
        tokens[b, : new[b]] = torch.tensor(new_tokens, dtype=torch.long)
        read_slots[b, : len(slots)] = torch.tensor(slots, dtype=torch.long)
        write_slots.extend(slots[cached[b] :])

    cached_t = torch.tensor(cached)[:, None]
    new_t = torch.tensor(new)[:, None]
    t = torch.arange(T)[None, :]
    positions = cached_t + t  # [B, T]
    # New token t sees context slot s iff s is at or before it and inside this sequence's
    # context. Padding rows of T keep at least slot 0 visible, so softmax never sees all -inf.
    s = torch.arange(S)[None, None, :]
    attn_mask = (s <= positions[:, :, None]) & (s < (cached_t + new_t)[:, :, None])

    return PagedBatch(
        tokens=tokens.to(device),
        positions=positions.to(device),
        write_mask=(t < new_t).to(device),
        write_slots=torch.tensor(write_slots, dtype=torch.long, device=device),
        read_slots=read_slots.to(device),
        attn_mask=attn_mask.to(device),
        last_index=(new_t[:, 0] - 1).to(device),
    )


class Engine:
    """Ties the pieces together: scheduler picks sequences, the model runs them against the
    block pool, the sampler picks tokens. ``step()`` is one iteration; because the batch is
    rebuilt every step, requests join and leave mid-flight (continuous batching)."""

    def __init__(
        self,
        model: Llama,
        tokenizer: Tokenizer,
        model_cfg: ModelConfig,
        engine_cfg: EngineConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg
        self.cfg = engine_cfg or EngineConfig()
        param = next(model.parameters())
        self.device = param.device
        self.pool = BlockPool(
            num_layers=model_cfg.num_hidden_layers,
            num_blocks=self.cfg.num_blocks,
            block_size=self.cfg.block_size,
            num_kv_heads=model_cfg.num_key_value_heads,
            head_dim=model_cfg.head_dim,
            device=self.device,
            dtype=param.dtype,
        )
        self.prefix_cache = PrefixCache(self.pool)
        self.pool.reclaimer = self.prefix_cache.evict_one
        self.scheduler = Scheduler(self.pool, self.prefix_cache, self.cfg.max_batch_size)
        self.computed_tokens = 0
        self._ids = itertools.count()

    @classmethod
    def from_pretrained(
        cls, name_or_path: str | Path, engine_cfg: EngineConfig | None = None
    ) -> Engine:
        engine_cfg = engine_cfg or EngineConfig()
        model_dir = resolve_model_dir(name_or_path)
        model_cfg = ModelConfig.from_file(model_dir / "config.json")
        model = Llama.from_pretrained(
            model_dir, model_cfg, engine_cfg.resolve_device(), engine_cfg.resolve_dtype()
        )
        return cls(model, load_tokenizer(model_dir), model_cfg, engine_cfg)

    # -- requests -----------------------------------------------------------------------------

    def make_sequence(
        self, prompt: str | Seq[int], params: SamplingParams | None = None
    ) -> Sequence:
        """Validate a request and build its sequence without queueing it."""
        params = params or SamplingParams()
        ids = self.tokenizer.encode(prompt).ids if isinstance(prompt, str) else list(prompt)
        if not ids:
            raise ValueError("prompt is empty")
        if params.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        total = len(ids) + params.max_tokens
        if total > self.model_cfg.max_position_embeddings:
            raise ValueError(
                f"prompt ({len(ids)}) + max_tokens ({params.max_tokens}) exceeds the model's "
                f"context of {self.model_cfg.max_position_embeddings} tokens"
            )
        # A request that cannot fit even with the pool to itself would never be admitted.
        capacity = self.cfg.num_blocks * self.cfg.block_size
        if total > capacity:
            raise ValueError(
                f"prompt ({len(ids)}) + max_tokens ({params.max_tokens}) exceeds the KV pool "
                f"capacity of {capacity} tokens"
            )
        rng = None
        if params.seed is not None:
            rng = torch.Generator().manual_seed(params.seed)
        return Sequence(id=next(self._ids), prompt_ids=ids, params=params, rng=rng)

    def submit(self, seq: Sequence) -> None:
        self.scheduler.add(seq)

    def add_request(self, prompt: str | Seq[int], params: SamplingParams | None = None) -> Sequence:
        seq = self.make_sequence(prompt, params)
        self.submit(seq)
        return seq

    def abort(self, seq: Sequence) -> None:
        self.scheduler.finish(seq, "abort")

    def abort_all(self) -> list[Sequence]:
        """Abort every queued and running sequence, releasing their blocks."""
        seqs = [*self.scheduler.running, *self.scheduler.waiting]
        for seq in seqs:
            self.abort(seq)
        return seqs

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    # -- execution ----------------------------------------------------------------------------

    @torch.inference_mode()
    def step(self) -> list[StepOutput]:
        """Run every scheduled sequence forward by one sampled token."""
        scheduled = self.scheduler.schedule()
        # Prefill (many new tokens) and decode (one) run as separate batches so a long prompt
        # does not pad every decoding sequence out to its length.
        prefill = [s for s in scheduled if s.num_pending > 1]
        decode = [s for s in scheduled if s.num_pending == 1]

        outputs: list[StepOutput] = []
        for group in (prefill, decode):
            if not group:
                continue
            batch = build_batch([(s.table, s.tokens[s.num_cached :]) for s in group], self.device)
            logits = self.model(batch, self.pool).float().cpu()
            for seq, row in zip(group, logits, strict=True):
                self.computed_tokens += seq.num_pending
                seq.table.advance(seq.num_pending)
                if group is prefill:
                    # Share a freshly computed prompt right away, not only once it finishes.
                    self.prefix_cache.insert(seq.tokens[: seq.num_cached], seq.table.blocks)
                token = sample(row, seq.params, seq.rng)
                seq.output_ids.append(token)
                reason = self._finish_reason(seq, token)
                if reason is not None:
                    self.scheduler.finish(seq, reason)
                outputs.append(StepOutput(seq.id, token, reason))
        return outputs

    def _finish_reason(self, seq: Sequence, token: int) -> str | None:
        params = seq.params
        if token in params.stop_token_ids:
            return "stop"
        if not params.ignore_eos and token in self.model_cfg.eos_token_ids:
            return "stop"
        if len(seq.output_ids) >= params.max_tokens:
            return "length"
        return None

    def generate(
        self, prompts: Seq[str | Seq[int]], params: SamplingParams | None = None
    ) -> list[list[int]]:
        """Run a set of prompts to completion and return each one's generated token ids."""
        seqs = [self.add_request(p, params) for p in prompts]
        while self.has_work():
            self.step()
        return [s.output_ids for s in seqs]

    def stats(self) -> dict[str, int]:
        return {
            "blocks_total": self.pool.num_blocks,
            "blocks_free": self.pool.num_free,
            "blocks_prefix_cached": len(self.prefix_cache),
            "running": len(self.scheduler.running),
            "waiting": len(self.scheduler.waiting),
            "computed_tokens": self.computed_tokens,
            "prefix_hit_tokens": self.scheduler.stats.prefix_hit_tokens,
            "preemptions": self.scheduler.stats.preemptions,
        }
