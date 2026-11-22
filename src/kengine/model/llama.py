from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from kengine.config import ModelConfig
from kengine.loader import load_state_dict
from kengine.model.attention import KVStore, PagedAttention, PagedBatch
from kengine.model.layers import MLP, RMSNorm, rope_cos_sin


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.self_attn = PagedAttention(cfg)
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        batch: PagedBatch,
        kv: KVStore,
        layer: int,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, batch, kv, layer)
        return x + self.mlp(self.post_attention_layernorm(x))


class Llama(nn.Module):
    """Llama-family decoder. Stateless: every K/V it needs is read from / written to ``kv``.

    Submodule names mirror the HF checkpoint so weights load without a key mapping.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, batch: PagedBatch, kv: KVStore, last_only: bool = True) -> torch.Tensor:
        """Logits for each sequence's last new token ([B, V]), or for all of them ([B, T, V])."""
        h = self.embed_tokens(batch.tokens)
        cos, sin = rope_cos_sin(batch.positions, self.cfg.head_dim, self.cfg.rope_theta)
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, batch, kv, i)
        if last_only:
            h = h[torch.arange(h.shape[0], device=h.device), batch.last_index]
        return self.lm_head(self.norm(h))

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        cfg: ModelConfig,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Llama:
        state = {k.removeprefix("model."): v for k, v in load_state_dict(model_dir).items()}
        # Checkpoints with tied embeddings omit the output projection.
        state.setdefault("lm_head.weight", state["embed_tokens.weight"])
        model = cls(cfg)
        model.load_state_dict(state, strict=True)
        return model.to(device=device, dtype=dtype).eval()
