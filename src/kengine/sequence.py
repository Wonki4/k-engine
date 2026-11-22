from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from kengine.cache.block_table import BlockTable


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 0.0  # 0 means greedy
    top_k: int = 0  # 0 disables
    top_p: float = 1.0
    seed: int | None = None
    ignore_eos: bool = False
    stop_token_ids: tuple[int, ...] = ()


@dataclass
class Sequence:
    """One generation request and the KV blocks it currently holds.

    The engine's only notion of progress is ``num_cached``: tokens whose K/V are in the pool.
    Each step feeds ``tokens[num_cached:]``. That is the whole prompt on admission, one token
    while decoding, and everything again after a preemption dropped the blocks, so prefill,
    decode, and resume are the same code path.
    """

    id: int
    prompt_ids: list[int]
    params: SamplingParams
    output_ids: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING
    table: BlockTable | None = None
    finish_reason: str | None = None
    rng: torch.Generator | None = None

    @property
    def tokens(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def num_cached(self) -> int:
        return self.table.num_tokens if self.table is not None else 0

    @property
    def num_pending(self) -> int:
        return self.num_tokens - self.num_cached
