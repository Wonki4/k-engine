from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn

from kengine.config import ModelConfig
from kengine.model.layers import apply_rope


class KVStore(Protocol):
    """Where attention keeps K/V. ``BlockPool`` is the implementation.

    The gather-then-attend read path below is the portable baseline. A custom kernel that
    attends over blocks in place would replace ``read`` + SDPA behind this same boundary.
    """

    def write(self, layer: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None: ...

    def read(self, layer: int, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass
class PagedBatch:
    """One forward pass over B sequences, each feeding some new tokens on top of cached ones.

    T is the longest run of new tokens in the batch, S the longest total context.
    """

    tokens: torch.Tensor  # [B, T] new token ids, right-padded
    positions: torch.Tensor  # [B, T] absolute positions of the new tokens
    write_mask: torch.Tensor  # [B, T] True where a real (non-padding) new token sits
    write_slots: torch.Tensor  # [N] pool slots for the real new tokens, row-major order
    read_slots: torch.Tensor  # [B, S] pool slots of each sequence's full context, right-padded
    attn_mask: torch.Tensor  # [B, T, S] True where new token t may attend to context slot s
    last_index: torch.Tensor  # [B] index along T of each sequence's last real new token


class PagedAttention(nn.Module):
    """Causal self-attention whose K/V live in an external block pool, not in the module."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        batch: PagedBatch,
        kv: KVStore,
        layer: int,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = apply_rope(self.q_proj(x).view(B, T, self.num_heads, self.head_dim), cos, sin)
        k = apply_rope(self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim), cos, sin)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # Write the new tokens first, then read the whole context back through the block
        # table. Rotary positions are baked into K, so cached K never needs recomputing.
        kv.write(layer, batch.write_slots, k[batch.write_mask], v[batch.write_mask])
        k_ctx, v_ctx = kv.read(layer, batch.read_slots)  # [B, S, kv_heads, head_dim]

        groups = self.num_heads // self.num_kv_heads
        k_ctx = k_ctx.transpose(1, 2).repeat_interleave(groups, dim=1)  # [B, heads, S, head_dim]
        v_ctx = v_ctx.transpose(1, 2).repeat_interleave(groups, dim=1)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k_ctx, v_ctx, attn_mask=batch.attn_mask[:, None]
        )
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))
