from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize in fp32 so half-precision activations do not overflow when squared.
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


class MLP(nn.Module):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def rope_cos_sin(
    positions: torch.Tensor, head_dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary tables for absolute ``positions`` ([B, T]) -> cos/sin of shape [B, T, head_dim]."""
    exponent = torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (exponent / head_dim))
    freqs = positions[..., None].float() * inv_freq
    angles = torch.cat((freqs, freqs), dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ([B, T, heads, head_dim]) by its position's angles."""
    cos = cos[:, :, None, :].to(x.dtype)
    sin = sin[:, :, None, :].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin
