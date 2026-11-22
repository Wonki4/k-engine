from __future__ import annotations

import torch

from kengine.sequence import SamplingParams

MIN_TEMPERATURE = 1e-5


def sample(logits: torch.Tensor, params: SamplingParams, rng: torch.Generator | None = None) -> int:
    """Pick the next token from one sequence's logits ([V], fp32 on CPU)."""
    # Dividing by a near-zero temperature overflows to inf and turns softmax into NaN, so
    # anything this small is treated as what it means: greedy.
    if params.temperature < MIN_TEMPERATURE:
        return int(logits.argmax())

    logits = logits / params.temperature
    if 0 < params.top_k < logits.shape[-1]:
        kth = torch.topk(logits, params.top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)

    if params.top_p < 1.0:
        sorted_probs, order = torch.sort(probs, descending=True)
        # Keep the smallest set of tokens whose mass reaches top_p (always at least one).
        mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
        sorted_probs = sorted_probs.masked_fill(mass_before >= params.top_p, 0.0)
        probs = torch.zeros_like(probs).scatter_(0, order, sorted_probs)
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, 1, generator=rng))
