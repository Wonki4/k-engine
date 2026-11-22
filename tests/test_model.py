from __future__ import annotations

import pytest
import torch

from kengine.cache import BlockTable
from kengine.engine import build_batch

ATOL = 1e-4


@pytest.fixture(scope="module")
def hf_logits(hf_model):
    def run(ids: list[int]) -> torch.Tensor:
        with torch.inference_mode():
            return hf_model(torch.tensor([ids])).logits[0]

    return run


def feed(model, pool, table: BlockTable, tokens: list[int]) -> torch.Tensor:
    """Feed ``tokens`` on top of what ``table`` already caches; logits for each ([T, V])."""
    table.reserve(len(tokens))
    with torch.inference_mode():
        logits = model(build_batch([(table, tokens)]), pool, last_only=False)[0]
    table.advance(len(tokens))
    return logits


def test_prefill_logits_match_hf(model, make_pool, prompt_ids, hf_logits):
    pool = make_pool()
    for ids in prompt_ids:
        table = BlockTable(pool)
        logits = feed(model, pool, table, ids)
        torch.testing.assert_close(logits, hf_logits(ids), atol=ATOL, rtol=0)
        table.free()


def test_incremental_decoding_matches_hf(model, make_pool, prompt_ids, hf_logits):
    """Prefill part of the prompt, then one token at a time across several block boundaries."""
    ids = prompt_ids[3]
    reference = hf_logits(ids)
    pool = make_pool()
    table = BlockTable(pool)
    split = 5
    rows = [feed(model, pool, table, ids[:split])]
    rows += [feed(model, pool, table, [token]) for token in ids[split:]]
    assert len(table.blocks) > 2
    torch.testing.assert_close(torch.cat(rows), reference, atol=ATOL, rtol=0)


def test_ragged_batch_matches_hf(model, make_pool, prompt_ids, hf_logits):
    """Sequences of different lengths and different cached amounts in one forward pass."""
    pool = make_pool()
    tables = [BlockTable(pool) for _ in prompt_ids]
    cached = [0, 7, 3, 20]
    for table, ids, n in zip(tables, prompt_ids, cached, strict=True):
        if n:
            feed(model, pool, table, ids[:n])

    items = [(t, ids[n:]) for t, ids, n in zip(tables, prompt_ids, cached, strict=True)]
    for table, new in items:
        table.reserve(len(new))
    with torch.inference_mode():
        batch = build_batch(items)
        logits = model(batch, pool, last_only=False)
        last = model(batch, pool, last_only=True)

    for b, (ids, n) in enumerate(zip(prompt_ids, cached, strict=True)):
        reference = hf_logits(ids)
        torch.testing.assert_close(logits[b, : len(ids) - n], reference[n:], atol=ATOL, rtol=0)
        torch.testing.assert_close(last[b], reference[-1], atol=ATOL, rtol=0)


def test_forked_sequences_diverge_without_corrupting_each_other(
    model, make_pool, prompt_ids, hf_logits
):
    pool = make_pool()
    prefix = prompt_ids[1][:11]  # not a multiple of the block size: the tail block is shared
    a = BlockTable(pool)
    feed(model, pool, a, prefix)
    b = a.fork()
    used_before = pool.num_used

    cont_a, cont_b = prompt_ids[0], prompt_ids[2]
    logits_a = feed(model, pool, a, cont_a)
    logits_b = feed(model, pool, b, cont_b)
    logits_a2 = feed(model, pool, a, cont_b[:3])  # a keeps going after b wrote

    assert a.blocks[0] != b.blocks[0], "the shared tail was copied on write"
    assert pool.num_used > used_before
    ref_a = hf_logits(prefix + cont_a + cont_b[:3])
    ref_b = hf_logits(prefix + cont_b)
    torch.testing.assert_close(
        torch.cat([logits_a, logits_a2]), ref_a[len(prefix) :], atol=ATOL, rtol=0
    )
    torch.testing.assert_close(logits_b, ref_b[len(prefix) :], atol=ATOL, rtol=0)

    a.free()
    b.free()
    assert pool.num_free == pool.num_blocks
    pool.check()
