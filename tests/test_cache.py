from __future__ import annotations

import pytest
import torch

from kengine.cache import BlockPool, BlockTable, OutOfBlocks, PrefixCache


def tiny_pool(num_blocks: int = 4, block_size: int = 4) -> BlockPool:
    return BlockPool(
        num_layers=2, num_blocks=num_blocks, block_size=block_size, num_kv_heads=1, head_dim=2
    )


# -- BlockPool ----------------------------------------------------------------------------------


def test_pool_allocates_until_empty_then_raises():
    pool = tiny_pool(num_blocks=3)
    blocks = [pool.allocate() for _ in range(3)]
    assert sorted(blocks) == [0, 1, 2]
    assert pool.num_free == 0
    with pytest.raises(OutOfBlocks):
        pool.allocate()
    pool.check()


def test_pool_block_is_freed_only_when_last_reference_drops():
    pool = tiny_pool()
    block = pool.allocate()
    pool.incref(block)
    pool.decref(block)
    assert pool.num_used == 1
    pool.decref(block)
    assert pool.num_used == 0
    with pytest.raises(ValueError):
        pool.decref(block)
    with pytest.raises(ValueError):
        pool.incref(block)
    pool.check()


def test_pool_asks_reclaimer_before_giving_up():
    pool = tiny_pool(num_blocks=1)
    held = [pool.allocate()]

    def reclaim() -> bool:
        if not held:
            return False
        pool.decref(held.pop())
        return True

    pool.reclaimer = reclaim
    assert pool.allocate() == 0  # the reclaimed block is handed out again
    with pytest.raises(OutOfBlocks):
        pool.allocate()


def test_pool_write_then_read_round_trips_per_layer():
    pool = tiny_pool()
    slots = torch.tensor([5, 2, 9])
    k, v = torch.randn(3, 1, 2), torch.randn(3, 1, 2)
    pool.write(1, slots, k, v)
    k_out, v_out = pool.read(1, slots.view(1, 3))
    assert torch.equal(k_out[0], k) and torch.equal(v_out[0], v)
    assert pool.kv[0].abs().sum() == 0  # the other layer is untouched
    # Slot 5 is block 1, offset 1.
    assert torch.equal(pool.kv[1, 0, 1, 1], k[0])


# -- BlockTable ---------------------------------------------------------------------------------


def test_table_reserve_allocates_only_missing_blocks():
    pool = tiny_pool(num_blocks=4, block_size=4)
    table = BlockTable(pool)
    table.reserve(5)
    assert len(table.blocks) == 2
    table.advance(5)
    table.reserve(3)  # fits in the second block
    assert len(table.blocks) == 2
    table.advance(3)
    table.reserve(1)
    assert len(table.blocks) == 3
    table.free()
    assert pool.num_free == 4
    pool.check()


def test_table_slots_follow_block_mapping():
    pool = tiny_pool(num_blocks=4, block_size=4)
    pool.allocate()  # take block 0 so the table gets non-trivial ids
    table = BlockTable(pool)
    table.reserve(6)
    assert table.blocks == [1, 2]
    assert table.slots(2, 6) == [6, 7, 8, 9]


def test_table_advance_past_capacity_is_rejected():
    table = BlockTable(tiny_pool(block_size=4))
    table.reserve(4)
    with pytest.raises(ValueError):
        table.advance(5)


def test_table_failed_reserve_leaves_table_and_pool_untouched():
    pool = tiny_pool(num_blocks=3, block_size=4)
    table = BlockTable(pool)
    table.reserve(4)
    table.advance(4)
    with pytest.raises(OutOfBlocks):
        table.reserve(12)  # needs 3 more, only 2 free
    assert len(table.blocks) == 1 and table.num_tokens == 4
    assert pool.num_free == 2
    pool.check()


def test_fork_shares_blocks_and_copies_tail_on_write():
    pool = tiny_pool(num_blocks=6, block_size=4)
    a = BlockTable(pool)
    a.reserve(6)
    a.advance(6)
    pool.kv[:, :, a.blocks[1]] = 7.0

    b = a.fork()
    assert b.blocks == a.blocks and pool.num_used == 2

    b.reserve(1)  # writes into the shared, half-full tail block
    assert b.blocks[0] == a.blocks[0], "full blocks stay shared"
    assert b.blocks[1] != a.blocks[1], "the tail was copied"
    assert torch.equal(pool.kv[:, :, b.blocks[1]], pool.kv[:, :, a.blocks[1]])
    assert pool.refcount(a.blocks[0]) == 2 and pool.refcount(a.blocks[1]) == 1

    a.reserve(1)  # a's tail is private again: no copy
    assert pool.num_used == 3

    a.free()
    b.free()
    assert pool.num_free == 6
    pool.check()


def test_fork_with_full_tail_needs_no_copy():
    pool = tiny_pool(num_blocks=4, block_size=4)
    a = BlockTable(pool)
    a.reserve(4)
    a.advance(4)
    b = a.fork()
    b.reserve(1)
    assert b.blocks[0] == a.blocks[0] and len(b.blocks) == 2
    assert pool.num_used == 2


def test_fork_copy_on_write_rolls_back_when_pool_is_full():
    pool = tiny_pool(num_blocks=1, block_size=4)
    a = BlockTable(pool)
    a.reserve(2)
    a.advance(2)
    b = a.fork()
    with pytest.raises(OutOfBlocks):
        b.reserve(1)
    assert b.blocks == a.blocks and pool.refcount(a.blocks[0]) == 2


# -- PrefixCache --------------------------------------------------------------------------------


def filled_table(pool: BlockPool, num_tokens: int) -> BlockTable:
    table = BlockTable(pool)
    table.reserve(num_tokens)
    table.advance(num_tokens)
    return table


def test_prefix_cache_returns_longest_matching_full_blocks():
    pool = tiny_pool(num_blocks=8, block_size=4)
    cache = PrefixCache(pool)
    tokens = list(range(10))  # two full blocks and a partial one
    table = filled_table(pool, 10)
    cache.insert(tokens, table.blocks)
    assert len(cache) == 2

    assert cache.lookup(tokens) == table.blocks[:2]
    assert cache.lookup(tokens[:4] + [99] * 6) == table.blocks[:1]
    assert cache.lookup([99] + tokens[1:]) == []
    # Same block contents at a different depth must not match: K/V depend on the full prefix.
    assert cache.lookup(tokens[4:8]) == []


def test_prefix_cache_lookup_respects_max_tokens():
    pool = tiny_pool(num_blocks=8, block_size=4)
    cache = PrefixCache(pool)
    tokens = list(range(8))
    table = filled_table(pool, 8)
    cache.insert(tokens, table.blocks)
    # With all 8 tokens cached, the last one must still be left for the model to compute.
    assert cache.lookup(tokens, max_tokens=7) == table.blocks[:1]


def test_prefix_cache_lookup_gives_caller_a_reference():
    pool = tiny_pool(num_blocks=8, block_size=4)
    cache = PrefixCache(pool)
    table = filled_table(pool, 4)
    cache.insert(list(range(4)), table.blocks)
    block = table.blocks[0]
    assert pool.refcount(block) == 2
    cache.lookup(list(range(4)))
    assert pool.refcount(block) == 3


def test_prefix_cache_evicts_only_unreferenced_blocks_deepest_first():
    pool = tiny_pool(num_blocks=8, block_size=4)
    cache = PrefixCache(pool)
    tokens = list(range(12))
    table = filled_table(pool, 12)
    blocks = list(table.blocks)
    cache.insert(tokens, blocks)

    assert not cache.evict_one(), "blocks still held by the table are not evictable"
    table.free()
    assert pool.num_used == 3

    assert cache.evict_one()
    assert cache.lookup(tokens) == blocks[:2], "the deepest block went first"
    for block in blocks[:2]:
        pool.decref(block)

    assert cache.evict_one() and cache.evict_one()
    assert not cache.evict_one()
    assert pool.num_free == 8
    pool.check()


def test_pool_reclaims_from_prefix_cache_under_pressure():
    pool = tiny_pool(num_blocks=2, block_size=4)
    cache = PrefixCache(pool)
    pool.reclaimer = cache.evict_one
    table = filled_table(pool, 8)
    cache.insert(list(range(8)), table.blocks)
    table.free()
    assert pool.num_free == 0

    fresh = filled_table(pool, 8)  # only possible by evicting both cached blocks
    assert len(cache) == 0
    fresh.free()
    assert pool.num_free == 2


def test_prefix_cache_clear_releases_everything():
    pool = tiny_pool(num_blocks=4, block_size=4)
    cache = PrefixCache(pool)
    table = filled_table(pool, 8)
    cache.insert(list(range(8)), table.blocks)
    table.free()
    cache.clear()
    assert pool.num_free == 4 and len(cache) == 0
