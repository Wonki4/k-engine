from __future__ import annotations

from array import array
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from hashlib import blake2b

from kengine.cache.block_pool import BlockPool


class PrefixCache:
    """Reuses full KV blocks across requests that share a token prefix.

    A block's K/V depends on every token before it, so block ``i`` is keyed by a hash chain:
    ``key_i = H(key_{i-1}, tokens of block i)``. Only full blocks are cached, and the cache
    holds one reference on each. A block is evictable when that is the only reference left.

    LRU order is kept so that within one chain the deepest block is evicted first; evicting a
    shallow block first would strand its descendants (lookup walks the chain from the start).
    """

    def __init__(self, pool: BlockPool) -> None:
        self.pool = pool
        self._entries: OrderedDict[bytes, int] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def _keys(self, tokens: Sequence[int], num_blocks: int) -> Iterator[bytes]:
        bs = self.pool.block_size
        key = b""
        for i in range(num_blocks):
            chunk = array("q", tokens[i * bs : (i + 1) * bs]).tobytes()
            key = blake2b(key + chunk, digest_size=16).digest()
            yield key

    def _touch(self, keys: list[bytes]) -> None:
        for key in reversed(keys):
            self._entries.move_to_end(key)

    def lookup(self, tokens: Sequence[int], max_tokens: int | None = None) -> list[int]:
        """Blocks covering the longest cached prefix of ``tokens`` (at most ``max_tokens``).

        The caller receives one reference on each returned block.
        """
        limit = len(tokens) if max_tokens is None else min(max_tokens, len(tokens))
        keys: list[bytes] = []
        blocks: list[int] = []
        for key in self._keys(tokens, limit // self.pool.block_size):
            block = self._entries.get(key)
            if block is None:
                break
            keys.append(key)
            blocks.append(block)
        for block in blocks:
            self.pool.incref(block)
        self._touch(keys)
        return blocks

    def insert(self, tokens: Sequence[int], blocks: Sequence[int]) -> None:
        """Offer a sequence's blocks; full ones not yet cached are retained."""
        num_full = min(len(tokens) // self.pool.block_size, len(blocks))
        keys = list(self._keys(tokens, num_full))
        for key, block in zip(keys, blocks, strict=False):
            if key not in self._entries:
                self._entries[key] = block
                self.pool.incref(block)
        self._touch(keys)

    def evict_one(self) -> bool:
        """Release the least recently used block nobody else references. False if none."""
        for key, block in self._entries.items():
            if self.pool.refcount(block) == 1:
                del self._entries[key]
                self.pool.decref(block)
                return True
        return False

    def clear(self) -> None:
        for block in self._entries.values():
            self.pool.decref(block)
        self._entries.clear()
