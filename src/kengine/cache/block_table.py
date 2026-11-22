from __future__ import annotations

from kengine.cache.block_pool import BlockPool, OutOfBlocks


class BlockTable:
    """Maps one sequence's logical token positions onto physical blocks in a ``BlockPool``.

    ``blocks[i]`` holds positions ``[i * block_size, (i + 1) * block_size)``. The table owns one
    reference on each of its blocks. ``num_tokens`` is how many positions hold valid K/V.
    """

    def __init__(
        self, pool: BlockPool, blocks: list[int] | None = None, num_tokens: int = 0
    ) -> None:
        self.pool = pool
        self.blocks: list[int] = blocks if blocks is not None else []
        self.num_tokens = num_tokens
        if num_tokens > self.capacity:
            raise ValueError("num_tokens exceeds the capacity of the given blocks")

    @property
    def capacity(self) -> int:
        return len(self.blocks) * self.pool.block_size

    def reserve(self, n_new: int) -> None:
        """Make room to write ``n_new`` tokens after ``num_tokens``.

        Allocates missing blocks, and if the block that will be written next is shared (after a
        ``fork``), replaces it with a private copy first (copy-on-write). Either fully succeeds
        or raises ``OutOfBlocks`` leaving the table untouched.
        """
        if n_new <= 0:
            return
        pool, bs = self.pool, self.pool.block_size
        tail_shared = self.num_tokens % bs != 0 and pool.refcount(self.blocks[-1]) > 1
        missing = max(0, -(-(self.num_tokens + n_new) // bs) - len(self.blocks))
        fresh: list[int] = []
        try:
            for _ in range(missing + int(tail_shared)):
                fresh.append(pool.allocate())
        except OutOfBlocks:
            for block in fresh:
                pool.decref(block)
            raise
        if tail_shared:
            private, old = fresh.pop(), self.blocks[-1]
            pool.copy_block(old, private)
            pool.decref(old)
            self.blocks[-1] = private
        self.blocks.extend(fresh)

    def advance(self, n: int) -> None:
        """Mark ``n`` more positions as holding valid K/V."""
        if self.num_tokens + n > self.capacity:
            raise ValueError("advance past reserved capacity")
        self.num_tokens += n

    def slots(self, start: int, end: int) -> list[int]:
        """Flat pool slots for logical positions ``[start, end)``."""
        bs, blocks = self.pool.block_size, self.blocks
        return [blocks[pos // bs] * bs + pos % bs for pos in range(start, end)]

    def fork(self) -> BlockTable:
        """A second table sharing every block. No K/V is copied until one side writes."""
        for block in self.blocks:
            self.pool.incref(block)
        return BlockTable(self.pool, list(self.blocks), self.num_tokens)

    def free(self) -> None:
        for block in self.blocks:
            self.pool.decref(block)
        self.blocks = []
        self.num_tokens = 0
