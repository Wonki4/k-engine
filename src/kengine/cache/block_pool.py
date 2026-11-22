from __future__ import annotations

from collections.abc import Callable

import torch


class OutOfBlocks(RuntimeError):
    """Raised when the pool is empty and nothing can be reclaimed."""


class BlockPool:
    """All KV memory the engine will ever use, allocated once and handed out in blocks.

    Storage layout: ``[num_layers, 2 (K/V), num_blocks, block_size, num_kv_heads, head_dim]``.
    A *slot* is a flat token index ``block_id * block_size + offset`` shared by every layer.

    Blocks are reference counted. A block returns to the free list when its count hits zero.
    When the free list is empty, ``reclaimer`` (the prefix cache's eviction hook) is asked to
    release a block before giving up.
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.kv = torch.zeros(
            num_layers,
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        # Flat per-layer views over the same memory: [2, num_blocks * block_size, kvh, hd].
        self._flat = self.kv.view(num_layers, 2, num_blocks * block_size, num_kv_heads, head_dim)
        # Pop from the end so low block ids are handed out first.
        self._free: list[int] = list(range(num_blocks - 1, -1, -1))
        self._ref: list[int] = [0] * num_blocks
        self.reclaimer: Callable[[], bool] | None = None

    # -- allocation ---------------------------------------------------------------------------

    def allocate(self) -> int:
        """Return a block with refcount 1."""
        while not self._free:
            if self.reclaimer is None or not self.reclaimer():
                raise OutOfBlocks(f"all {self.num_blocks} blocks are in use")
        block = self._free.pop()
        self._ref[block] = 1
        return block

    def incref(self, block: int) -> None:
        if self._ref[block] <= 0:
            raise ValueError(f"incref on free block {block}")
        self._ref[block] += 1

    def decref(self, block: int) -> None:
        if self._ref[block] <= 0:
            raise ValueError(f"decref on free block {block}")
        self._ref[block] -= 1
        if self._ref[block] == 0:
            self._free.append(block)

    def refcount(self, block: int) -> int:
        return self._ref[block]

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def check(self) -> None:
        """Assert the accounting invariant: every block is either free or referenced, never both."""
        free = set(self._free)
        assert len(free) == len(self._free), "duplicate block in free list"
        for block, ref in enumerate(self._ref):
            assert ref >= 0, f"negative refcount on block {block}"
            assert (ref == 0) == (block in free), f"block {block}: ref={ref}, free={block in free}"

    # -- data ---------------------------------------------------------------------------------

    def copy_block(self, src: int, dst: int) -> None:
        self.kv[:, :, dst] = self.kv[:, :, src]

    def write(self, layer: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store K/V for ``slots`` ([N]) from tensors shaped ``[N, num_kv_heads, head_dim]``."""
        flat = self._flat[layer]
        flat[0].index_copy_(0, slots, k)
        flat[1].index_copy_(0, slots, v)

    def read(self, layer: int, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for ``slots`` (any shape) -> ``[*slots.shape, num_kv_heads, head_dim]``."""
        flat = self._flat[layer]
        return flat[0][slots], flat[1][slots]
