from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from kengine.cache import BlockPool, BlockTable, OutOfBlocks, PrefixCache
from kengine.sequence import SeqStatus, Sequence


@dataclass
class SchedulerStats:
    prefix_hit_tokens: int = 0
    preemptions: int = 0


class Scheduler:
    """Decides which sequences run each step, using KV blocks as the budget.

    First come, first served. ``running`` and ``waiting`` are both kept in arrival order, with
    everything in ``running`` older than everything in ``waiting``. When blocks run out, the
    newest running sequence is preempted: its blocks are released (full ones stay reachable
    through the prefix cache until evicted) and it goes back to the front of the queue. The
    oldest sequence is never the victim while others run, so some request always progresses.
    """

    def __init__(self, pool: BlockPool, prefix_cache: PrefixCache, max_batch_size: int) -> None:
        self.pool = pool
        self.prefix_cache = prefix_cache
        self.max_batch_size = max_batch_size
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.stats = SchedulerStats()

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> list[Sequence]:
        """Reserve blocks for this step and return the sequences that will run in it."""
        # Running sequences first, oldest first, so older requests get blocks before newer ones.
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            try:
                seq.table.reserve(seq.num_pending)
                i += 1
            except OutOfBlocks:
                # May be ``seq`` itself, in which case the loop ends here.
                self._preempt(self.running[-1])

        # Then admit from the queue while blocks last. Never preempt to admit.
        while self.waiting and len(self.running) < self.max_batch_size:
            if not self._admit(self.waiting[0]):
                break
            self.running.append(self.waiting.popleft())
        return list(self.running)

    def finish(self, seq: Sequence, reason: str) -> None:
        if seq.status is SeqStatus.RUNNING:
            self.running.remove(seq)
            self._release(seq)
        elif seq.status is SeqStatus.WAITING:
            self.waiting.remove(seq)
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = reason

    def _admit(self, seq: Sequence) -> bool:
        tokens = seq.tokens
        # The last token must always be computed: its logits pick the next token.
        blocks = self.prefix_cache.lookup(tokens, max_tokens=len(tokens) - 1)
        table = BlockTable(self.pool, blocks, len(blocks) * self.pool.block_size)
        try:
            table.reserve(len(tokens) - table.num_tokens)
        except OutOfBlocks:
            table.free()
            return False
        seq.table = table
        seq.status = SeqStatus.RUNNING
        self.stats.prefix_hit_tokens += table.num_tokens
        return True

    def _preempt(self, seq: Sequence) -> None:
        self.running.remove(seq)
        self._release(seq)
        seq.status = SeqStatus.WAITING
        self.waiting.appendleft(seq)
        self.stats.preemptions += 1

    def _release(self, seq: Sequence) -> None:
        """Drop a sequence's blocks, offering the full ones to the prefix cache first."""
        table = seq.table
        self.prefix_cache.insert(seq.tokens[: table.num_tokens], table.blocks)
        table.free()
        seq.table = None
