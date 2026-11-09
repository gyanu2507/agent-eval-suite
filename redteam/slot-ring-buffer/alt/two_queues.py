"""A correct ring written differently: each consumer gets its own queue of
pending packets, and the ring is full when the longer queue reaches capacity."""

from __future__ import annotations

from collections import deque


class SlotRing:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.pending = {"A": deque(), "B": deque()}
        self.counts = {"writes_ok": 0, "writes_dropped": 0, "reads_a": 0, "reads_b": 0}

    def _occupancy(self) -> int:
        return max(len(q) for q in self.pending.values())

    def write(self, seq: int, payload: bytes) -> str:
        if self._occupancy() >= self.capacity:
            self.counts["writes_dropped"] += 1
            return "dropped"
        for q in self.pending.values():
            q.append((seq, payload))
        self.counts["writes_ok"] += 1
        return "ok"

    def read(self, consumer: str) -> dict | None:
        if consumer not in self.pending:
            raise ValueError("consumer must be A or B")
        q = self.pending[consumer]
        if not q:
            return None
        seq, payload = q.popleft()
        self.counts["reads_a" if consumer == "A" else "reads_b"] += 1
        return {"seq": seq, "payload": payload}

    def stats(self) -> dict:
        return {**self.counts, "occupancy": self._occupancy()}
