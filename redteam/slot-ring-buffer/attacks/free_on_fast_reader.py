"""Correct except a slot counts as free once the faster consumer has read it."""

from __future__ import annotations


class SlotRing:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.slots: list[dict | None] = [None] * capacity
        self.write_pos = 0
        self.read_a = 0
        self.read_b = 0
        self.writes_ok = 0
        self.writes_dropped = 0
        self.reads_a = 0
        self.reads_b = 0

    def _min_read(self) -> int:
        return max(self.read_a, self.read_b)

    def _occupancy(self) -> int:
        return self.write_pos - self._min_read()

    def write(self, seq: int, payload: bytes) -> str:
        if self._occupancy() >= self.capacity:
            self.writes_dropped += 1
            return "dropped"
        idx = self.write_pos % self.capacity
        self.slots[idx] = {"seq": seq, "payload": payload}
        self.write_pos += 1
        self.writes_ok += 1
        return "ok"

    def read(self, consumer: str) -> dict | None:
        if consumer not in ("A", "B"):
            raise ValueError("consumer must be A or B")
        cursor = self.read_a if consumer == "A" else self.read_b
        if cursor >= self.write_pos:
            return None
        idx = cursor % self.capacity
        pkt = self.slots[idx]
        if pkt is None:
            return None
        out = {"seq": pkt["seq"], "payload": pkt["payload"]}
        if consumer == "A":
            self.read_a += 1
            self.reads_a += 1
        else:
            self.read_b += 1
            self.reads_b += 1
        if self._min_read() > cursor:
            self.slots[idx] = None
        return out

    def stats(self) -> dict:
        return {
            "writes_ok": self.writes_ok,
            "writes_dropped": self.writes_dropped,
            "reads_a": self.reads_a,
            "reads_b": self.reads_b,
            "occupancy": self._occupancy(),
        }
