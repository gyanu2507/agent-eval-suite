"""Broken starter: single-cursor ring that overwrites on full."""

from __future__ import annotations


class SlotRing:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.slots: list[dict | None] = [None] * capacity
        self.head = 0  # next write
        self.tail = 0  # next read (shared — BUG)
        self.size = 0
        self.writes_ok = 0
        self.writes_dropped = 0
        self.reads_a = 0
        self.reads_b = 0

    def write(self, seq: int, payload: bytes) -> str:
        # BUG: overwrites oldest instead of dropping incoming when full
        if self.size == self.capacity:
            self.tail = (self.tail + 1) % self.capacity
            self.size -= 1
            self.writes_dropped += 1
            # still writes — wrong policy
        self.slots[self.head] = {"seq": seq, "payload": payload}
        self.head = (self.head + 1) % self.capacity
        self.size += 1
        self.writes_ok += 1
        return "ok"

    def read(self, consumer: str) -> dict | None:
        # BUG: ignores consumer identity; shared cursor
        if self.size == 0:
            return None
        pkt = self.slots[self.tail]
        self.slots[self.tail] = None
        self.tail = (self.tail + 1) % self.capacity
        self.size -= 1
        if consumer == "A":
            self.reads_a += 1
        else:
            self.reads_b += 1
        return pkt

    def stats(self) -> dict:
        return {
            "writes_ok": self.writes_ok,
            "writes_dropped": self.writes_dropped,
            "reads_a": self.reads_a,
            "reads_b": self.reads_b,
            "occupancy": self.size,
        }
