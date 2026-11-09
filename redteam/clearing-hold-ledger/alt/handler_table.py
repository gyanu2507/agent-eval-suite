"""A correct replay written differently: a small Ledger class with one handler per event type."""

from __future__ import annotations

from typing import Any, Callable


class Ledger:
    def __init__(self) -> None:
        self.available: dict[str, float] = {}
        self.held: dict[str, float] = {}
        self.open_holds: dict[str, tuple[str, float]] = {}
        self.released: list[str] = []

    def _wallet(self, wid: str) -> None:
        self.available.setdefault(wid, 0.0)
        self.held.setdefault(wid, 0.0)

    def post(self, e: dict) -> bool:
        self._wallet(e["wallet"])
        self.available[e["wallet"]] += float(e["amount"])
        return True

    def draw(self, e: dict) -> bool:
        self._wallet(e["wallet"])
        self.available[e["wallet"]] -= float(e["amount"])
        return True

    def move(self, e: dict) -> bool:
        self._wallet(e["src"])
        self._wallet(e["dst"])
        self.available[e["src"]] -= float(e["amount"])
        self.available[e["dst"]] += float(e["amount"])
        return True

    def hold(self, e: dict) -> bool:
        wid, amount, key = e["wallet"], float(e["amount"]), e["hold_key"]
        self._wallet(wid)
        if key in self.open_holds or self.available[wid] < amount - 1e-12:
            return False
        self.available[wid] -= amount
        self.held[wid] += amount
        self.open_holds[key] = (wid, amount)
        return True

    def release(self, e: dict) -> bool:
        found = self.open_holds.pop(e["hold_key"], None)
        if found is None:
            return False
        wid, amount = found
        self._wallet(wid)
        self.held[wid] -= amount
        self.available[wid] += amount
        self.released.append(e["hold_key"])
        return True


def replay_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    ledger = Ledger()
    handlers: dict[str, Callable[[dict], bool]] = {
        "POST": ledger.post,
        "DRAW": ledger.draw,
        "MOVE": ledger.move,
        "HOLD": ledger.hold,
        "RELEASE": ledger.release,
    }
    done: set[str] = set()
    applied = []
    for e in sorted(events, key=lambda ev: int(ev["seq"])):
        key = str(e["dedupe_key"])
        handler = handlers.get(e["type"])
        if key in done or handler is None:
            continue
        if handler(e):
            done.add(key)
            applied.append({"seq": e["seq"], "type": e["type"], "dedupe_key": key})
    names = sorted(ledger.available)
    return {
        "available": {w: ledger.available[w] for w in names},
        "held": {w: ledger.held[w] for w in names},
        "applied": applied,
        "released": sorted(ledger.released),
    }
