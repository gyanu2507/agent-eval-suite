"""Broken starter: intentional bugs for the agent to repair.

Known failures (do not leave these in a solution):
- sorts seq as strings ("10" before "2")
- prefix match on dedupe_key (false collisions)
- RELEASE matching by wallet instead of hold_key
- partial HOLD when funds are short
- drops wallets that end at zero
"""

from __future__ import annotations

from typing import Any


def replay_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    available: dict[str, float] = {}
    held: dict[str, float] = {}
    holds: dict[str, tuple[str, float]] = {}
    applied: list[dict[str, Any]] = []
    released: list[str] = []
    seen: set[str] = set()

    # BUG: lexicographic sort on seq
    ordered = sorted(events, key=lambda e: str(e["seq"]))

    for event in ordered:
        dedupe = str(event["dedupe_key"])
        # BUG: prefix collision instead of exact key
        if any(dedupe.startswith(s) or s.startswith(dedupe) for s in seen):
            continue
        seen.add(dedupe)

        etype = event["type"]
        if etype == "POST":
            wallet = event["wallet"]
            available[wallet] = available.get(wallet, 0.0) + float(event["amount"])
        elif etype == "DRAW":
            wallet = event["wallet"]
            available[wallet] = available.get(wallet, 0.0) - float(event["amount"])
        elif etype == "MOVE":
            src, dst = event["src"], event["dst"]
            amount = float(event["amount"])
            available[src] = available.get(src, 0.0) - amount
            available[dst] = available.get(dst, 0.0) + amount
        elif etype == "HOLD":
            wallet = event["wallet"]
            amount = float(event["amount"])
            hold_key = event["hold_key"]
            # BUG: partial hold when short
            take = min(amount, available.get(wallet, 0.0))
            if take <= 0:
                continue
            available[wallet] = available.get(wallet, 0.0) - take
            held[wallet] = held.get(wallet, 0.0) + take
            holds[hold_key] = (wallet, take)
        elif etype == "RELEASE":
            hold_key = event["hold_key"]
            # BUG: match by wallet field, ignore hold_key identity
            wallet = event.get("wallet")
            victim = None
            for hk, (w, amt) in list(holds.items()):
                if wallet is not None and w == wallet:
                    victim = (hk, w, amt)
                    break
            if victim is None:
                continue
            hk, w, amt = victim
            del holds[hk]
            held[w] = held.get(w, 0.0) - amt
            available[w] = available.get(w, 0.0) + amt
            released.append(hk)
        else:
            continue

        applied.append(
            {
                "seq": event["seq"],
                "type": etype,
                "dedupe_key": dedupe,
            }
        )

    # BUG: drop zero wallets
    available = {k: v for k, v in available.items() if abs(v) > 1e-12}
    held = {k: v for k, v in held.items() if abs(v) > 1e-12}

    return {
        "available": available,
        "held": held,
        "applied": applied,
        "released": sorted(released),
    }
