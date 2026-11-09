"""Correct except a HOLD bigger than the balance holds whatever is there."""

from __future__ import annotations

from typing import Any


def replay_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    available: dict[str, float] = {}
    held: dict[str, float] = {}
    holds: dict[str, tuple[str, float]] = {}
    applied: list[dict[str, Any]] = []
    released: list[str] = []
    seen: set[str] = set()
    wallets: set[str] = set()

    def touch(*ids: str) -> None:
        for wid in ids:
            wallets.add(wid)
            available.setdefault(wid, 0.0)
            held.setdefault(wid, 0.0)

    ordered = sorted(events, key=lambda e: int(e["seq"]))

    for event in ordered:
        dedupe = str(event["dedupe_key"])
        if dedupe in seen:
            continue

        etype = event["type"]
        did = False

        if etype == "POST":
            wallet = event["wallet"]
            touch(wallet)
            available[wallet] += float(event["amount"])
            did = True
        elif etype == "DRAW":
            wallet = event["wallet"]
            touch(wallet)
            available[wallet] -= float(event["amount"])
            did = True
        elif etype == "MOVE":
            src, dst = event["src"], event["dst"]
            touch(src, dst)
            amount = float(event["amount"])
            available[src] -= amount
            available[dst] += amount
            did = True
        elif etype == "HOLD":
            wallet = event["wallet"]
            amount = float(event["amount"])
            hold_key = event["hold_key"]
            touch(wallet)
            if hold_key in holds:
                continue
            amount = min(amount, available[wallet])
            if amount <= 0:
                continue
            available[wallet] -= amount
            held[wallet] += amount
            holds[hold_key] = (wallet, amount)
            did = True
        elif etype == "RELEASE":
            hold_key = event["hold_key"]
            if hold_key not in holds:
                continue
            wallet, amount = holds.pop(hold_key)
            touch(wallet)
            held[wallet] -= amount
            available[wallet] += amount
            released.append(hold_key)
            did = True
        else:
            continue

        if not did:
            continue

        seen.add(dedupe)
        applied.append(
            {
                "seq": event["seq"],
                "type": etype,
                "dedupe_key": dedupe,
            }
        )

    for wid in wallets:
        available.setdefault(wid, 0.0)
        held.setdefault(wid, 0.0)

    return {
        "available": {k: available[k] for k in sorted(wallets)},
        "held": {k: held[k] for k in sorted(wallets)},
        "applied": applied,
        "released": sorted(released),
    }
