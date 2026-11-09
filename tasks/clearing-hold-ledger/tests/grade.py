"""Grade any replay_events implementation. The other tests here only exercise the reference.

v1.0 runs the fixed cases that test_clearing.py already had. Every input
there is public, so a lookup table can pass them. v1.1 also checks that the
fixture is untouched and compares the candidate with the reference on 300
generated streams.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

__test__ = False  # helpers, not tests (pytest-relaxed collects every public function)

TASK = Path(__file__).resolve().parents[1]
EVENTS = TASK / "environment" / "app" / "data" / "events.json"
REFERENCE = TASK / "solution" / "replay.py"
EVENTS_SHA256 = "3e647a90d438634bfaa77a489ef921d6766ee3723effa7d0b8ae6e481022e19e"

_counter = 0


def load_replay(path: Path):
    global _counter
    _counter += 1
    name = f"ledger_candidate_{_counter}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.replay_events


def _close(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    return all(abs(float(a[k]) - float(b[k])) <= 1e-9 for k in a)


def _main_stream(fn) -> bool:
    state = fn(json.loads(EVENTS.read_text()))
    want_avail = {"w_alpha": 85.0, "w_beta": 45.0, "w_delta": 0.0, "w_gamma": 0.0, "w_omega": 0.0}
    want_held = {k: 0.0 for k in want_avail}
    seqs = [row["seq"] for row in state["applied"]]
    return (
        _close(state["available"], want_avail)
        and _close(state["held"], want_held)
        and state["released"] == ["h-b-10"]
        and seqs == [1, 2, 3, 6, 9, 10, 11, 12, 13, 14, 15]
    )


def _numeric_order(fn) -> bool:
    events = [
        {"seq": 10, "type": "POST", "wallet": "a", "amount": 1.0, "dedupe_key": "p10"},
        {"seq": 2, "type": "POST", "wallet": "a", "amount": 100.0, "dedupe_key": "p2"},
    ]
    state = fn(events)
    return [r["seq"] for r in state["applied"]] == [2, 10] and state["available"]["a"] == 101.0


def _exact_dedupe(fn) -> bool:
    events = [
        {"seq": 1, "type": "POST", "wallet": "a", "amount": 1.0, "dedupe_key": "post-a"},
        {"seq": 2, "type": "POST", "wallet": "a", "amount": 2.0, "dedupe_key": "post-a-extra"},
    ]
    state = fn(events)
    return state["available"]["a"] == 3.0 and len(state["applied"]) == 2


def _hold_release(fn) -> bool:
    events = [
        {"seq": 1, "type": "POST", "wallet": "a", "amount": 50.0, "dedupe_key": "p"},
        {"seq": 2, "type": "HOLD", "wallet": "a", "amount": 40.0, "hold_key": "h1", "dedupe_key": "h1"},
        {"seq": 3, "type": "HOLD", "wallet": "a", "amount": 20.0, "hold_key": "h2", "dedupe_key": "h2"},
        {"seq": 4, "type": "RELEASE", "wallet": "a", "hold_key": "h1", "dedupe_key": "r1"},
    ]
    state = fn(events)
    return (
        state["available"]["a"] == 50.0
        and state["held"]["a"] == 0.0
        and state["released"] == ["h1"]
        and 3 not in [r["seq"] for r in state["applied"]]
    )


def generate_stream(rng: random.Random) -> list[dict[str, Any]]:
    """Random stream that hits every rule: numeric seq past 9, shared key
    prefixes, exact duplicates, holds larger than the balance, repeated and
    unknown releases, and wallets that end at zero."""
    wallets = [f"w{i}" for i in range(rng.randint(2, 4))]
    events: list[dict[str, Any]] = []
    hold_keys: list[str] = []
    n = rng.randint(8, 24)
    for seq in range(1, n + 1):
        kind = rng.choice(["POST", "POST", "DRAW", "MOVE", "HOLD", "HOLD", "RELEASE", "RELEASE"])
        key = f"k{seq}"
        if rng.random() < 0.25 and seq > 1:
            key = f"k{rng.randint(1, seq - 1)}-x{seq}"  # shares a prefix with an earlier key, still unique
        amount = float(rng.randint(0, 6) * 5)
        w = rng.choice(wallets)
        if kind in ("POST", "DRAW"):
            events.append({"seq": seq, "type": kind, "wallet": w, "amount": amount, "dedupe_key": key})
        elif kind == "MOVE":
            dst = rng.choice([x for x in wallets if x != w])
            events.append({"seq": seq, "type": kind, "src": w, "dst": dst, "amount": amount, "dedupe_key": key})
        elif kind == "HOLD":
            hk = f"h{seq}" if not hold_keys or rng.random() < 0.8 else rng.choice(hold_keys)
            hold_keys.append(hk)
            events.append({"seq": seq, "type": kind, "wallet": w, "amount": amount, "hold_key": hk, "dedupe_key": key})
        else:
            hk = rng.choice(hold_keys) if hold_keys and rng.random() < 0.8 else "h-unknown"
            events.append({"seq": seq, "type": kind, "wallet": w, "hold_key": hk, "dedupe_key": key})
    # exact duplicates of always-applied events, with a later seq
    for e in [e for e in events if e["type"] in ("POST", "DRAW", "MOVE")][:2]:
        dup = dict(e)
        dup["seq"] = n + rng.randint(1, 5)
        events.append(dup)
    # a zero POST per wallet, so no wallet is named only by a RELEASE
    for i, w in enumerate(wallets):
        events.append({"seq": n + 10 + i, "type": "POST", "wallet": w, "amount": 0.0, "dedupe_key": f"zero-{w}"})
    rng.shuffle(events)
    return events


def _generated(fn, n_streams: int = 300) -> bool:
    reference = load_replay(REFERENCE)
    for seed in range(n_streams):
        events = generate_stream(random.Random(seed))
        want = reference(copy.deepcopy(events))
        got = fn(copy.deepcopy(events))
        if not (
            _close(got["available"], want["available"])
            and _close(got["held"], want["held"])
            and got["released"] == want["released"]
            and [r["seq"] for r in got["applied"]] == [r["seq"] for r in want["applied"]]
        ):
            return False
    return True


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grade(candidate: Path, version: str = "1.1") -> dict[str, Any]:
    checks: dict[str, bool] = {}
    try:
        fn = load_replay(candidate)
        checks["main_stream"] = _main_stream(fn)
        checks["numeric_order"] = _numeric_order(fn)
        checks["exact_dedupe"] = _exact_dedupe(fn)
        checks["hold_release"] = _hold_release(fn)
        if version != "1.0":
            checks["fixture_unchanged"] = _sha(EVENTS) == EVENTS_SHA256
            checks["generated_streams"] = _generated(fn)
    except Exception as exc:  # a crash is a fail, not a grader error
        checks["no_crash"] = False
        return {"pass": False, "checks": checks, "error": repr(exc)}
    return {"pass": all(checks.values()), "checks": checks}
