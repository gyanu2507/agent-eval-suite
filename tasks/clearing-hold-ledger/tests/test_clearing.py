from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
APP = TASK / "environment" / "app"
EVENTS = APP / "data" / "events.json"
SOLUTION = TASK / "solution" / "replay.py"
STARTER = APP / "clearing" / "replay.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _almost(a: dict, b: dict, tol: float = 1e-9) -> None:
    assert set(a) == set(b)
    for k in a:
        assert abs(float(a[k]) - float(b[k])) <= tol, (k, a[k], b[k])


def test_starter_fails_oracle_expectations():
    events = json.loads(EVENTS.read_text())
    oracle = _load(SOLUTION, "oracle_replay").replay_events(events)
    starter = _load(STARTER, "starter_replay").replay_events(events)
    assert starter != oracle


def test_oracle_balances_and_holds():
    events = json.loads(EVENTS.read_text())
    state = _load(SOLUTION, "oracle_replay2").replay_events(events)

    _almost(
        state["available"],
        {
            "w_alpha": 85.0,
            "w_beta": 45.0,
            "w_delta": 0.0,
            "w_gamma": 0.0,
            "w_omega": 0.0,
        },
    )
    _almost(
        state["held"],
        {
            "w_alpha": 0.0,
            "w_beta": 0.0,
            "w_delta": 0.0,
            "w_gamma": 0.0,
            "w_omega": 0.0,
        },
    )
    assert state["released"] == ["h-b-10"]

    applied_seqs = [row["seq"] for row in state["applied"]]
    assert applied_seqs == [1, 2, 3, 6, 9, 10, 11, 12, 13, 14, 15]
    assert "post-alpha-100" in {row["dedupe_key"] for row in state["applied"]}
    # Duplicate seq=16 must not appear
    assert 16 not in applied_seqs
    # Failed early HOLDs must not appear
    assert 4 not in applied_seqs
    assert 5 not in applied_seqs


def test_numeric_seq_ordering_not_lexicographic():
    events = [
        {"seq": 10, "type": "POST", "wallet": "a", "amount": 1.0, "dedupe_key": "p10"},
        {"seq": 2, "type": "POST", "wallet": "a", "amount": 100.0, "dedupe_key": "p2"},
    ]
    state = _load(SOLUTION, "oracle_order").replay_events(events)
    assert [r["seq"] for r in state["applied"]] == [2, 10]
    assert state["available"]["a"] == 101.0


def test_exact_dedupe_not_prefix():
    events = [
        {"seq": 1, "type": "POST", "wallet": "a", "amount": 1.0, "dedupe_key": "post-a"},
        {
            "seq": 2,
            "type": "POST",
            "wallet": "a",
            "amount": 2.0,
            "dedupe_key": "post-a-extra",
        },
    ]
    state = _load(SOLUTION, "oracle_dedupe").replay_events(events)
    assert state["available"]["a"] == 3.0
    assert len(state["applied"]) == 2


def test_hold_is_all_or_nothing_and_release_by_key():
    events = [
        {"seq": 1, "type": "POST", "wallet": "a", "amount": 50.0, "dedupe_key": "p"},
        {
            "seq": 2,
            "type": "HOLD",
            "wallet": "a",
            "amount": 40.0,
            "hold_key": "h1",
            "dedupe_key": "h1",
        },
        {
            "seq": 3,
            "type": "HOLD",
            "wallet": "a",
            "amount": 20.0,
            "hold_key": "h2",
            "dedupe_key": "h2",
        },
        {
            "seq": 4,
            "type": "RELEASE",
            "wallet": "a",
            "hold_key": "h1",
            "dedupe_key": "r1",
        },
    ]
    state = _load(SOLUTION, "oracle_hold").replay_events(events)
    assert state["available"]["a"] == 50.0
    assert state["held"]["a"] == 0.0
    assert state["released"] == ["h1"]
    assert 3 not in [r["seq"] for r in state["applied"]]
