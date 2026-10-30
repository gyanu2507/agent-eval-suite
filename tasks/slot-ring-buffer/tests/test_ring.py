from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
STARTER = TASK / "environment" / "app" / "ring.py"
SOLUTION = TASK / "solution" / "ring.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_starter_diverges_on_dual_consumer_drop_policy():
    starter = _load(STARTER, "starter_ring").SlotRing(2)
    oracle = _load(SOLUTION, "oracle_ring").SlotRing(2)

    for ring in (starter, oracle):
        assert ring.write(1, b"a") == "ok"
        assert ring.write(2, b"b") == "ok"

    assert starter.write(3, b"c") == "ok"  # overwrites
    assert oracle.write(3, b"c") == "dropped"

    assert starter.stats()["writes_dropped"] >= 1
    assert oracle.stats()["writes_dropped"] == 1
    assert oracle.stats()["writes_ok"] == 2


def test_dual_cursors_independent():
    ring = _load(SOLUTION, "oracle_dual").SlotRing(4)
    for i in range(3):
        assert ring.write(i, bytes([i])) == "ok"

    a0 = ring.read("A")
    b0 = ring.read("B")
    assert a0 == b0 == {"seq": 0, "payload": b"\x00"}

    a1 = ring.read("A")
    assert a1["seq"] == 1
    # write_pos=3, min(A=2,B=1)=1 → occupancy 2 (slot 0 freed for both)
    assert ring.stats()["occupancy"] == 2

    assert ring.read("B")["seq"] == 1
    assert ring.read("B")["seq"] == 2
    assert ring.read("A")["seq"] == 2
    assert ring.stats()["occupancy"] == 0
    assert ring.read("A") is None
    assert ring.read("B") is None


def test_slow_consumer_blocks_reuse_until_caught_up():
    ring = _load(SOLUTION, "oracle_slow").SlotRing(2)
    assert ring.write(10, b"x") == "ok"
    assert ring.write(11, b"y") == "ok"
    assert ring.write(12, b"z") == "dropped"

    assert ring.read("A")["seq"] == 10
    # B still at start; still full
    assert ring.write(13, b"w") == "dropped"
    assert ring.read("B")["seq"] == 10
    # now one slot free
    assert ring.write(13, b"w") == "ok"
    assert ring.stats()["writes_ok"] == 3
    assert ring.stats()["writes_dropped"] == 2


def test_stats_counters():
    ring = _load(SOLUTION, "oracle_stats").SlotRing(3)
    ring.write(1, b"a")
    ring.write(2, b"b")
    ring.write(3, b"c")
    ring.write(4, b"d")
    ring.read("A")
    ring.read("A")
    ring.read("B")
    s = ring.stats()
    assert s == {
        "writes_ok": 3,
        "writes_dropped": 1,
        "reads_a": 2,
        "reads_b": 1,
        "occupancy": 2,
    }
