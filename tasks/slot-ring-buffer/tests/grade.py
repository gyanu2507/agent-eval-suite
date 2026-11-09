"""Grade any SlotRing implementation. The other tests here only exercise the reference.

v1.0 runs the four scripted scenarios that test_ring.py already had. v1.1 also
replays 500 random write/read sequences against the reference and compares
every return value and every stats() call.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Any

__test__ = False  # helpers, not tests (pytest-relaxed collects every public function)

TASK = Path(__file__).resolve().parents[1]
REFERENCE = TASK / "solution" / "ring.py"

_counter = 0


def load_ring(path: Path):
    global _counter
    _counter += 1
    name = f"ring_candidate_{_counter}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.SlotRing


def _drop_policy(ring_cls) -> bool:
    ring = ring_cls(2)
    return (
        ring.write(1, b"a") == "ok"
        and ring.write(2, b"b") == "ok"
        and ring.write(3, b"c") == "dropped"
        and ring.stats()["writes_dropped"] == 1
        and ring.stats()["writes_ok"] == 2
    )


def _dual_cursors(ring_cls) -> bool:
    ring = ring_cls(4)
    for i in range(3):
        if ring.write(i, bytes([i])) != "ok":
            return False
    a0, b0 = ring.read("A"), ring.read("B")
    if not (a0 == b0 == {"seq": 0, "payload": b"\x00"}):
        return False
    if ring.read("A")["seq"] != 1 or ring.stats()["occupancy"] != 2:
        return False
    if ring.read("B")["seq"] != 1 or ring.read("B")["seq"] != 2 or ring.read("A")["seq"] != 2:
        return False
    return ring.stats()["occupancy"] == 0 and ring.read("A") is None and ring.read("B") is None


def _slow_consumer(ring_cls) -> bool:
    ring = ring_cls(2)
    ok = ring.write(10, b"x") == "ok" and ring.write(11, b"y") == "ok" and ring.write(12, b"z") == "dropped"
    ok = ok and ring.read("A")["seq"] == 10 and ring.write(13, b"w") == "dropped"
    ok = ok and ring.read("B")["seq"] == 10 and ring.write(13, b"w") == "ok"
    s = ring.stats()
    return ok and s["writes_ok"] == 3 and s["writes_dropped"] == 2


def _stats(ring_cls) -> bool:
    ring = ring_cls(3)
    for i in range(1, 5):
        ring.write(i, bytes([i]))
    ring.read("A")
    ring.read("A")
    ring.read("B")
    return ring.stats() == {"writes_ok": 3, "writes_dropped": 1, "reads_a": 2, "reads_b": 1, "occupancy": 2}


def _random_ops(ring_cls, n_runs: int = 500) -> bool:
    reference = load_ring(REFERENCE)
    for seed in range(n_runs):
        rng = random.Random(seed)
        cap = rng.randint(1, 6)
        ref, got = reference(cap), ring_cls(cap)
        seq = rng.randint(0, 2**16)
        for _ in range(rng.randint(10, 80)):
            op = rng.random()
            if op < 0.45:
                seq = (seq + 1) % 2**16  # seq wraps, per contract rule 8
                payload = bytes([rng.randint(0, 255)])
                if ref.write(seq, payload) != got.write(seq, payload):
                    return False
            elif op < 0.9:
                who = rng.choice("AB")
                if ref.read(who) != got.read(who):
                    return False
            elif ref.stats() != got.stats():
                return False
        if ref.stats() != got.stats():
            return False
    return True


def grade(candidate: Path, version: str = "1.1") -> dict[str, Any]:
    checks: dict[str, bool] = {}
    try:
        ring_cls = load_ring(candidate)
        checks["drop_policy"] = _drop_policy(ring_cls)
        checks["dual_cursors"] = _dual_cursors(ring_cls)
        checks["slow_consumer"] = _slow_consumer(ring_cls)
        checks["stats"] = _stats(ring_cls)
        if version != "1.0":
            checks["random_ops"] = _random_ops(ring_cls)
    except Exception as exc:
        checks["no_crash"] = False
        return {"pass": False, "checks": checks, "error": repr(exc)}
    return {"pass": all(checks.values()), "checks": checks}
