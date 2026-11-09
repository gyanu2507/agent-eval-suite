#!/usr/bin/env python3
"""Run every attack and every alternate solution against each task's grader.

Each candidate is graded twice: with the v1.0 rules the tasks first shipped
with, and with the current v1.1 rules. The tasks are copied to a temp dir
first, so an attack that edits fixtures can't touch the repo.

Exit code is 1 if any v1.1 result differs from what's expected below.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

TASKS = {
    "site-shift-meter": ("solution/train_and_eval.py", "environment/app/train_and_eval.py"),
    "clearing-hold-ledger": ("solution/replay.py", "environment/app/clearing/replay.py"),
    "slot-ring-buffer": ("solution/ring.py", "environment/app/ring.py"),
    "incident-tool-desk": ("solution/policy.py", "environment/app/policy.py"),
}

# Holes v1.1 does not close yet. They are expected to pass and are listed as open.
KNOWN_OPEN = {
    ("site-shift-meter", "label_leak"): "eval labels ship in the agent-visible eval.csv",
}


def _load_grader(task_dir: Path):
    spec = importlib.util.spec_from_file_location(f"grade_{task_dir.name}", task_dir / "tests" / "grade.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.grade


def _candidates(task: str, task_dir: Path) -> list[tuple[str, str, Path]]:
    reference, starter = TASKS[task]
    out = [("reference", "reference", task_dir / reference), ("starter", "starter", task_dir / starter)]
    out += [("alt", p.stem, p) for p in sorted((HERE / task / "alt").glob("*.py"))]
    out += [("attack", p.stem, p) for p in sorted((HERE / task / "attacks").glob("*.py"))]
    return out


def _expected(task: str, kind: str, name: str) -> bool:
    if (task, name) in KNOWN_OPEN:
        return True
    return kind in ("reference", "alt")


def _doc(path: Path) -> str:
    text = path.read_text()
    if text.startswith('"""'):
        return text[3:].split('"""', 1)[0].strip().replace("\n", " ")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="also write results here")
    args = parser.parse_args()

    rows, unexpected = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for task in TASKS:
            task_dir = Path(tmp) / task
            shutil.copytree(REPO / "tasks" / task, task_dir, ignore=shutil.ignore_patterns("artifacts", "__pycache__"))
            grade = _load_grader(task_dir)
            for kind, name, path in _candidates(task, task_dir):
                v10 = grade(path.resolve(), version="1.0")
                v11 = grade(path.resolve(), version="1.1")
                want = _expected(task, kind, name)
                ok = v11["pass"] == want
                unexpected += not ok
                rows.append({
                    "task": task,
                    "kind": kind,
                    "name": name,
                    "what": "" if kind in ("reference", "starter") else _doc(path),
                    "v1.0": v10["pass"],
                    "v1.1": v11["pass"],
                    "failed_checks_v1.1": [k for k, v in v11["checks"].items() if not v],
                    "as_expected": ok,
                    "known_open": KNOWN_OPEN.get((task, name), ""),
                })

    def mark(passed: bool) -> str:
        return "pass" if passed else "fail"

    print(f"{'task':<22} {'kind':<9} {'candidate':<24} {'v1.0':<5} {'v1.1':<5} note")
    for r in rows:
        if r["known_open"]:
            note = f"OPEN: {r['known_open']}"
        elif not r["as_expected"]:
            note = "UNEXPECTED"
        elif r["kind"] == "attack" and r["v1.0"]:
            note = "hole in v1.0, closed in v1.1"
        else:
            note = ""
        print(f"{r['task']:<22} {r['kind']:<9} {r['name']:<24} {mark(r['v1.0']):<5} {mark(r['v1.1']):<5} {note}")

    attacks = [r for r in rows if r["kind"] == "attack"]
    print(
        f"\n{len(attacks)} attacks: {sum(r['v1.0'] for r in attacks)} pass v1.0, "
        f"{sum(r['v1.1'] for r in attacks)} pass v1.1 ({len(KNOWN_OPEN)} known open). "
        f"{unexpected} unexpected result(s)."
    )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
