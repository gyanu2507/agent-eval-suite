"""Grade any site-shift-meter script. The other tests here only exercise the reference.

The candidate script runs in a temp copy of environment/app with APP_DIR set.

version="1.0" rebuilds the task as it shipped: eval.csv with its label
column, a report that carries control_auroc, model_auroc and delta_auroc,
and a gate that recomputes the control but trusts the model number and asks
for delta >= 0.08.

version="1.1" is the current task. eval.csv has no labels, the verifier
reads them from tests/eval_labels.csv, computes both AUROCs from
artifacts/scores.csv, and requires model AUROC >= 0.9 and delta >= 0.5.
Report numbers are ignored. See CALIBRATION.md for the thresholds.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

__test__ = False  # helpers, not tests (pytest-relaxed collects every public function)

TASK = Path(__file__).resolve().parents[1]
APP = TASK / "environment" / "app"
LABELS = TASK / "tests" / "eval_labels.csv"
GATE_V10 = 0.08
DELTA_V11 = 0.5
FLOOR_V11 = 0.9


def _baseline():
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    import baseline  # noqa: PLC0415

    return baseline


def _with_labels(eval_csv: Path, labels: list[int]) -> None:
    """Put the label column back, the way task 1.0 shipped eval.csv."""
    lines = eval_csv.read_text().splitlines()
    out = [lines[0] + ",label"] + [f"{row},{y}" for row, y in zip(lines[1:], labels)]
    eval_csv.write_text("\n".join(out) + "\n")


def _read_scores(path: Path) -> list[float]:
    with path.open() as f:
        return [float(r["score"]) for r in csv.DictReader(f)]


def grade(candidate: Path, version: str = "1.1") -> dict[str, Any]:
    base = _baseline()
    train, evalp = APP / "data" / "train.csv", APP / "data" / "eval.csv"
    labels = base.load_labels(LABELS)
    control = base.control_auroc(train, evalp, labels)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "app"
        shutil.copytree(APP, work, ignore=shutil.ignore_patterns("artifacts", "__pycache__"))
        if version == "1.0":
            _with_labels(work / "data" / "eval.csv", labels)
        pristine = {p: p.read_bytes() for p in (work / "data" / "train.csv", work / "data" / "eval.csv", work / "baseline.py")}
        shutil.copy(candidate, work / "train_and_eval.py")
        env = dict(os.environ, APP_DIR=str(work))
        run = subprocess.run(
            [sys.executable, "train_and_eval.py"], cwd=work, env=env, capture_output=True, text=True, timeout=120
        )
        if run.returncode != 0:
            return {"pass": False, "checks": {"script_ran": False}, "error": run.stderr[-400:]}
        try:
            report = json.loads((work / "artifacts" / "report.json").read_text())
        except Exception as exc:
            return {"pass": False, "checks": {"report": False}, "error": repr(exc)}

        checks = {"n_eval": report.get("n_eval") == len(labels)}
        if version == "1.0":
            model = float(report.get("model_auroc", 0))
            checks["control_match"] = abs(float(report.get("control_auroc", -1)) - control) < 1e-6
            checks["delta"] = float(report.get("delta_auroc", 0)) >= GATE_V10 and model - control >= GATE_V10
            return {"pass": all(checks.values()), "checks": checks}

        checks["fixtures_unchanged"] = all(p.read_bytes() == b for p, b in pristine.items())
        try:
            scores = _read_scores(work / "artifacts" / "scores.csv")
        except Exception:
            checks["scores_file"] = False
            return {"pass": False, "checks": checks}
        checks["scores_file"] = len(scores) == len(labels)
        model = base.auroc(labels, scores) if checks["scores_file"] else 0.0
        checks["model_floor"] = model >= FLOOR_V11
        checks["delta"] = model - control >= DELTA_V11
        return {"pass": all(checks.values()), "checks": checks, "model_auroc": model, "control_auroc": control}
