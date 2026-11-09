"""Grade any site-shift-meter script. The other tests here only exercise the reference.

The candidate script runs in a temp copy of environment/app with APP_DIR set.

version="1.0" is the gate as it shipped: recompute the control, then trust
model_auroc and delta_auroc from report.json and require delta >= 0.08.

version="1.1" also requires artifacts/scores.csv (one score per eval row),
recomputes model AUROC from it against pristine labels, rejects a report
that disagrees, and gates on the recomputed delta >= 0.5. See CALIBRATION.md
for where 0.5 comes from.
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
GATE_V10 = 0.08
GATE_V11 = 0.5


def _baseline():
    sys.path.insert(0, str(APP))
    import baseline  # noqa: PLC0415

    return baseline


def _read_scores(path: Path) -> list[float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return [float(r["score"]) for r in rows]


def grade(candidate: Path, version: str = "1.1") -> dict[str, Any]:
    base = _baseline()
    train, evalp = APP / "data" / "train.csv", APP / "data" / "eval.csv"
    pristine = {p.name: p.read_bytes() for p in (train, evalp, APP / "baseline.py")}
    control = base.control_auroc(train, evalp)
    _, labels, _ = base.load_csv(evalp)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "app"
        shutil.copytree(APP, work, ignore=shutil.ignore_patterns("artifacts", "__pycache__"))
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

        checks = {
            "control_match": abs(float(report.get("control_auroc", -1)) - control) < 1e-6,
            "n_eval": report.get("n_eval") == len(labels),
        }
        if version == "1.0":
            model = float(report.get("model_auroc", 0))
            checks["delta"] = float(report.get("delta_auroc", 0)) >= GATE_V10 and model - control >= GATE_V10
            return {"pass": all(checks.values()), "checks": checks}

        checks["data_unchanged"] = all((work / "data" / n).read_bytes() == b for n, b in pristine.items() if n != "baseline.py")
        checks["baseline_unchanged"] = (work / "baseline.py").read_bytes() == pristine["baseline.py"]
        scores_path = work / "artifacts" / "scores.csv"
        try:
            scores = _read_scores(scores_path)
        except Exception:
            checks["scores_file"] = False
            return {"pass": False, "checks": checks}
        checks["scores_file"] = len(scores) == len(labels)
        recomputed = base.auroc(labels, scores) if checks["scores_file"] else 0.0
        checks["report_matches_scores"] = abs(float(report.get("model_auroc", -1)) - recomputed) < 1e-6
        checks["delta"] = recomputed - control >= GATE_V11
        return {"pass": all(checks.values()), "checks": checks, "model_auroc": recomputed, "control_auroc": control}
