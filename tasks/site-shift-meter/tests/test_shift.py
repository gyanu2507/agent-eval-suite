from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
APP = TASK / "environment" / "app"
SOLUTION = TASK / "solution" / "train_and_eval.py"
STARTER = APP / "train_and_eval.py"
REPORT = APP / "artifacts" / "report.json"


def _run(script: Path) -> dict:
    if REPORT.exists():
        REPORT.unlink()
    subprocess.check_call([sys.executable, str(script)], cwd=str(APP))
    return json.loads(REPORT.read_text())


def test_starter_delta_is_zero():
    report = _run(STARTER)
    assert abs(report["delta_auroc"]) < 1e-9
    assert report["method"] == "control-copy"


def test_oracle_clears_gate():
    sys.path.insert(0, str(APP))
    from baseline import auroc, control_auroc, load_csv  # noqa: WPS433

    report = _run(SOLUTION)
    ctrl = control_auroc(APP / "data" / "train.csv", APP / "data" / "eval.csv")
    assert abs(report["control_auroc"] - ctrl) < 1e-6
    assert report["n_eval"] == 180
    assert report["delta_auroc"] >= 0.5
    assert report["model_auroc"] - report["control_auroc"] >= 0.5
    _, labels, _ = load_csv(APP / "data" / "eval.csv")
    with (APP / "artifacts" / "scores.csv").open() as f:
        scores = [float(row["score"]) for row in csv.DictReader(f)]
    assert abs(auroc(labels, scores) - report["model_auroc"]) < 1e-6


def test_control_is_recomputable():
    sys.path.insert(0, str(APP))
    from baseline import control_auroc

    a = control_auroc(APP / "data" / "train.csv", APP / "data" / "eval.csv")
    b = control_auroc(APP / "data" / "train.csv", APP / "data" / "eval.csv")
    assert a == b
    assert 0.0 <= a <= 1.0
    # Control latches onto a train-only spurious feature and collapses under shift.
    assert a < 0.55
