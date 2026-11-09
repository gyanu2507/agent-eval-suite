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
ARTIFACTS = APP / "artifacts"
LABELS = TASK / "tests" / "eval_labels.csv"

sys.path.insert(0, str(APP))
from baseline import auroc, control_auroc, load_labels  # noqa: E402


def _run(script: Path) -> tuple[dict, list[float]]:
    for name in ("report.json", "scores.csv"):
        (ARTIFACTS / name).unlink(missing_ok=True)
    subprocess.check_call([sys.executable, str(script)], cwd=str(APP))
    report = json.loads((ARTIFACTS / "report.json").read_text())
    with (ARTIFACTS / "scores.csv").open() as f:
        scores = [float(row["score"]) for row in csv.DictReader(f)]
    return report, scores


def _control() -> float:
    return control_auroc(APP / "data" / "train.csv", APP / "data" / "eval.csv", load_labels(LABELS))


def test_eval_csv_has_no_labels():
    header = (APP / "data" / "eval.csv").read_text().splitlines()[0]
    assert header == "site,f1,f2,f3,f4"


def test_starter_delta_is_zero():
    report, scores = _run(STARTER)
    assert report["method"] == "control-copy"
    assert abs(auroc(load_labels(LABELS), scores) - _control()) < 1e-9


def test_oracle_clears_gate():
    report, scores = _run(SOLUTION)
    model = auroc(load_labels(LABELS), scores)
    assert report["n_eval"] == len(scores) == 180
    assert model >= 0.9
    assert model - _control() >= 0.5


def test_control_is_recomputable():
    a, b = _control(), _control()
    assert a == b
    assert 0.0 <= a <= 1.0
    # Control latches onto a train-only spurious feature and collapses under shift.
    assert a < 0.55
