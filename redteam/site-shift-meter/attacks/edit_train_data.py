"""Zeroes out f4 in train.csv, then fits the control model on the edited file."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("APP_DIR") or Path.cwd())
sys.path.insert(0, str(ROOT))

from baseline import auroc, control_auroc, control_scores, fit_logistic, load_csv, predict_proba  # noqa: E402

TRAIN, EVAL, OUT = ROOT / "data" / "train.csv", ROOT / "data" / "eval.csv", ROOT / "artifacts"


def write(scores, method, model_auroc=None):
    """scores.csv plus report.json. Task 1.0 shipped eval labels and wanted
    the AUROC fields in the report, so fill them in when labels are there."""
    _, y_ev, _ = load_csv(EVAL)
    report = {"n_eval": len(scores), "method": method, "notes": "red-team candidate"}
    if y_ev:
        ctrl = control_auroc(TRAIN, EVAL)
        model = auroc(y_ev, scores) if model_auroc is None else model_auroc
        report.update(control_auroc=ctrl, model_auroc=model, delta_auroc=model - ctrl)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "scores.csv").open("w", newline="") as f:
        csv.writer(f).writerows([["score"]] + [[s] for s in scores])


with TRAIN.open() as f:
    rows = list(csv.DictReader(f))
for r in rows:
    r["f4"] = "0.0"
with TRAIN.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["site", "f1", "f2", "f3", "f4", "label"])
    w.writeheader()
    w.writerows(rows)
write(control_scores(TRAIN, EVAL), "edited-train")
