#!/usr/bin/env python3
"""Broken starter: submits the control model's own scores (delta 0)."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from baseline import control_auroc, control_scores, load_csv

ROOT = Path(os.environ.get("APP_DIR") or Path(__file__).resolve().parent)
TRAIN = ROOT / "data" / "train.csv"
EVAL = ROOT / "data" / "eval.csv"
OUT = ROOT / "artifacts" / "report.json"


def main() -> None:
    x_ev, y_ev, _ = load_csv(EVAL)
    scores = control_scores(TRAIN, EVAL)  # BUG: this is the control model, unchanged
    report = {
        "n_eval": len(x_ev),
        "method": "control-copy",
        "notes": "starter submits the control model's scores",
    }
    if y_ev:  # task 1.0 shipped eval labels and asked for these three fields
        ctrl = control_auroc(TRAIN, EVAL)
        report.update(control_auroc=ctrl, model_auroc=ctrl, delta_auroc=0.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    with (OUT.parent / "scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score"])
        w.writerows([[s] for s in scores])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
