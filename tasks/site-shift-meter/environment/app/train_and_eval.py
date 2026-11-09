#!/usr/bin/env python3
"""Broken starter: reports control as the model (delta 0)."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from baseline import control_auroc, fit_logistic, load_csv, predict_proba

ROOT = Path(os.environ.get("APP_DIR") or Path(__file__).resolve().parent)
TRAIN = ROOT / "data" / "train.csv"
EVAL = ROOT / "data" / "eval.csv"
OUT = ROOT / "artifacts" / "report.json"


def main() -> None:
    ctrl = control_auroc(TRAIN, EVAL)
    x_tr, y_tr, _ = load_csv(TRAIN)
    x_ev, y_ev, _ = load_csv(EVAL)
    scores = predict_proba(fit_logistic(x_tr, y_tr), x_ev)  # BUG: this is just the control model
    report = {
        "control_auroc": ctrl,
        "model_auroc": ctrl,  # BUG: no improvement attempted
        "delta_auroc": 0.0,
        "n_eval": len(y_ev),
        "method": "control-copy",
        "notes": "starter copies the control score",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    with (OUT.parent / "scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["score"])
        w.writerows([[s] for s in scores])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
