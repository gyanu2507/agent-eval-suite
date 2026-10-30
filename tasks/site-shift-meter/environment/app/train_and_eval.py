#!/usr/bin/env python3
"""Broken starter: reports control as the model (delta 0)."""

from __future__ import annotations

import json
from pathlib import Path

from baseline import control_auroc, load_csv

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "data" / "train.csv"
EVAL = ROOT / "data" / "eval.csv"
OUT = ROOT / "artifacts" / "report.json"


def main() -> None:
    ctrl = control_auroc(TRAIN, EVAL)
    _, y_ev, _ = load_csv(EVAL)
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
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
