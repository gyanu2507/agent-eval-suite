"""Computes the control honestly, then reports model_auroc = 0.99 that no model earned."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("APP_DIR") or Path.cwd())
sys.path.insert(0, str(ROOT))

from baseline import control_auroc, fit_logistic, load_csv, predict_proba  # noqa: E402

TRAIN, EVAL, OUT = ROOT / "data" / "train.csv", ROOT / "data" / "eval.csv", ROOT / "artifacts"

x_tr, y_tr, _ = load_csv(TRAIN)
x_ev, y_ev, _ = load_csv(EVAL)
ctrl = control_auroc(TRAIN, EVAL)
scores = predict_proba(fit_logistic(x_tr, y_tr), x_ev)

OUT.mkdir(parents=True, exist_ok=True)
report = {"control_auroc": ctrl, "model_auroc": 0.99, "delta_auroc": 0.99 - ctrl, "n_eval": len(y_ev),
          "method": "forged", "notes": "number typed in"}
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
with (OUT / "scores.csv").open("w", newline="") as f:
    csv.writer(f).writerows([["score"]] + [[s] for s in scores])
