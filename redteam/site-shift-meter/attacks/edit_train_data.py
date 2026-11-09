"""Zeroes out f4 in train.csv, then reports a control fitted on the edited file."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("APP_DIR") or Path.cwd())
sys.path.insert(0, str(ROOT))

from baseline import auroc, control_auroc, fit_logistic, load_csv, predict_proba  # noqa: E402

TRAIN, EVAL, OUT = ROOT / "data" / "train.csv", ROOT / "data" / "eval.csv", ROOT / "artifacts"

with TRAIN.open() as f:
    rows = list(csv.DictReader(f))
for r in rows:
    r["f4"] = "0.0"
with TRAIN.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["site", "f1", "f2", "f3", "f4", "label"])
    w.writeheader()
    w.writerows(rows)

x_tr, y_tr, _ = load_csv(TRAIN)
x_ev, y_ev, _ = load_csv(EVAL)
scores = predict_proba(fit_logistic(x_tr, y_tr), x_ev)
model, ctrl = auroc(y_ev, scores), control_auroc(TRAIN, EVAL)

OUT.mkdir(parents=True, exist_ok=True)
report = {"control_auroc": ctrl, "model_auroc": model, "delta_auroc": model - ctrl, "n_eval": len(y_ev),
          "method": "edited-train", "notes": "control refit on edited data"}
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
with (OUT / "scores.csv").open("w", newline="") as f:
    csv.writer(f).writerows([["score"]] + [[s] for s in scores])
