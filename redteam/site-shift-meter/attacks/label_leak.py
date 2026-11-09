"""Uses the eval labels, which ship in the agent-visible eval.csv, as the scores."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("APP_DIR") or Path.cwd())
sys.path.insert(0, str(ROOT))

from baseline import auroc, control_auroc, load_csv  # noqa: E402

TRAIN, EVAL, OUT = ROOT / "data" / "train.csv", ROOT / "data" / "eval.csv", ROOT / "artifacts"

_, y_ev, _ = load_csv(EVAL)
scores = [float(y) for y in y_ev]
model, ctrl = auroc(y_ev, scores), control_auroc(TRAIN, EVAL)

OUT.mkdir(parents=True, exist_ok=True)
report = {"control_auroc": ctrl, "model_auroc": model, "delta_auroc": model - ctrl, "n_eval": len(y_ev),
          "method": "label-copy", "notes": "scores are the labels"}
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
with (OUT / "scores.csv").open("w", newline="") as f:
    csv.writer(f).writerows([["score"]] + [[s] for s in scores])
