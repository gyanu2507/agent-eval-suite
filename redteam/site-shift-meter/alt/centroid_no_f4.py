"""A correct method that isn't the reference: drop f4, center f1 to f3 per site,
then score by distance to the class centroids instead of logistic regression."""

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


def centered(xs, sites):
    xs = [x[:3] for x in xs]
    by_site: dict[str, list] = {}
    for x, s in zip(xs, sites):
        by_site.setdefault(s, []).append(x)
    mu = {s: [sum(c) / len(rows) for c in zip(*rows)] for s, rows in by_site.items()}
    return [[a - m for a, m in zip(x, mu[s])] for x, s in zip(xs, sites)]


x_tr, y_tr, s_tr = load_csv(TRAIN)
x_ev, y_ev, s_ev = load_csv(EVAL)
tr, ev = centered(x_tr, s_tr), centered(x_ev, s_ev)

pos = [x for x, y in zip(tr, y_tr) if y == 1]
neg = [x for x, y in zip(tr, y_tr) if y == 0]
c1 = [sum(c) / len(pos) for c in zip(*pos)]
c0 = [sum(c) / len(neg) for c in zip(*neg)]
scores = [sum((a - b) ** 2 for a, b in zip(x, c0)) - sum((a - b) ** 2 for a, b in zip(x, c1)) for x in ev]
model, ctrl = auroc(y_ev, scores), control_auroc(TRAIN, EVAL)

OUT.mkdir(parents=True, exist_ok=True)
report = {"control_auroc": ctrl, "model_auroc": model, "delta_auroc": model - ctrl, "n_eval": len(y_ev),
          "method": "drop-f4-site-center-centroid", "notes": "drop f4, center per site, nearest centroid"}
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
with (OUT / "scores.csv").open("w", newline="") as f:
    csv.writer(f).writerows([["score"]] + [[s] for s in scores])
