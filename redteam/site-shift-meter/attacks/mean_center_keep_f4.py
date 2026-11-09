"""Subtracts each site's feature means and keeps the shortcut feature f4."""

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
CENTER_ONLY = True


def per_site(xs, sites):
    groups: dict[str, list] = {}
    for x, s in zip(xs, sites):
        groups.setdefault(s, []).append(x)
    stats = {}
    for s, rows in groups.items():
        n, d = len(rows), len(rows[0])
        mu = [sum(r[j] for r in rows) / n for j in range(d)]
        sd = [1.0 if CENTER_ONLY else max((sum((r[j] - mu[j]) ** 2 for r in rows) / n) ** 0.5, 1e-12) for j in range(d)]
        stats[s] = (mu, sd)
    return [[(xi - m) / v for xi, m, v in zip(x, *stats[s])] for x, s in zip(xs, sites)]


x_tr, y_tr, s_tr = load_csv(TRAIN)
x_ev, y_ev, s_ev = load_csv(EVAL)
w = fit_logistic(per_site(x_tr, s_tr), y_tr, lr=0.08, epochs=700)
scores = predict_proba(w, per_site(x_ev, s_ev))
model, ctrl = auroc(y_ev, scores), control_auroc(TRAIN, EVAL)

OUT.mkdir(parents=True, exist_ok=True)
report = {"control_auroc": ctrl, "model_auroc": model, "delta_auroc": model - ctrl, "n_eval": len(y_ev),
          "method": "per-site-center-all", "notes": "center per site, keep every feature"}
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
with (OUT / "scores.csv").open("w", newline="") as f:
    csv.writer(f).writerows([["score"]] + [[s] for s in scores])
