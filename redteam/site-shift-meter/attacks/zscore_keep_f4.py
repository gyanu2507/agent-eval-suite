"""Z-scores every feature per site and keeps the shortcut feature f4."""

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


def per_site(xs, sites, scale):
    groups: dict[str, list] = {}
    for x, s in zip(xs, sites):
        groups.setdefault(s, []).append(x)
    stats = {}
    for s, rows in groups.items():
        n, d = len(rows), len(rows[0])
        mu = [sum(r[j] for r in rows) / n for j in range(d)]
        sd = [max((sum((r[j] - mu[j]) ** 2 for r in rows) / n) ** 0.5, 1e-12) if scale else 1.0 for j in range(d)]
        stats[s] = (mu, sd)
    return [[(xi - m) / v for xi, m, v in zip(x, *stats[s])] for x, s in zip(xs, sites)]


x_tr, y_tr, s_tr = load_csv(TRAIN)
x_ev, _, s_ev = load_csv(EVAL)
w = fit_logistic(per_site(x_tr, s_tr, True), y_tr, lr=0.08, epochs=700)
write(predict_proba(w, per_site(x_ev, s_ev, True)), "per-site-standardize-all")
