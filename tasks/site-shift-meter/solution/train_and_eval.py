#!/usr/bin/env python3
"""Reference: drop spurious f4, per-site z-score f1-f3, logistic."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "environment" / "app"
sys.path.insert(0, str(ROOT))

from baseline import (  # noqa: E402
    auroc,
    control_auroc,
    fit_logistic,
    load_csv,
    predict_proba,
)

TRAIN = ROOT / "data" / "train.csv"
EVAL = ROOT / "data" / "eval.csv"
OUT = ROOT / "artifacts" / "report.json"


def _take(xs: list[list[float]], cols: list[int]) -> list[list[float]]:
    return [[row[j] for j in cols] for row in xs]


def _site_stats(
    xs: list[list[float]], sites: list[str]
) -> dict[str, tuple[list[float], list[float]]]:
    buckets: dict[str, list[list[float]]] = defaultdict(list)
    for x, s in zip(xs, sites):
        buckets[s].append(x)
    out: dict[str, tuple[list[float], list[float]]] = {}
    for site, rows in buckets.items():
        dim = len(rows[0])
        means = [sum(r[j] for r in rows) / len(rows) for j in range(dim)]
        stds = []
        for j in range(dim):
            var = sum((r[j] - means[j]) ** 2 for r in rows) / max(len(rows), 1)
            stds.append(var**0.5 if var > 1e-12 else 1.0)
        out[site] = (means, stds)
    return out


def _transform(
    xs: list[list[float]],
    sites: list[str],
    stats: dict[str, tuple[list[float], list[float]]],
    fallback: tuple[list[float], list[float]],
) -> list[list[float]]:
    out: list[list[float]] = []
    for x, s in zip(xs, sites):
        means, stds = stats.get(s, fallback)
        out.append([(xi - m) / sd for xi, m, sd in zip(x, means, stds)])
    return out


def main() -> None:
    x_tr, y_tr, s_tr = load_csv(TRAIN)
    x_ev, y_ev, s_ev = load_csv(EVAL)

    # f4 is a train-only shortcut / eval trap; use f1..f3 only.
    cols = [0, 1, 2]
    x_tr = _take(x_tr, cols)
    x_ev = _take(x_ev, cols)

    stats = _site_stats(x_tr, s_tr)
    for site, ms in _site_stats(x_ev, s_ev).items():
        stats[site] = ms

    dim = len(x_tr[0])
    g_mean = [sum(r[j] for r in x_tr) / len(x_tr) for j in range(dim)]
    g_std = []
    for j in range(dim):
        var = sum((r[j] - g_mean[j]) ** 2 for r in x_tr) / len(x_tr)
        g_std.append(var**0.5 if var > 1e-12 else 1.0)
    fallback = (g_mean, g_std)

    x_tr_z = _transform(x_tr, s_tr, stats, fallback)
    x_ev_z = _transform(x_ev, s_ev, stats, fallback)

    weights = fit_logistic(x_tr_z, y_tr, lr=0.08, epochs=700)
    scores = predict_proba(weights, x_ev_z)
    model = auroc(y_ev, scores)
    ctrl = control_auroc(TRAIN, EVAL)

    report = {
        "control_auroc": ctrl,
        "model_auroc": model,
        "delta_auroc": model - ctrl,
        "n_eval": len(y_ev),
        "method": "drop-f4-site-zscore-logistic",
        "notes": "ignore spurious f4; z-score f1-f3 per site including unlabeled eval",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
