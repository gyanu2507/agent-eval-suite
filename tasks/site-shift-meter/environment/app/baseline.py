"""Frozen control model. Agents must not edit this file."""

from __future__ import annotations

import csv
import math
from pathlib import Path

FEATURES = ("f1", "f2", "f3", "f4")


def load_csv(path: Path) -> tuple[list[list[float]], list[int], list[str]]:
    xs: list[list[float]] = []
    ys: list[int] = []
    sites: list[str] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append([float(row[c]) for c in FEATURES])
            ys.append(int(row["label"]))
            sites.append(row["site"])
    return xs, ys, sites


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fit_logistic(
    xs: list[list[float]], ys: list[int], lr: float = 0.05, epochs: int = 400
) -> list[float]:
    """Unregularized logistic on raw features + bias. Deterministic."""
    dim = len(xs[0])
    w = [0.0] * dim
    b = 0.0
    n = len(xs)
    for _ in range(epochs):
        gw = [0.0] * dim
        gb = 0.0
        for x, y in zip(xs, ys):
            z = b + sum(wi * xi for wi, xi in zip(w, x))
            p = _sigmoid(z)
            err = p - y
            for j in range(dim):
                gw[j] += err * x[j]
            gb += err
        for j in range(dim):
            w[j] -= lr * gw[j] / n
        b -= lr * gb / n
    return w + [b]


def predict_proba(weights: list[float], xs: list[list[float]]) -> list[float]:
    w, b = weights[:-1], weights[-1]
    out: list[float] = []
    for x in xs:
        z = b + sum(wi * xi for wi, xi in zip(w, x))
        out.append(_sigmoid(z))
    return out


def auroc(ys: list[int], scores: list[float]) -> float:
    pairs = sorted(zip(scores, ys), key=lambda t: t[0])
    # Mann-Whitney / Wilcoxon form
    n_pos = sum(ys)
    n_neg = len(ys) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0
    i = 0
    n = len(pairs)
    rank = 1
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        rank += j - i + 1
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def control_auroc(train_path: Path, eval_path: Path) -> float:
    x_tr, y_tr, _ = load_csv(train_path)
    x_ev, y_ev, _ = load_csv(eval_path)
    weights = fit_logistic(x_tr, y_tr)
    scores = predict_proba(weights, x_ev)
    return auroc(y_ev, scores)
