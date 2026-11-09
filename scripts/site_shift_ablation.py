#!/usr/bin/env python3
"""Ablation for site-shift-meter: which fixes actually move eval AUROC.

Runs the frozen control plus three candidate methods on the checked-in data,
writes docs/figures/site-shift-ablation.json and an SVG bar chart next to it.
Stdlib only. Deterministic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASK = REPO / "tasks" / "site-shift-meter"
APP = TASK / "environment" / "app"
sys.path.insert(0, str(APP))

from baseline import auroc, control_auroc, fit_logistic, load_csv, predict_proba  # noqa: E402

_spec = importlib.util.spec_from_file_location("oracle", TASK / "solution" / "train_and_eval.py")
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)

TRAIN = APP / "data" / "train.csv"
EVAL = APP / "data" / "eval.csv"
OUT_DIR = REPO / "docs" / "figures"
GATE = 0.5  # v1.1 gate, see tasks/site-shift-meter/CALIBRATION.md


def _zscore(x_tr, s_tr, x_ev, s_ev):
    stats = oracle._site_stats(x_tr, s_tr)
    stats.update(oracle._site_stats(x_ev, s_ev))
    dim = len(x_tr[0])
    g_mean = [sum(r[j] for r in x_tr) / len(x_tr) for j in range(dim)]
    g_std = []
    for j in range(dim):
        var = sum((r[j] - g_mean[j]) ** 2 for r in x_tr) / len(x_tr)
        g_std.append(var**0.5 if var > 1e-12 else 1.0)
    fb = (g_mean, g_std)
    return oracle._transform(x_tr, s_tr, stats, fb), oracle._transform(x_ev, s_ev, stats, fb)


def _center(xs, sites):
    stats = oracle._site_stats(xs, sites)
    return [[xi - m for xi, m in zip(x, stats[s][0])] for x, s in zip(xs, sites)]


def _fit_eval(x_tr, y_tr, x_ev, y_ev) -> float:
    w = fit_logistic(x_tr, y_tr, lr=0.08, epochs=700)
    return auroc(y_ev, predict_proba(w, x_ev))


def main() -> None:
    x_tr, y_tr, s_tr = load_csv(TRAIN)
    x_ev, y_ev, s_ev = load_csv(EVAL)
    keep = [0, 1, 2]
    ctrl = control_auroc(TRAIN, EVAL)

    rows = [("Frozen control (all features, raw)", ctrl)]
    rows.append(("Starter: copy control into model field", ctrl))
    ct, ce = _center(x_tr, s_tr), _center(x_ev, s_ev)
    rows.append(("Mean-only: per-site centering, keeps f4", _fit_eval(ct, y_tr, ce, y_ev)))
    zt, ze = _zscore(x_tr, s_tr, x_ev, s_ev)
    rows.append(("Per-site z-score, keeps f4", _fit_eval(zt, y_tr, ze, y_ev)))
    rows.append(("Drop shortcut f4, no adaptation", _fit_eval(oracle._take(x_tr, keep), y_tr, oracle._take(x_ev, keep), y_ev)))
    zt, ze = _zscore(oracle._take(x_tr, keep), s_tr, oracle._take(x_ev, keep), s_ev)
    rows.append(("Oracle: drop f4 + per-site z-score", _fit_eval(zt, y_tr, ze, y_ev)))

    result = [
        {"method": m, "eval_auroc": round(a, 4), "delta_vs_control": round(a - ctrl, 4), "passes_gate": a - ctrl >= GATE}
        for m, a in rows
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "site-shift-ablation.json").write_text(json.dumps({"gate": GATE, "n_eval": len(y_ev), "rows": result}, indent=2) + "\n")
    (OUT_DIR / "site-shift-ablation.svg").write_text(_svg(result))
    for r in result:
        print(f"{r['eval_auroc']:.4f}  delta {r['delta_vs_control']:+.4f}  {'PASS' if r['passes_gate'] else 'fail'}  {r['method']}")


def _svg(rows) -> str:
    w, left, bar_h, gap, top = 760, 360, 26, 14, 48
    plot_w = w - left - 70
    h = top + len(rows) * (bar_h + gap) + 40
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="ui-sans-serif,system-ui,sans-serif" font-size="13">',
        '<style>.t{fill:#24292f}.m{fill:#57606a}.pass{fill:#1a7f37}.fail{fill:#afb8c1}.ax{stroke:#d0d7de}'
        '@media (prefers-color-scheme:dark){.t{fill:#e6edf3}.m{fill:#9198a1}.pass{fill:#3fb950}.fail{fill:#484f58}.ax{stroke:#30363d}}</style>',
        f'<text class="t" x="0" y="20" font-weight="600" font-size="15">site-shift-meter: eval AUROC by method (n=180, gate = control + {GATE})</text>',
    ]
    ctrl = rows[0]["eval_auroc"]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + frac * plot_w
        out.append(f'<line class="ax" x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{h - 30}"/>')
        out.append(f'<text class="m" x="{x:.1f}" y="{h - 12}" text-anchor="middle">{frac:.2f}</text>')
    gx = left + (ctrl + GATE) * plot_w
    out.append(f'<line x1="{gx:.1f}" y1="{top - 8}" x2="{gx:.1f}" y2="{h - 30}" stroke="#cf222e" stroke-dasharray="4 3"/>')
    out.append(f'<text x="{gx + 4:.1f}" y="{top - 12}" fill="#cf222e" font-size="12">gate</text>')
    for i, r in enumerate(rows):
        y = top + i * (bar_h + gap)
        bw = max(r["eval_auroc"] * plot_w, 2)
        cls = "pass" if r["passes_gate"] else "fail"
        out.append(f'<text class="t" x="{left - 10}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end">{r["method"]}</text>')
        out.append(f'<rect class="{cls}" x="{left}" y="{y}" width="{bw:.1f}" height="{bar_h}" rx="3"/>')
        out.append(f'<text class="t" x="{left + bw + 6:.1f}" y="{y + bar_h / 2 + 4:.1f}">{r["eval_auroc"]:.3f}</text>')
    out.append("</svg>\n")
    return "\n".join(out)


if __name__ == "__main__":
    main()
