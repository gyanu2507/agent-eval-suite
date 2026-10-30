# Calibration: site-shift-meter

## Why this task

Public distribution-shift sample: train on two sites, eval on a third with shifted feature means. The agent must keep the frozen control honest and clear a delta gate with a shift-aware method.

## Pass bar

- Starter reports `delta_auroc == 0` (control copy).
- Oracle / good agent: `delta_auroc >= 0.08` and `control_auroc` matches `baseline.control_auroc` within `1e-6`.
- `n_eval` must match the eval CSV row count (180).

## Anti-shortcuts

- `data/` and `baseline.py` are read-only.
- Verifier recomputes the control; agents cannot invent a lower control to inflate delta.
- Synthetic CSVs only. No private research packages.
