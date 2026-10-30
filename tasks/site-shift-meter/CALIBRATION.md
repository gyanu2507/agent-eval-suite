# Calibration: site-shift-meter

Train on two sites, eval on a third. The interesting failure is not “accuracy is low.” It is “the report looks fine while the method did nothing useful.”

## Pass bar

- Starter: `delta_auroc == 0` (copies the control into the model field).
- Good run: `delta_auroc >= 0.08`, and `control_auroc` matches `baseline.control_auroc` within `1e-6`.
- `n_eval` equals the eval CSV length (180).

## Why 0.08

Big enough that copying the control or cosmetic z-scoring on mean-only shift will not clear it once the spurious train feature is in play. Small enough that a clean drop-`f4` + per-site z-score method passes without needing a perfect model.

## Anti-shortcuts

- `data/` and `baseline.py` are read-only.
- Verifier recomputes the control. You cannot invent a weaker baseline in the report.
- Synthetic CSVs only.
