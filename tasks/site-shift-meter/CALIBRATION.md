# Calibration: site-shift-meter

Train on two sites, eval on a third. The failure worth catching is a report that looks fine while the method did nothing useful.

## Pass bar

- Starter: delta 0. It writes the control model's scores and copies the control number.
- Good run: model AUROC recomputed from `scores.csv` beats the recomputed control by at least 0.5, the report agrees with the scores within `1e-6`, and `n_eval` is 180.

## Why 0.5 (v1.1)

v1.0 used 0.08 and claimed that cosmetic per-site z-scoring wouldn't clear it. The red team (`redteam/run.py`) measured it on these CSVs:

| Method | Eval AUROC | Delta |
| --- | --- | --- |
| Frozen control | 0.0108 | 0 |
| Per-site mean-centering, keeps `f4` | 0.0107 | -0.0001 |
| Per-site z-score, keeps `f4` | 0.1068 | +0.0959 |
| Drop `f4`, logistic (reference, with or without z-scoring) | 0.9766 | +0.9658 |
| Drop `f4`, nearest centroid (`redteam/.../alt`) | 0.9749 | +0.9641 |

Mean-centering is invariant, as the report argues. Z-scoring also rescales, which moves the ranking enough to clear 0.08 without touching the shortcut. Both shortcut-free methods land near +0.965, so the gap between keeping and dropping `f4` is wide. 0.5 sits in the middle of it.

## Anti-shortcuts

- `data/` and `baseline.py` are read-only.
- Verifier recomputes the control. You cannot invent a weaker baseline in the report.
- Verifier recomputes model AUROC from per-row scores. v1.0 trusted `model_auroc` in the report, so a typed-in 0.99 passed.

## Known open hole

`eval.csv` ships with its `label` column, so a script can score with the labels and get AUROC 1.0. `redteam/site-shift-meter/attacks/label_leak.py` still passes v1.1. Closing it means moving eval labels behind the verifier, which changes the task interface.
- Synthetic CSVs only.
