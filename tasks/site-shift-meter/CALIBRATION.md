# Calibration: site-shift-meter

Train on two sites, eval on a third. The failure worth catching is a report that looks fine while the method did nothing useful.

## Pass bar

- Starter: submits the control model's own scores, so delta is 0.
- Good run: the verifier scores `scores.csv` against the hidden S2 labels in `tests/eval_labels.csv`. Model AUROC is at least 0.9 and at least 0.5 above the control, and `n_eval` is 180.

## Why these thresholds (v1.1)

v1.0 asked only for delta >= 0.08 and claimed cosmetic per-site z-scoring wouldn't clear it. The red team (`redteam/run.py`) and `scripts/site_shift_ablation.py` measured these on the checked-in CSVs:

| Method | Eval AUROC | Delta |
| --- | --- | --- |
| Frozen control | 0.0108 | 0 |
| Constant score for every row | 0.5 | +0.4892 |
| Random scores, 5,000 seeds | mean 0.500, 99th pct 0.600, max 0.650 | about +0.49 |
| Per-site mean-centering, keeps `f4` | 0.0107 | -0.0001 |
| Per-site z-score, keeps `f4` | 0.1068 | +0.0959 |
| Drop `f4`, logistic (reference, with or without z-scoring) | 0.9766 | +0.9658 |
| Drop `f4`, nearest centroid (`redteam/.../alt`) | 0.9749 | +0.9641 |

Mean-centering is invariant, as the report argues. Z-scoring also rescales, which moves the ranking enough to clear 0.08 without touching the shortcut.

The bigger problem is the control itself. It sits at 0.011, far below chance, so any delta measured against it starts with a free +0.49. Random scores cleared 0.08 in all 5,000 seeds and cleared 0.5 in about 40% of them. A delta gate alone can't tell chance from skill here, so v1.1 adds an absolute floor. Random scores never got above 0.65, and both shortcut-free methods sit near 0.975, so 0.9 separates them with room on both sides. The 0.5 delta is kept because it's the check the construction is about.

## Anti-shortcuts

- `data/` and `baseline.py` are read-only.
- Eval labels live in `tests/eval_labels.csv`, outside `environment/`. v1.0 shipped them in `eval.csv`, so a script could score with the labels and get 1.0.
- The verifier computes both AUROCs itself. Report numbers are ignored. v1.0 trusted `model_auroc` from the report, so a typed-in 0.99 passed.
- Synthetic CSVs only.

## Threat model and what's still open

The agent is assumed to see `instruction.md` and `environment/app` only, the way a harness would run it. One red-team attack still passes, and it needs information outside that view.

This repo is public, and its docs say that `f4` flips its sign at S2. Someone who has read them can submit the control's scores negated and get 0.989 (`redteam/site-shift-meter/attacks/flip_control.py`). Blocking that needs a second hidden eval site where `f4` doesn't flip, so a method that bets on the flip fails there.
