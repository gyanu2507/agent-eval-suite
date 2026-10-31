# Case study: mean shift without ranking change

Full writeup: [tech report](tech-report.md) ([DOI](https://doi.org/10.5281/zenodo.23829461)).

Task: `site-shift-meter`. Train on sites S0/S1, evaluate on S2.

## Starter

Trains logistic regression on raw features, then copies that AUROC into both `control_auroc` and `model_auroc`. Reported delta is zero. The JSON schema appears complete.

## Why mean-only adaptation is insufficient

Per-site z-scoring followed by the same logistic is a common first intervention. Under mean-only synthetic shift, ranking AUROC for a linear scorer is largely invariant to a constant score offset. Order is preserved, so the adapted model clears almost nothing against the control.

## What breaks the control

A strong label shortcut in `f4` on training sites is flipped on evaluation (anti-correlated noise). The frozen control in `baseline.py` relies on `f4` and collapses on S2. Dropping `f4` and z-scoring `f1`-`f3` per site (evaluation-site moments from unlabeled evaluation features) recovers a large delta.

On the checked-in CSVs: control AUROC ≈ 0.01, reference method ≈ 0.98. Gate: `delta_auroc >= 0.08`, with control recomputed within `1e-6`.

## Pass conditions

1. Starter fails (delta 0).
2. Verifier recomputes the control from `baseline.py` and the CSVs.
3. `data/` and `baseline.py` are read-only.
4. `n_eval` equals the evaluation row count (180).

## Blocked shortcuts

Inflating delta without beating the frozen control. Weakening the baseline in-place. Mean-only normalization that does not change ranking metrics. Editing goldens instead of changing the method.
