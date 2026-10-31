# Case study: when a “shift fix” does nothing

Full writeup: [tech report](tech-report.md) (PDF in [v0.1.0](https://github.com/gyanu2507/agent-eval-suite/releases/tag/v0.1.0)).

Task: `site-shift-meter`. Train on sites S0/S1, score on S2.

## What the starter does

The starter trains a plain logistic on raw features, then copies that AUROC into both `control_auroc` and `model_auroc`. Delta is zero. Looks finished if you only glance at the JSON shape.

## What looked like a real fix

Per-site z-scoring, then the same logistic. Textbook adaptation. On the first synthetic draw I tried, mean shift alone barely moved ranking AUROC for a linear scorer. Add a constant to every score and the order stays the same. So the “adapted” model cleared almost nothing against the control.

## What actually broke the control

I put a strong label shortcut in `f4` on the train sites and flipped it on eval (anti-correlated noise). The frozen control in `baseline.py` latches onto `f4` and collapses on S2. A method that drops `f4` and z-scores `f1`–`f3` per site (eval moments from unlabeled features) clears a large delta.

On the checked-in CSVs the control lands near 0.01 AUROC and the reference method near 0.98. The gate is milder on purpose: `delta_auroc >= 0.08`, and `control_auroc` must match a verifier recompute within `1e-6`.

## Pass bar

1. Starter fails (delta 0).
2. Verifier recomputes the control from `baseline.py` + the CSVs. You cannot lower the control by editing a report field.
3. `data/` and `baseline.py` are read-only.
4. `n_eval` must equal the eval row count (180).

## Shortcuts this blocks

- Printing a nicer delta without beating the frozen control
- Rewriting the baseline to make the gap look good
- Mean-only “normalization” that never changes ranking metrics
- Editing goldens instead of fixing the method

## Why this task exists

Across the private work the pattern that keeps showing up is the same: nop fails, oracle passes, score comes back from artifacts, and the control is honest even when it hurts. This public task is the small version of that, with synthetic data only.
