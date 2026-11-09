# Site Shift Meter

Synthetic energy-meter fault detection under site shift.

Train on sites `S0` and `S1`. Score held-out site `S2`, whose feature means are shifted. The verifier holds the S2 labels and compares your scores with a frozen control.

## Data

- `environment/app/data/train.csv`: rows from sites S0 and S1
- `environment/app/data/eval.csv`: rows from site S2, without labels
- Train columns: `site,f1,f2,f3,f4,label` with `label` in `{0,1}`. Eval has the same columns minus `label`.

## Required outputs

Write two files under `environment/app/artifacts/`.

`scores.csv` has a `score` header and one row per eval row, in the same order as `eval.csv`. Higher means more likely to be a fault.

`report.json` has:

```json
{
  "n_eval": <int>,
  "method": "<short name>",
  "notes": "<one sentence on shift handling>"
}
```

## Gate

The verifier computes both AUROCs on the hidden S2 labels: the frozen logistic control from `baseline.py`, and your model from `scores.csv`.

1. Model AUROC must be at least 0.9.
2. Model AUROC must be at least 0.5 above the control.
3. `n_eval` must equal the eval row count.
4. Do not edit `data/` or `baseline.py`.

## Starter

`train_and_eval.py` currently submits the control model's own scores, so its delta is 0. Replace it with a shift-aware method, for example per-site standardization, dropping features that only work on train sites, or simple adaptation.
