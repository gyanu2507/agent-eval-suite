# Site Shift Meter

Synthetic energy-meter fault detection under **site shift**.

Train on sites `S0` and `S1`. Evaluate on held-out site `S2`, whose feature means are shifted. You must produce an artifact that beats a documented control and reports calibration honestly.

## Data

- `environment/app/data/train.csv`: rows from sites S0 and S1
- `environment/app/data/eval.csv`: rows from site S2
- Columns: `site,f1,f2,f3,f4,label` with `label` in `{0,1}`

## Required outputs

Write two files under `environment/app/artifacts/`.

`scores.csv` has a `score` header and one row per eval row, in the same order as `eval.csv`. Higher means more likely to be a fault.

`report.json` has:

```json
{
  "control_auroc": <float>,
  "model_auroc": <float>,
  "delta_auroc": <float>,
  "n_eval": <int>,
  "method": "<short name>",
  "notes": "<one sentence on shift handling>"
}
```

## Gate

1. `control_auroc` must match the frozen logistic control in `baseline.py` within `1e-6` on eval. The verifier recomputes it.
2. The verifier recomputes model AUROC from `scores.csv`. `model_auroc` in the report must match it within `1e-6`.
3. Recomputed model AUROC must be at least 0.5 above the control.
4. `n_eval` must equal the eval row count.
5. Do not edit `data/` or `baseline.py`.

## Starter

`train_and_eval.py` currently writes the control model's scores and copies the control AUROC into `model_auroc` (delta 0). Replace it with a shift-aware method (per-site standardization, dropping features that only work on train sites, simple adaptation, etc.). Keep the control number honest.
