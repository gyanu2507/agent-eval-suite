# Site Shift Meter

Synthetic energy-meter fault detection under **site shift**.

Train on sites `S0` and `S1`. Evaluate on held-out site `S2`, whose feature means are shifted. You must produce an artifact that beats a documented control and reports calibration honestly.

## Data

- `environment/app/data/train.csv` — rows from sites S0, S1
- `environment/app/data/eval.csv` — rows from site S2
- Columns: `site,f1,f2,f3,f4,label` with `label` in `{0,1}`

## Required outputs

Write `environment/app/artifacts/report.json` with:

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

1. `control_auroc` must match the frozen logistic control in `baseline.py` within `1e-6` on eval (recomputed by the verifier).
2. `model_auroc` must be **at least 0.08** above that control (`delta_auroc >= 0.08`).
3. `n_eval` must equal the eval row count.
4. Do not edit `data/` or `baseline.py`.

## Starter

`train_and_eval.py` currently copies the control score into `model_auroc` (delta 0). Replace it with a shift-aware method (per-site standardization, dropping features that only work on train sites, simple adaptation, etc.). Keep the control number honest.
