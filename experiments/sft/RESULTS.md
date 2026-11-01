# Results

| Run | Train GPU | Train wall | Train loss | Verifier gen | Notes |
| --- | --- | --- | --- | --- | --- |
| base instruct | — | — | — | fail (0 tools) | prior gen-eval |
| SFT-LoRA smoke v1 | Kaggle T4 | 84.6 s | 5.011 | fail (0 tools) | raw `<\|role\|>` format |
| DPO-LoRA smoke v1 | Kaggle T4 | 88.8 s | 0.693 | fail (0 tools) | prior |
| **SFT-LoRA green** | Kaggle T4 | 437.4 s | 0.0126 | **pass** (17/17) | chat template, assistant-only, 20 ep |
| **DPO-LoRA green** | Kaggle T4 | 152.8 s | 0.5205109715461731 | **pass** (17/17) | continue SFT LoRA, merged-SFT ref, 1 ep, β=0.5 |

Kernels:
- SFT green: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-green
- DPO green: https://www.kaggle.com/code/gyanumayank/incident-desk-dpo-green

Artifacts:
- `experiments/sft/runs/kaggle-t4-sft-green/`
- `experiments/sft/runs/kaggle-t4-dpo-green/`

## What stayed green under DPO

- Continue the green SFT LoRA (`is_trainable=True`); do **not** merge then attach a fresh random LoRA.
- Use a **merged SFT** checkpoint as `ref_model` (TRL rejects PeftModel refs).
- Tiny step: 1 epoch, 4 pairs, lr `5e-6`, beta `0.5`.

Merge+new-LoRA DPO (v1/v2) wiped tool traces even with gentle hparams.

## Ablations still open

| Change | Effect |
| --- | --- |
| fewer SFT epochs / smaller r | unknown |
| held-out ticket / scenario shift | not run |
| stronger DPO (more steps) while staying green | not run |
