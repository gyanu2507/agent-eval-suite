# Results

| Run | Train GPU | Train wall | Train loss | Verifier gen | Notes |
| --- | --- | --- | --- | --- | --- |
| base instruct | — | — | — | fail (0 tools) | prior gen-eval |
| SFT-LoRA smoke v1 | Kaggle T4 | 84.6 s | 5.011 | fail (0 tools) | raw `<\|role\|>` format |
| DPO-LoRA smoke v1 | Kaggle T4 | 88.8 s | 0.693 | fail (0 tools) | prior |
| **SFT-LoRA green** | Kaggle T4 | 437.4 s | 0.0126 | **pass** (17/17) | chat template, assistant-only, 20 ep, r=16 |
| SFT ablate ep5_r16 | Kaggle T4 | (shared 516.7 s) | 0.0510 | **pass** (17/17) | fewer epochs vs green |
| SFT ablate ep20_r8 | Kaggle T4 | (shared 516.7 s) | 0.0192 | **pass** (17/17) | smaller LoRA r vs green |
| **DPO-LoRA green** | Kaggle T4 | 152.8 s | 0.5205109715461731 | **pass** (17/17) | continue SFT LoRA, merged-SFT ref, 1 ep, β=0.5 |
| held-out shift (SFT) | Kaggle T4 | 220.0 s eval | — | train **pass**, shift **fail** | INC-7777; evidence ids stuck on L-9/L-11 |
| held-out shift (DPO) | Kaggle T4 | (same kernel) | — | train **pass**, shift **fail** | same evidence fail as SFT |

Kernels:
- SFT green: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-green
- SFT ablate: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-ablate
- DPO green: https://www.kaggle.com/code/gyanumayank/incident-desk-dpo-green
- Held-out shift: https://www.kaggle.com/code/gyanumayank/incident-desk-heldout-shift

Artifacts:
- `experiments/sft/runs/kaggle-t4-sft-green/`
- `experiments/sft/runs/kaggle-t4-sft-ablate/`
- `experiments/sft/runs/kaggle-t4-dpo-green/`
- `experiments/sft/runs/kaggle-t4-heldout-shift/`

## What stayed green under DPO

- Continue the green SFT LoRA (`is_trainable=True`); do **not** merge then attach a fresh random LoRA.
- Use a **merged SFT** checkpoint as `ref_model` (TRL rejects PeftModel refs).
- Tiny step: 1 epoch, 4 pairs, lr `5e-6`, beta `0.5`.

Merge+new-LoRA DPO (v1/v2) wiped tool traces even with gentle hparams.

## SFT epoch / rank ablations

Same chat-template + assistant-only path as green (`n_dup=32`). Train ticket only:

| Config | Train loss | Verifier |
| --- | --- | --- |
| 20 ep, r=16 (green) | 0.0126 | pass |
| 5 ep, r=16 | 0.0510 | pass |
| 20 ep, r=8 | 0.0192 | pass |

Green’s 20 ep / r=16 is stronger than needed for this single-ticket smoke.

## Held-out ticket / scenario shift

Shifted scenario `INC-7777` (new ticket id, clock, log ids `E-*`, baseline timeout `9000`). Oracle still passes. Both green adapters:

- Still pass the **train** ticket (`INC-2048`).
- On shift: pick the new ticket and restore timeout `9000`, but submit evidence `["L-9","L-11"]` instead of `["E-9","E-11"]` → verifier `evidence` fails.

So the green cells are train-ticket competent, not scenario-general.

## Ablations still open

| Change | Effect |
| --- | --- |
| stronger DPO (more steps) while staying green | not run |
