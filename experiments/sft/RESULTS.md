# Results

| Run | Train GPU | Train wall | Train loss | Verifier gen | Notes |
| --- | --- | --- | --- | --- | --- |
| base instruct | — | — | — | fail (0 tools) | prior gen-eval |
| SFT-LoRA smoke v1 | Kaggle T4 | 84.6 s | 5.011 | fail (0 tools) | raw `<\|role\|>` format |
| DPO-LoRA smoke v1 | Kaggle T4 | 88.8 s | 0.693 | fail (0 tools) | prior |
| **SFT-LoRA green** | Kaggle T4 | 437.4 s | 0.0126 | **pass** (17/17) | chat template, assistant-only, 20 ep, r=16 |
| SFT ablate ep5_r16 | Kaggle T4 | (shared 516.7 s) | 0.0510 | **pass** (17/17) | fewer epochs vs green |
| SFT ablate ep20_r8 | Kaggle T4 | (shared 516.7 s) | 0.0192 | **pass** (17/17) | smaller LoRA r vs green |
| **DPO-LoRA green** | Kaggle T4 | 152.8 s | 0.5205 | **pass** (17/17) | continue SFT LoRA, 1 ep × 4 pairs, β=0.5 |
| **DPO-LoRA strong** | Kaggle T4 | 196.2 s | 0.0574 | **pass** (17/17) | same recipe, 4 ep × 16 pairs (16× steps) |
| held-out shift (SFT) | Kaggle T4 | 220.0 s eval | — | train **pass**, shift **fail** | evidence ids stuck on L-9/L-11 |
| held-out shift (DPO) | Kaggle T4 | (same kernel) | — | train **pass**, shift **fail** | same evidence fail as SFT |
| held-out shift (DPO-strong) | Kaggle T4 | (same as DPO-strong) | — | train **pass**, shift **fail** | still L-9/L-11 |

Kernels:
- SFT green: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-green
- SFT ablate: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-ablate
- DPO green: https://www.kaggle.com/code/gyanumayank/incident-desk-dpo-green
- DPO strong: https://www.kaggle.com/code/gyanumayank/incident-desk-dpo-strong
- Held-out shift: https://www.kaggle.com/code/gyanumayank/incident-desk-heldout-shift

Artifacts:
- `experiments/sft/runs/kaggle-t4-sft-green/`
- `experiments/sft/runs/kaggle-t4-sft-ablate/`
- `experiments/sft/runs/kaggle-t4-dpo-green/`
- `experiments/sft/runs/kaggle-t4-dpo-strong/`
- `experiments/sft/runs/kaggle-t4-heldout-shift/`

## What stayed green under DPO

- Continue the green SFT LoRA (`is_trainable=True`); do **not** merge then attach a fresh random LoRA.
- Use a **merged SFT** checkpoint as `ref_model` (TRL rejects PeftModel refs).
- Green DPO: 1 epoch, 4 pairs, lr `5e-6`, beta `0.5`.
- Stronger DPO (16× steps): still train-ticket green; shift still fails the same way.

Merge+new-LoRA DPO (v1/v2) wiped tool traces even with gentle hparams.

## SFT epoch / rank ablations

Same chat-template + assistant-only path as green (`n_dup=32`). Train ticket only:

| Config | Train loss | Verifier |
| --- | --- | --- |
| 20 ep, r=16 (green) | 0.0126 | pass |
| 5 ep, r=16 | 0.0510 | pass |
| 20 ep, r=8 | 0.0192 | pass |

## Held-out ticket / scenario shift

Shifted scenario `INC-7777` (new ticket id, clock, log ids `E-*`, baseline timeout `9000`). Oracle still passes. Green SFT, green DPO, and stronger DPO:

- Pass the **train** ticket (`INC-2048`).
- On shift: use the new ticket and timeout `9000`, but submit evidence `["L-9","L-11"]` instead of `["E-9","E-11"]`.

Train-ticket green ≠ scenario-general.

## Ablations still open

None from the planned mini-study list.
