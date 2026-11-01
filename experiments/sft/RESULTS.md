# Results

| Run | Train GPU | Train wall | Train loss | Verifier gen | Notes |
| --- | --- | --- | --- | --- | --- |
| base instruct | — | — | — | fail (0 tools) | prior gen-eval |
| SFT-LoRA smoke v1 | Kaggle T4 | 84.6 s | 5.011 | fail (0 tools) | raw `<\|role\|>` format |
| DPO-LoRA smoke v1 | Kaggle T4 | 88.8 s | 0.693 | fail (0 tools) | prior |
| **SFT-LoRA green** | Kaggle T4 | 437.4 s | 0.0126 | **pass** (17/17) open-loop | single ticket, 20 ep, r=16 |
| SFT ablate ep5_r16 | Kaggle T4 | (shared 516.7 s) | 0.0510 | **pass** (17/17) | fewer epochs |
| SFT ablate ep20_r8 | Kaggle T4 | (shared 516.7 s) | 0.0192 | **pass** (17/17) | smaller LoRA r |
| **DPO-LoRA green** | Kaggle T4 | 152.8 s | 0.5205 | **pass** (17/17) | continue SFT LoRA, 1×4, β=0.5 |
| **DPO-LoRA strong** | Kaggle T4 | 196.2 s | 0.0574 | **pass** (17/17) | 4×16 pairs (16× steps) |
| **SFT-LoRA multi** | Kaggle T4 | 477.1 s | 0.0241 | open-loop: train pass, OOD fail | 8 tickets × 4 dups |
| **tool-loop decode** | Kaggle T4 | 919.7 s eval | — | see table below | live desk obs between calls |

Kernels:
- SFT green: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-green
- SFT ablate: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-ablate
- SFT multi: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-multi
- Tool-loop: https://www.kaggle.com/code/gyanumayank/incident-desk-tool-loop
- DPO green: https://www.kaggle.com/code/gyanumayank/incident-desk-dpo-green
- DPO strong: https://www.kaggle.com/code/gyanumayank/incident-desk-dpo-strong
- Held-out shift: https://www.kaggle.com/code/gyanumayank/incident-desk-heldout-shift

Artifacts:
- `experiments/sft/runs/kaggle-t4-sft-green/`
- `experiments/sft/runs/kaggle-t4-sft-ablate/`
- `experiments/sft/runs/kaggle-t4-sft-multi/`
- `experiments/sft/runs/kaggle-t4-tool-loop/`
- `experiments/sft/runs/kaggle-t4-dpo-green/`
- `experiments/sft/runs/kaggle-t4-dpo-strong/`
- `experiments/sft/runs/kaggle-t4-heldout-shift/`

## What stayed green under DPO

- Continue the green SFT LoRA (`is_trainable=True`); do **not** merge then attach a fresh random LoRA.
- Use a **merged SFT** checkpoint as `ref_model` (TRL rejects PeftModel refs).
- Green DPO: 1 epoch, 4 pairs, lr `5e-6`, beta `0.5`.
- Stronger DPO (16× steps): still train-ticket green under open-loop.

Merge+new-LoRA DPO (v1/v2) wiped tool traces even with gentle hparams.

## SFT epoch / rank ablations

| Config | Train loss | Verifier (train ticket) |
| --- | --- | --- |
| 20 ep, r=16 (green) | 0.0126 | pass |
| 5 ep, r=16 | 0.0510 | pass |
| 20 ep, r=8 | 0.0192 | pass |

## Multi-scenario SFT (open-loop decode)

| Eval | Pass | Evidence submitted | Expected |
| --- | --- | --- | --- |
| train `INC-2048` | yes | L-9, L-11 | L-9, L-11 |
| shift `INC-7777` | no | B-3, B-15 | E-9, E-11 |
| unseen `INC-9999` | no | D-8, D-17 | Z-21, Z-22 |

Open-loop invents `->` observations, then copies fake ids into submit.

## Interactive tool-loop decode

Code: `experiments/sft/tool_loop_eval.py`. Each step: generate one `tool=` line, strip hallucinated `->`, execute on the live desk, append the **real** observation, continue.

| Adapter | train | shift `INC-7777` | unseen `INC-9999` |
| --- | --- | --- | --- |
| sft_multi | **pass** | **pass** (`E-9`,`E-11`) | fail (timeout set to 1000, not 12000; evidence ids correct) |
| sft_green | **pass** | **pass** (`E-9`,`E-11`) | fail (evidence order `Z-22`,`Z-21` vs `Z-21`,`Z-22`) |

Tool-loop is what makes shift green. Multi-ticket train is not required for that; even single-ticket green SFT copies real `E-*` ids once observations are grounded. Remaining unseen misses are baseline value / evidence sort order, not invented log prefixes.
