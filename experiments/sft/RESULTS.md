# Results

| Run | Train GPU | Train wall | Train loss | Verifier gen | Notes |
| --- | --- | --- | --- | --- | --- |
| base instruct | — | — | — | fail (0 tools) | prior gen-eval |
| SFT-LoRA smoke v1 | Kaggle T4 | 84.6 s | 5.011 | fail (0 tools) | raw `<\|role\|>` format |
| DPO-LoRA smoke v1 | Kaggle T4 | 88.8 s | 0.693 | fail (0 tools) | prior |
| **SFT-LoRA green** | Kaggle T4 | 437.4 s | 0.0126 | **pass** open-loop train | single ticket, 20 ep, r=16 |
| SFT ablate ep5_r16 | Kaggle T4 | (shared 516.7 s) | 0.0510 | **pass** | fewer epochs |
| SFT ablate ep20_r8 | Kaggle T4 | (shared 516.7 s) | 0.0192 | **pass** | smaller LoRA r |
| **DPO-LoRA green** | Kaggle T4 | 152.8 s | 0.5205 | **pass** | continue SFT LoRA, 1×4, β=0.5 |
| **DPO-LoRA strong** | Kaggle T4 | 196.2 s | 0.0574 | **pass** | 4×16 pairs (16× steps) |
| **SFT-LoRA multi** | Kaggle T4 | 477.1 s | 0.0241 | open-loop OOD fail | 8 tickets × 4 dups |
| **tool-loop decode** | Kaggle T4 | 919.7 s eval | — | shift pass; unseen residual | green + multi adapters |
| **SFT-LoRA step** | Kaggle T4 | 727.5 s | 0.0101 | **train/shift/unseen all pass** | assistant=call, user=`OBS` |

Kernels:
- SFT green: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-green
- SFT multi: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-multi
- SFT step: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-step
- Tool-loop: https://www.kaggle.com/code/gyanumayank/incident-desk-tool-loop
- DPO green: https://www.kaggle.com/code/gyanumayank/incident-desk-dpo-green
- DPO strong: https://www.kaggle.com/code/gyanumayank/incident-desk-dpo-strong

Artifacts:
- `experiments/sft/runs/kaggle-t4-sft-green/`
- `experiments/sft/runs/kaggle-t4-sft-multi/`
- `experiments/sft/runs/kaggle-t4-sft-step/`
- `experiments/sft/runs/kaggle-t4-tool-loop/`
- `experiments/sft/runs/kaggle-t4-dpo-green/`
- `experiments/sft/runs/kaggle-t4-dpo-strong/`

## Story in one page

1. Open-loop full-transcript SFT invents `->` observations and fails OOD evidence.
2. Interactive tool-loop decode (live desk obs) makes **shift** green even for single-ticket SFT.
3. Step-level SFT trains **only** on assistant tool calls; real results live in user `OBS …` turns. With chat tool-loop decode: train, shift, and unseen all pass.

## Step-level SFT

Data: `build_step_sft.py` → `step_oracle.jsonl` (136 steps across 8 tickets). Decode: `tool_loop_eval.eval_adapter_tool_loop_chat`.

| Eval | Pass | Evidence | Timeout |
| --- | --- | --- | --- |
| train `INC-2048` | **yes** | L-9, L-11 | 5000 |
| shift `INC-7777` | **yes** | E-9, E-11 | 9000 |
| unseen `INC-9999` | **yes** | Z-21, Z-22 | 12000 |

Train: 6 epochs, LoRA r=16, 2× dup of 136 steps, loss ≈ 0.010.

## Earlier ablations (short)

- DPO: continue green LoRA + merged-SFT ref; merge+new-LoRA wiped tools.
- Epoch/rank: 5 ep and r=8 still train-ticket green.
- Tool-loop on transcript-SFT: shift pass; unseen missed baseline value or evidence order until step-SFT.
