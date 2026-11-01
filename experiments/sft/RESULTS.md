# Results

| Run | Train GPU | Train wall | Train loss | Verifier gen | Prefers oracle | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| base instruct | — | — | — | fail (0 tools) | yes (gap ~0.30) | prior gen-eval |
| SFT-LoRA smoke v1 | Kaggle T4 | 84.6 s | 5.011 | fail (0 tools) | yes | custom `<\|role\|>` format |
| DPO-LoRA smoke | Kaggle T4 | 88.8 s | 0.693 | fail (0 tools) | yes | prior |
| **SFT-LoRA green** | Kaggle T4 | 437.4 s | 0.012551155857568119 | **pass** (17/17 checks) | — | Qwen chat template, assistant-only loss, 20 ep × 32 dup, LoRA r=16 |

Green kernel: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-green  
Artifacts: `experiments/sft/runs/kaggle-t4-sft-green/`

## What made verifier green

- Train with `apply_chat_template` (not raw `<\|role\|>` markers).
- Loss only on assistant tokens.
- Harder smoke: 20 epochs, 32 oracle duplicates, LoRA r=16.
- Decode with the same chat template + greedy generation.

## Ablations still open

| Change | Effect |
| --- | --- |
| fewer epochs / smaller r | unknown |
| DPO on top of green SFT | not run |
| held-out ticket / scenario shift | not run |

## Failure modes observed

- Short v1 SFT/DPO smokes did not emit parseable `tool=` lines.
- Do not run torch train/eval on the 8 GB Mac; use Kaggle.
- Multi-GPU DPO needs `CUDA_VISIBLE_DEVICES=0`.
