# Results

| Run | Train GPU | Train wall | Train loss | Verifier gen | Prefers oracle | NLL gap (prem − ora) |
| --- | --- | --- | --- | --- | --- | --- |
| base instruct | — | — | — | fail (0 tool calls) | yes | 0.298 |
| SFT-LoRA | Kaggle T4 | 84.6 s | 5.011 | fail (0 tool calls) | yes | 0.297 |
| DPO-LoRA | Kaggle T4 | 88.8 s | 0.693 | fail (0 tool calls) | yes | 0.298 |

Gen-eval kernel (T4, 187 s wall): https://www.kaggle.com/code/gyanumayank/incident-desk-gen-eval  
Summary: `experiments/sft/runs/gen-eval/summary.json`

Preference: teacher-forced mean NLL of oracle vs premature on `held_out_prompt.json`.  
Generation: greedy decode → parse `tool=` lines → desk replay → verifier. Smoke adapters do not emit parseable tool traces yet, so verifier pass rate is 0 for all three.

Train kernels:
- SFT: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-smoke
- DPO: https://www.kaggle.com/code/gyanumayank/incident-desk-dpo-smoke

Artifacts: `experiments/sft/runs/kaggle-t4-sft-lora/`, `kaggle-t4-dpo-lora/`.

## Ablations

| Change | Effect |
| --- | --- |
| epochs 1 vs 2 | not run yet |
| longer train / tool-loop decode | needed for verifier-passing gen |

## Failure modes observed

- Kaggle L4 / no CUDA; T4 worked.
- `torchao>=0.16`; CPU collate / `pin_memory=False`.
- DPO: filter TRL `DPOConfig` kwargs; `CUDA_VISIBLE_DEVICES=0`.
- Do not run torch eval/train on the 8 GB Mac; use Kaggle.
- Short LoRA smokes leave generation format broken; preference NLL still ranks oracle above premature for base/SFT/DPO.
