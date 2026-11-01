# Results

| Run | GPU | Wall | Train loss | Env verifier (gen) | Prefers oracle (NLL) | NLL gap (prem − ora) |
| --- | --- | --- | --- | --- | --- | --- |
| base instruct | — | — | — | fail (0 tool calls) | yes | 0.298 |
| SFT-LoRA | Kaggle T4 | 84.6 s | 5.011 | fail (0 tool calls) | yes | 0.298 |
| DPO-LoRA | Kaggle T4 | 88.8 s | 0.693 | aborted / not scored | yes | 0.298 |

Preference protocol: teacher-forced mean NLL of oracle vs premature assistant traces on `held_out_prompt.json`. Lower oracle NLL ⇒ prefers oracle.

Generation protocol: greedy decode then desk replay + verifier. Smoke adapters still do not emit parseable `tool=... args=...` lines, so verifier pass rate is 0 for base/SFT. DPO generation was aborted mid-run on the Mac.

Kernels:
- SFT: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-smoke
- DPO: https://www.kaggle.com/code/gyanumayank/incident-desk-dpo-smoke

Artifacts:
- `experiments/sft/runs/kaggle-t4-sft-lora/`
- `experiments/sft/runs/kaggle-t4-dpo-lora/`
- `experiments/sft/runs/gen-eval/summary.json`

## Ablations

| Change | Effect |
| --- | --- |
| epochs 1 vs 2 | not run yet |
| DPO oracle vs premature | smoke train complete; preference gap ≈ base |

## Failure modes observed

- Kaggle L4 / no CUDA; T4 worked.
- `torchao>=0.16`; CPU collate / `pin_memory=False`.
- DPO: filter TRL `DPOConfig` kwargs; `CUDA_VISIBLE_DEVICES=0`.
- Generation smoke: Qwen chat defaults + short LoRA smokes do not yield verifier-passing tool traces; use preference NLL until a longer train or tool-loop decode.
