# Results

| Run | GPU | Wall | Train loss | Env verifier | Notes |
| --- | --- | --- | --- | --- | --- |
| base instruct (no train) | — | — | — | not scored | Qwen2.5-0.5B-Instruct |
| SFT-LoRA on oracle traces | Kaggle Tesla T4 | 84.6 s | 5.011 | TBD (generation eval next) | 16× oracle dupes, 2 epochs, LoRA r=8 |

Kernel: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-smoke  
Artifacts: `experiments/sft/runs/kaggle-t4-sft-lora/`

## Ablations

| Change | Effect |
| --- | --- |
| epochs 1 vs 2 | not run yet |
| DPO oracle vs premature | not run yet |

## Failure modes observed

- Kaggle L4 request fell back / no CUDA on first try; T4 worked.
- peft required `torchao>=0.16` on current Kaggle image.
- Collate must keep tensors on CPU (`dataloader_pin_memory=False`).
