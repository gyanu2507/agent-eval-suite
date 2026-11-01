# Results

| Run | GPU | Wall | Train loss | Env verifier | Notes |
| --- | --- | --- | --- | --- | --- |
| base instruct (no train) | — | — | — | not scored | Qwen2.5-0.5B-Instruct |
| SFT-LoRA on oracle traces | Kaggle Tesla T4 | 84.6 s | 5.011 | TBD (generation eval next) | 16× oracle dupes, 2 epochs, LoRA r=8 |
| DPO-LoRA oracle vs premature | Kaggle Tesla T4 | 88.8 s | 0.693 | TBD (generation eval next) | 8× pairs, 1 epoch, LoRA r=8, single GPU |

Kernels:
- SFT: https://www.kaggle.com/code/gyanumayank/incident-desk-sft-smoke
- DPO: https://www.kaggle.com/code/gyanumayank/incident-desk-dpo-smoke

Artifacts:
- `experiments/sft/runs/kaggle-t4-sft-lora/`
- `experiments/sft/runs/kaggle-t4-dpo-lora/`

## Ablations

| Change | Effect |
| --- | --- |
| epochs 1 vs 2 | not run yet |
| DPO oracle vs premature | smoke complete (table row above) |

## Failure modes observed

- Kaggle L4 request fell back / no CUDA on first try; T4 worked.
- peft required `torchao>=0.16` on current Kaggle image.
- Collate must keep tensors on CPU (`dataloader_pin_memory=False`).
- DPO: TRL dropped `max_prompt_length`; filter DPOConfig kwargs by signature.
- DPO: T4 image can expose 2 GPUs; set `CUDA_VISIBLE_DEVICES=0` (DPOTrainer rejects DataParallel).
