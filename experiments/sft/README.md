# SFT mini-study (cloud)

Trajectories from `incident-tool-desk` for a small post-training loop. **Do not train on the 8 GB Mac** with a full torch stack. Use Colab / a cloud GPU.

## Data

Generate:

```bash
python3 experiments/sft/export_trajectories.py
```

Writes `experiments/sft/data/`:

- `oracle.jsonl` — positive demonstrations (label `chosen`)
- `starter.jsonl` — failing traces (label `rejected` for DPO)
- `premature.jsonl` — false-green resolutions (also `rejected`)

Each line is a chat-style record: system + user instruction + assistant tool-call transcript.

## Train (Colab / cloud)

```bash
# on a machine with GPU + torch
pip install 'transformers>=4.44' 'trl>=0.9' datasets accelerate peft
python3 experiments/sft/train_sft_lora.py \
  --train experiments/sft/data/oracle.jsonl \
  --eval experiments/sft/data/held_out_prompt.json \
  --out experiments/sft/runs/sft-lora
```

Optional DPO (oracle vs premature/starter):

```bash
python3 experiments/sft/train_dpo_lora.py \
  --chosen experiments/sft/data/oracle.jsonl \
  --rejected experiments/sft/data/premature.jsonl \
  --out experiments/sft/runs/dpo-lora
```

## Report template

After a run, fill `experiments/sft/RESULTS.md`:

| Run | Env verifier pass rate | Notes |
| --- | --- | --- |
| base model | | |
| SFT-LoRA | | |
| DPO-LoRA (optional) | | |

Ablations: train steps, LoRA rank, include/exclude premature as rejected.
