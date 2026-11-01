# SFT mini-study (cloud)

Trajectories from `incident-tool-desk` for a small post-training loop. **Do not train on the 8 GB Mac** with a full torch stack. Use Colab / a cloud GPU.

## Data

Single-ticket export (legacy smokes):

```bash
python3 experiments/sft/export_trajectories.py
```

Multi-scenario export (varied ticket / log-id prefixes; preferred for generalization):

```bash
python3 experiments/sft/build_multi_scenario.py
```

Writes `experiments/sft/data/`:

- `oracle.jsonl` — single-ticket positive demos
- `multi_oracle.jsonl` — 8 train tickets with distinct evidence id prefixes
- `scenario_shift.json` / `scenario_unseen.json` — holdouts (`INC-7777`, `INC-9999`)
- `starter.jsonl` / `premature.jsonl` — rejected traces for DPO

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
