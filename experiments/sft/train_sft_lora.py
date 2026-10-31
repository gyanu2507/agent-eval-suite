#!/usr/bin/env python3
"""LoRA SFT on oracle trajectories. Run on Colab/cloud with a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--epochs", type=int, default=2)
    args = p.parse_args()

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise SystemExit(
            "Install transformers, trl, peft, datasets, accelerate on a GPU machine.\n"
            f"Missing import: {e}"
        )

    row = json.loads(args.train.read_text().splitlines()[0])
    texts = []
    # Tiny demo set: duplicate oracle a few times so the trainer has batches.
    for _ in range(16):
        msgs = row["messages"]
        text = ""
        for m in msgs:
            text += f"<|{m['role']}|>\n{m['content']}\n"
        texts.append({"text": text})

    ds = Dataset.from_list(texts)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    args.out.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM"),
        args=SFTConfig(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            logging_steps=1,
            report_to=[],
        ),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(str(args.out / "adapter"))
    meta = {"model": args.model, "train": str(args.train), "epochs": args.epochs}
    (args.out / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("saved", args.out / "adapter")


if __name__ == "__main__":
    main()
