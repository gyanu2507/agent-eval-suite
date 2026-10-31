#!/usr/bin/env python3
"""Optional LoRA DPO: oracle chosen vs premature/starter rejected. Cloud GPU only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _plain(messages: list[dict]) -> str:
    return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chosen", type=Path, required=True)
    p.add_argument("--rejected", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = p.parse_args()

    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        raise SystemExit(
            "Install transformers, trl, peft, datasets, accelerate on a GPU machine.\n"
            f"Missing import: {e}"
        )

    chosen = json.loads(args.chosen.read_text().splitlines()[0])
    rejected = json.loads(args.rejected.read_text().splitlines()[0])
    prompt = _plain(chosen["messages"][:2])
    rows = []
    for _ in range(8):
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen["messages"][2]["content"],
                "rejected": rejected["messages"][2]["content"],
            }
        )
    ds = Dataset.from_list(rows)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    args.out.mkdir(parents=True, exist_ok=True)
    trainer = DPOTrainer(
        model=model,
        train_dataset=ds,
        peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM"),
        args=DPOConfig(
            output_dir=str(args.out),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            logging_steps=1,
            report_to=[],
        ),
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(str(args.out / "adapter"))
    print("saved", args.out / "adapter")


if __name__ == "__main__":
    main()
