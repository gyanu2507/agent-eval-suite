"""Kaggle GPU smoke: LoRA SFT on incident-tool-desk oracle traces."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = Path("/tmp/agent-eval-suite")
OUT = WORK / "sft-lora"
META = WORK / "run_meta.json"


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    t0 = time.time()
    import torch

    print("cuda", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("NO_CUDA")
    print("gpu", torch.cuda.get_device_name(0), flush=True)

    # Pin stack to avoid peft/torchao mismatch on current Kaggle images.
    sh(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "torchao>=0.16.0",
            "transformers>=4.44",
            "trl>=0.9",
            "peft>=0.13",
            "datasets",
            "accelerate",
        ]
    )

    if REPO.exists():
        sh(["rm", "-rf", str(REPO)])
    sh(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/gyanu2507/agent-eval-suite.git",
            str(REPO),
        ]
    )
    train = REPO / "experiments" / "sft" / "data" / "oracle.jsonl"
    assert train.is_file(), train

    try:
        run_sft(train, OUT)
        rc = 0
        err = None
    except Exception as e:
        rc = 1
        err = repr(e)
        print("SFT_FAILED", err, flush=True)
        raise

    wall = round(time.time() - t0, 1)
    meta = {
        "rc": rc,
        "wall_s": wall,
        "cuda": True,
        "gpu": torch.cuda.get_device_name(0),
        "out_exists": (OUT / "adapter").exists(),
        "out_files": sorted(p.name for p in OUT.rglob("*") if p.is_file())[:40],
        "error": err,
    }
    META.write_text(json.dumps(meta, indent=2) + "\n")
    print("RUN_META", json.dumps(meta), flush=True)
    raise SystemExit(rc)


def run_sft(train: Path, out: Path) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    row = json.loads(train.read_text().splitlines()[0])
    texts = []
    for _ in range(16):
        text = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in row["messages"])
        texts.append({"text": text})
    ds = Dataset.from_list(texts)

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=1536, padding="max_length")

    ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    out.mkdir(parents=True, exist_ok=True)

    def collate(features):
        # Keep on CPU; Trainer moves to GPU. Pinning CUDA tensors crashes.
        batch = {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
        }
        batch["labels"] = batch["input_ids"].clone()
        return batch

    args = TrainingArguments(
        output_dir=str(out / "trainer"),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        fp16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model = get_peft_model(
        model,
        LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM"),
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    train_out = trainer.train()
    model.save_pretrained(str(out / "adapter"))
    (out / "train_metrics.json").write_text(
        json.dumps({"train_loss": getattr(train_out, "training_loss", None)}, indent=2)
        + "\n"
    )
    print("saved", out / "adapter", "loss", getattr(train_out, "training_loss", None), flush=True)


if __name__ == "__main__":
    main()
