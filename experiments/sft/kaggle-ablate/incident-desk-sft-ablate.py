"""Kaggle T4: SFT ablations vs green (20 ep, r=16) — fewer epochs and smaller r."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = Path("/tmp/agent-eval-suite")
OUT = WORK / "sft-ablate"
META = WORK / "run_meta.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Green baseline: 20 epochs, LoRA r=16, n_dup=32.
ABLATIONS = [
    {"name": "ep5_r16", "epochs": 5, "lora_r": 16, "n_dup": 32},
    {"name": "ep20_r8", "epochs": 20, "lora_r": 8, "n_dup": 32},
]


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def load_green():
    green_py = REPO / "experiments" / "sft" / "kaggle-green" / "incident-desk-sft-green.py"
    spec = importlib.util.spec_from_file_location("sft_green", green_py)
    assert spec and spec.loader
    green = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(green)
    return green


def train_sft(
    oracle_path: Path,
    adapter_out: Path,
    *,
    epochs: int,
    lora_r: int,
    n_dup: int,
    run_dir: Path,
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    row = json.loads(oracle_path.read_text().splitlines()[0])
    messages = row["messages"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    prompt_txt = tok.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True
    )
    full_txt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tok(prompt_txt, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_txt, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        raise RuntimeError("chat template produced empty assistant span")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]

    ds = Dataset.from_list(
        [
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
            for _ in range(n_dup)
        ]
    )

    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [tok.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v) for k, v in batch.items()}

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=2 * lora_r,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        ),
    )
    args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        fp16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    train_out = trainer.train()
    adapter_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_out))
    metrics = {
        "train_loss": getattr(train_out, "training_loss", None),
        "epochs": epochs,
        "n_dup": n_dup,
        "lora_r": lora_r,
    }
    (run_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    del model
    torch.cuda.empty_cache()
    print("saved", adapter_out, metrics, flush=True)
    return metrics


def main() -> None:
    t0 = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    import torch

    print("cuda", torch.cuda.is_available(), "n", torch.cuda.device_count(), flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("NO_CUDA")
    print("gpu", torch.cuda.get_device_name(0), flush=True)

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

    oracle_path = REPO / "experiments" / "sft" / "data" / "oracle.jsonl"
    held_path = REPO / "experiments" / "sft" / "data" / "held_out_prompt.json"
    desk_app = REPO / "tasks" / "incident-tool-desk" / "environment" / "app"
    assert oracle_path.is_file() and held_path.is_file()

    OUT.mkdir(parents=True, exist_ok=True)
    err = None
    rc = 0
    try:
        green = load_green()
        summary = {"baseline": {"epochs": 20, "lora_r": 16, "n_dup": 32}, "runs": {}}
        for cfg in ABLATIONS:
            name = cfg["name"]
            run_dir = OUT / name
            run_dir.mkdir(parents=True, exist_ok=True)
            print("START", name, cfg, flush=True)
            metrics = train_sft(
                oracle_path,
                run_dir / "adapter",
                epochs=cfg["epochs"],
                lora_r=cfg["lora_r"],
                n_dup=cfg["n_dup"],
                run_dir=run_dir,
            )
            eval_out = green.eval_adapter(
                adapter=run_dir / "adapter",
                held_path=held_path,
                desk_app=desk_app,
                out_dir=run_dir / "eval",
            )
            row = {
                "config": cfg,
                "train": metrics,
                "eval": eval_out,
                "pass": bool(eval_out.get("generation", {}).get("pass")),
            }
            summary["runs"][name] = row
            print("DONE", name, "pass", row["pass"], "loss", metrics.get("train_loss"), flush=True)

        (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (WORK / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY", json.dumps(summary), flush=True)
    except Exception as e:
        rc = 1
        err = repr(e)
        print("FAILED", err, flush=True)
        raise
    finally:
        meta = {
            "rc": rc,
            "wall_s": round(time.time() - t0, 1),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "error": err,
            "out": str(OUT),
        }
        if (OUT / "summary.json").is_file():
            meta["summary"] = json.loads((OUT / "summary.json").read_text())
        META.write_text(json.dumps(meta, indent=2) + "\n")
        print("META", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
