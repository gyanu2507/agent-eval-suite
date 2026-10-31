"""Kaggle GPU smoke: LoRA DPO (oracle chosen vs premature rejected)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = Path("/tmp/agent-eval-suite")
OUT = WORK / "dpo-lora"
META = WORK / "run_meta.json"


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _plain(messages: list[dict]) -> str:
    return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)


def main() -> None:
    t0 = time.time()
    import torch

    print("cuda", torch.cuda.is_available(), flush=True)
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
    chosen_path = REPO / "experiments" / "sft" / "data" / "oracle.jsonl"
    rejected_path = REPO / "experiments" / "sft" / "data" / "premature.jsonl"
    assert chosen_path.is_file(), chosen_path
    assert rejected_path.is_file(), rejected_path

    try:
        run_dpo(chosen_path, rejected_path, OUT)
        rc = 0
        err = None
    except Exception as e:
        rc = 1
        err = repr(e)
        print("DPO_FAILED", err, flush=True)
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


def run_dpo(chosen_path: Path, rejected_path: Path, out: Path) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    chosen = json.loads(chosen_path.read_text().splitlines()[0])
    rejected = json.loads(rejected_path.read_text().splitlines()[0])
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

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    import inspect

    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

    cfg_kwargs = {
        "output_dir": str(out / "trainer"),
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "logging_steps": 1,
        "report_to": [],
        "save_strategy": "no",
        "fp16": True,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "max_length": 2048,
        "max_prompt_length": 1024,
    }
    # TRL renames length knobs across versions; keep only accepted fields.
    accepted = set(inspect.signature(DPOConfig.__init__).parameters)
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in accepted}
    print("dpo_config_keys", sorted(cfg_kwargs), flush=True)

    trainer_kwargs = {
        "model": model,
        "train_dataset": ds,
        "peft_config": LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM"),
        "args": DPOConfig(**cfg_kwargs),
    }
    # Older TRL used tokenizer=; newer uses processing_class=.
    trainer_sig = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tok

    trainer = DPOTrainer(**trainer_kwargs)
    train_out = trainer.train()
    trainer.save_model(str(out / "adapter"))
    metrics = {"train_loss": getattr(train_out, "training_loss", None)}
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("saved", out / "adapter", "loss", metrics["train_loss"], flush=True)


if __name__ == "__main__":
    main()
