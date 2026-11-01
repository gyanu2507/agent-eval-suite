"""Kaggle T4: stronger DPO on green SFT LoRA (more steps), then verifier eval."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = Path("/tmp/agent-eval-suite")
OUT = WORK / "dpo-strong"
META = WORK / "run_meta.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Green DPO was 1 epoch × 4 pairs. This is 4 × 16 = 64 steps (16×).
N_PAIRS = 16
EPOCHS = 4
LR = 5e-6
BETA = 0.5


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


def train_dpo_strong(
    sft_adapter: Path, oracle_path: Path, premature_path: Path, adapter_out: Path
) -> dict:
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    chosen = json.loads(oracle_path.read_text().splitlines()[0])
    rejected = json.loads(premature_path.read_text().splitlines()[0])
    prompt = tok.apply_chat_template(
        chosen["messages"][:2], tokenize=False, add_generation_prompt=True
    )
    rows = [
        {
            "prompt": prompt,
            "chosen": chosen["messages"][2]["content"],
            "rejected": rejected["messages"][2]["content"],
        }
        for _ in range(N_PAIRS)
    ]
    ds = Dataset.from_list(rows)

    policy = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    policy = PeftModel.from_pretrained(policy, str(sft_adapter), is_trainable=True)

    ref = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    ref = PeftModel.from_pretrained(ref, str(sft_adapter))
    ref = ref.merge_and_unload()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    cfg_kwargs = {
        "output_dir": str(OUT / "trainer"),
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": 1,
        "learning_rate": LR,
        "logging_steps": 1,
        "report_to": [],
        "save_strategy": "no",
        "fp16": True,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "max_length": 2048,
        "max_prompt_length": 1024,
        "beta": BETA,
    }
    accepted = set(inspect.signature(DPOConfig.__init__).parameters)
    cfg_kwargs = {k: v for k, v in cfg_kwargs.items() if k in accepted}
    print("dpo_config_keys", sorted(cfg_kwargs), flush=True)

    trainer_kwargs = {
        "model": policy,
        "ref_model": ref,
        "train_dataset": ds,
        "args": DPOConfig(**cfg_kwargs),
    }
    sig = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = tok

    trainer = DPOTrainer(**trainer_kwargs)
    train_out = trainer.train()
    adapter_out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(adapter_out))
    metrics = {
        "train_loss": getattr(train_out, "training_loss", None),
        "epochs": EPOCHS,
        "n_pairs": N_PAIRS,
        "approx_steps": EPOCHS * N_PAIRS,
        "lr": LR,
        "beta": BETA,
        "init": "continue_sft_lora_ref_merged_sft",
        "vs_green_dpo": "16x_steps (4ep×16 pairs vs 1ep×4)",
    }
    (OUT / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    del policy, ref
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

    sft_adapter = REPO / "experiments" / "sft" / "runs" / "kaggle-t4-sft-green" / "adapter"
    oracle_path = REPO / "experiments" / "sft" / "data" / "oracle.jsonl"
    premature_path = REPO / "experiments" / "sft" / "data" / "premature.jsonl"
    train_prompt = REPO / "experiments" / "sft" / "data" / "held_out_prompt.json"
    shift_prompt = REPO / "experiments" / "sft" / "data" / "held_out_prompt_shift.json"
    shift_scenario = REPO / "experiments" / "sft" / "data" / "scenario_shift.json"
    desk_app = REPO / "tasks" / "incident-tool-desk" / "environment" / "app"
    train_scenario = desk_app / "data" / "scenario.json"
    assert sft_adapter.is_dir()

    OUT.mkdir(parents=True, exist_ok=True)
    err = None
    rc = 0
    try:
        green = load_green()
        metrics = train_dpo_strong(
            sft_adapter, oracle_path, premature_path, OUT / "adapter"
        )
        train = green.eval_adapter(
            adapter=OUT / "adapter",
            held_path=train_prompt,
            desk_app=desk_app,
            out_dir=OUT / "eval_train",
            scenario_path=train_scenario,
        )
        shift = green.eval_adapter(
            adapter=OUT / "adapter",
            held_path=shift_prompt,
            desk_app=desk_app,
            out_dir=OUT / "eval_shift",
            scenario_path=shift_scenario,
        )
        summary = {
            "train": metrics,
            "train_ticket": train,
            "shift_ticket": shift,
            "train_pass": bool(train.get("generation", {}).get("pass")),
            "shift_pass": bool(shift.get("generation", {}).get("pass")),
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (WORK / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY", json.dumps(summary), flush=True)
        if not summary["train_pass"]:
            err = "verifier_not_green"
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
