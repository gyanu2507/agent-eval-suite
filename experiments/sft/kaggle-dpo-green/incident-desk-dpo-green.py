"""Kaggle T4: light DPO continuing green SFT LoRA (no merge), then verifier eval."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = Path("/tmp/agent-eval-suite")
OUT = WORK / "dpo-green"
META = WORK / "run_meta.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


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
    held_path = REPO / "experiments" / "sft" / "data" / "held_out_prompt.json"
    desk_app = REPO / "tasks" / "incident-tool-desk" / "environment" / "app"
    assert sft_adapter.is_dir(), sft_adapter

    OUT.mkdir(parents=True, exist_ok=True)
    err = None
    rc = 0
    try:
        # Sanity: green SFT alone must still pass before DPO.
        import importlib.util

        green_py = (
            REPO / "experiments" / "sft" / "kaggle-green" / "incident-desk-sft-green.py"
        )
        spec = importlib.util.spec_from_file_location("sft_green", green_py)
        assert spec and spec.loader
        green = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(green)

        sft_check = green.eval_adapter(
            adapter=sft_adapter,
            held_path=held_path,
            desk_app=desk_app,
            out_dir=OUT / "eval_sft_baseline",
        )
        print("SFT_BASELINE", json.dumps(sft_check), flush=True)
        if not sft_check.get("generation", {}).get("pass"):
            raise RuntimeError("sft_green_baseline_not_passing")

        train_dpo_continue_lora(sft_adapter, oracle_path, premature_path, OUT / "adapter")

        summary = green.eval_adapter(
            adapter=OUT / "adapter",
            held_path=held_path,
            desk_app=desk_app,
            out_dir=OUT / "eval",
        )
        summary["sft_baseline"] = sft_check
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (WORK / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY", json.dumps(summary), flush=True)
        if not summary.get("generation", {}).get("pass"):
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
            "gpu": torch.cuda.get_device_name(0),
            "error": err,
        }
        if (OUT / "summary.json").is_file():
            meta["summary"] = json.loads((OUT / "summary.json").read_text())
        META.write_text(json.dumps(meta, indent=2) + "\n")
        print("RUN_META", json.dumps(meta), flush=True)
    raise SystemExit(rc)


def train_dpo_continue_lora(
    sft_adapter: Path, oracle_path: Path, premature_path: Path, adapter_out: Path
) -> None:
    """Continue the green SFT LoRA with a tiny DPO step (no merge / new random LoRA)."""
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
        for _ in range(4)
    ]
    ds = Dataset.from_list(rows)

    # Policy: continue green SFT LoRA. Ref: merged SFT weights (plain CausalLM).
    # TRL rejects PeftModel as ref_model, so merge a frozen copy for KL.
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
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "learning_rate": 5e-6,
        "logging_steps": 1,
        "report_to": [],
        "save_strategy": "no",
        "fp16": True,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "max_length": 2048,
        "max_prompt_length": 1024,
        "beta": 0.5,
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
    # model already has LoRA; do not pass peft_config

    trainer = DPOTrainer(**trainer_kwargs)
    train_out = trainer.train()
    adapter_out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(adapter_out))
    metrics = {
        "train_loss": getattr(train_out, "training_loss", None),
        "epochs": 1,
        "n_pairs": 4,
        "lr": 5e-6,
        "beta": 0.5,
        "init": "continue_sft_lora_ref_merged_sft",
    }
    (OUT / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("saved", adapter_out, metrics, flush=True)


if __name__ == "__main__":
    main()
