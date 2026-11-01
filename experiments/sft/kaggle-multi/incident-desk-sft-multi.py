"""Kaggle T4: multi-scenario SFT so evidence ids are read from get_logs, not memorized."""

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
OUT = WORK / "sft-multi"
META = WORK / "run_meta.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

N_DUP_PER = 4  # 8 scenarios × 4 = 32 rows (same as green's 32 dups of one)
EPOCHS = 20
LORA_R = 16


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


def train_multi(oracle_path: Path, adapter_out: Path) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    rows = [json.loads(line) for line in oracle_path.read_text().splitlines() if line.strip()]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    examples = []
    for row in rows:
        messages = row["messages"]
        prompt_txt = tok.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True
        )
        full_txt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tok(prompt_txt, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_txt, add_special_tokens=False)["input_ids"]
        if len(full_ids) <= len(prompt_ids):
            raise RuntimeError(f"empty assistant span for {row.get('ticket_id')}")
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        ex = {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "ticket_id": row.get("ticket_id"),
        }
        for _ in range(N_DUP_PER):
            examples.append(ex)

    ds = Dataset.from_list(
        [
            {
                "input_ids": e["input_ids"],
                "attention_mask": e["attention_mask"],
                "labels": e["labels"],
            }
            for e in examples
        ]
    )
    print(
        "n_unique",
        len(rows),
        "n_train",
        len(examples),
        "tickets",
        [r.get("ticket_id") for r in rows],
        flush=True,
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
        LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, task_type="CAUSAL_LM"),
    )
    args = TrainingArguments(
        output_dir=str(OUT / "trainer"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        logging_steps=8,
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
        "epochs": EPOCHS,
        "n_unique": len(rows),
        "n_dup_per": N_DUP_PER,
        "n_train": len(examples),
        "lora_r": LORA_R,
    }
    (OUT / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
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

    multi_path = REPO / "experiments" / "sft" / "data" / "multi_oracle.jsonl"
    train_prompt = REPO / "experiments" / "sft" / "data" / "held_out_prompt.json"
    shift_prompt = REPO / "experiments" / "sft" / "data" / "held_out_prompt_shift.json"
    unseen_prompt = REPO / "experiments" / "sft" / "data" / "held_out_prompt_unseen.json"
    shift_scenario = REPO / "experiments" / "sft" / "data" / "scenario_shift.json"
    unseen_scenario = REPO / "experiments" / "sft" / "data" / "scenario_unseen.json"
    desk_app = REPO / "tasks" / "incident-tool-desk" / "environment" / "app"
    train_scenario = desk_app / "data" / "scenario.json"
    assert multi_path.is_file()

    OUT.mkdir(parents=True, exist_ok=True)
    err = None
    rc = 0
    try:
        metrics = train_multi(multi_path, OUT / "adapter")
        green = load_green()
        evals = {}
        for name, prompt, scenario in [
            ("train_ticket", train_prompt, train_scenario),
            ("shift_ticket", shift_prompt, shift_scenario),
            ("unseen_ticket", unseen_prompt, unseen_scenario),
        ]:
            ev = green.eval_adapter(
                adapter=OUT / "adapter",
                held_path=prompt,
                desk_app=desk_app,
                out_dir=OUT / "eval" / name,
                scenario_path=scenario,
            )
            evals[name] = {
                "result": ev,
                "pass": bool(ev.get("generation", {}).get("pass")),
            }
            print(name, "pass", evals[name]["pass"], flush=True)

        summary = {
            "train": metrics,
            "evals": evals,
            "train_pass": evals["train_ticket"]["pass"],
            "shift_pass": evals["shift_ticket"]["pass"],
            "unseen_pass": evals["unseen_ticket"]["pass"],
        }
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
