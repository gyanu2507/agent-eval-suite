"""Kaggle T4: chat-template SFT (assistant-only loss) + verifier gen-eval.

Goal: at least one green env-verifier cell for the SFT adapter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = Path("/tmp/agent-eval-suite")
OUT = WORK / "sft-green"
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

    oracle_path = REPO / "experiments" / "sft" / "data" / "oracle.jsonl"
    held_path = REPO / "experiments" / "sft" / "data" / "held_out_prompt.json"
    desk_app = REPO / "tasks" / "incident-tool-desk" / "environment" / "app"
    assert oracle_path.is_file() and held_path.is_file()

    OUT.mkdir(parents=True, exist_ok=True)
    err = None
    rc = 0
    try:
        train_sft(oracle_path, OUT / "adapter")
        summary = eval_adapter(
            adapter=OUT / "adapter",
            held_path=held_path,
            desk_app=desk_app,
            out_dir=OUT / "eval",
        )
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (WORK / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY", json.dumps(summary), flush=True)
        # Keep kernel COMPLETE even if verifier is red so artifacts download cleanly.
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


def train_sft(oracle_path: Path, adapter_out: Path) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    row = json.loads(oracle_path.read_text().splitlines()[0])
    messages = row["messages"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Assistant-only loss via chat template (matches Instruct decode path).
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
    assert len(labels) == len(full_ids)

    # Duplicate for a denser smoke; more epochs to memorize the oracle tool trace.
    n_dup = 32
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
    print(
        "train_tokens",
        len(full_ids),
        "prompt_tokens",
        len(prompt_ids),
        "assistant_tokens",
        len(full_ids) - len(prompt_ids),
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
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM"),
    )
    args = TrainingArguments(
        output_dir=str(OUT / "trainer"),
        num_train_epochs=20,
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
        "epochs": 20,
        "n_dup": n_dup,
        "lora_r": 16,
    }
    (OUT / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("saved", adapter_out, metrics, flush=True)


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("tool="):
            continue
        body = line[len("tool=") :]
        if " args=" not in body:
            continue
        name, rest = body.split(" args=", 1)
        args_s = rest.split(" -> ", 1)[0].strip()
        try:
            args = json.loads(args_s)
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict):
            calls.append((name.strip(), args))
    return calls


def eval_adapter(
    *, adapter: Path, held_path: Path, desk_app: Path, out_dir: Path
) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    held = json.loads(held_path.read_text())
    messages = held["messages"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.cuda()
    model.eval()

    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    cfg = GenerationConfig(
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=cfg)
    text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generation.txt").write_text(text + "\n")
    calls = parse_tool_calls(text)
    (out_dir / "parsed_calls.json").write_text(json.dumps(calls, indent=2) + "\n")
    print("parsed_calls", len(calls), flush=True)

    sys.path.insert(0, str(desk_app))
    from desk import Desk
    from verifier import score

    scenario = desk_app / "data" / "scenario.json"
    desk = Desk.from_scenario_path(scenario)
    errors: list[str] = []
    for name, args in calls:
        fn = getattr(desk, name, None)
        if fn is None:
            errors.append(f"unknown_tool:{name}")
            break
        try:
            fn(**args)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}:{type(exc).__name__}:{exc}")
            break
        if name == "submit_resolution":
            break

    audit_path = out_dir / "audit.jsonl"
    with audit_path.open("w") as f:
        for row in desk.audit:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    submitted = desk._submitted
    if submitted is None:
        resolution = {
            "ticket_id": "INC-2048",
            "root_cause": "",
            "actions": [],
            "evidence_log_ids": [],
            "final_payments_health": "",
            "final_timeout_ms": None,
        }
    else:
        resolution = {
            "ticket_id": submitted["ticket_id"],
            "root_cause": submitted["root_cause"],
            "actions": submitted["actions"],
            "evidence_log_ids": submitted["evidence_log_ids"],
            "final_payments_health": submitted["final_payments_health"],
            "final_timeout_ms": submitted["final_timeout_ms"],
        }
    resolution_path = out_dir / "resolution.json"
    resolution_path.write_text(json.dumps(resolution, indent=2) + "\n")

    if not desk.audit:
        gen = {
            "pass": False,
            "error": "empty_audit",
            "n_calls_parsed": len(calls),
            "replay_errors": errors,
        }
    else:
        gen = score(scenario, audit_path, resolution_path)
        gen["n_calls_parsed"] = len(calls)
        gen["replay_errors"] = errors
    (out_dir / "verifier_score.json").write_text(json.dumps(gen, indent=2) + "\n")
    return {"generation": gen, "n_gen_chars": len(text)}


if __name__ == "__main__":
    main()
