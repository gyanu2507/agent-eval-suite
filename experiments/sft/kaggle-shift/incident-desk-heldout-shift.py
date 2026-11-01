"""Kaggle T4: train-ticket vs held-out INC-7777 scenario shift for green adapters."""

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
OUT = WORK / "heldout-shift"
META = WORK / "run_meta.json"


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


def oracle_pass_shift(scenario: Path, desk_app: Path, out_dir: Path) -> dict:
    """Prove the shifted scenario is solvable (oracle, not the model)."""
    sys.path.insert(0, str(desk_app))
    from desk import Desk
    from verifier import score

    desk = Desk.from_scenario_path(scenario)
    ticket = desk.get_ticket("INC-7777")
    opened = int(ticket["opened_ts"])
    pay_logs = desk.get_logs("payments", since_ts=opened)
    pay_logs = sorted(pay_logs, key=lambda row: int(row["ts"]))
    evidence = [
        row["id"]
        for row in pay_logs
        if "timeout" in row["msg"] and row["service"] == "payments"
    ]
    baseline = desk.scenario["baseline_config"]["payments.timeout_ms"]
    current = desk.get_config("payments.timeout_ms")
    if current != baseline:
        desk.set_config("payments.timeout_ms", baseline)
    desk.restart_service("payments")
    while True:
        h = desk.health("payments")
        if h.get("status") == "ok":
            break
        desk.advance_clock(1)
    desk.submit_resolution(
        ticket_id="INC-7777",
        root_cause="config_drift:payments.timeout_ms",
        actions=["set_config", "restart_service"],
        evidence_log_ids=evidence,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "audit.jsonl"
    with audit_path.open("w") as f:
        for row in desk.audit:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    submitted = desk._submitted
    assert submitted is not None
    resolution_path = out_dir / "resolution.json"
    resolution_path.write_text(
        json.dumps(
            {
                "ticket_id": submitted["ticket_id"],
                "root_cause": submitted["root_cause"],
                "actions": submitted["actions"],
                "evidence_log_ids": submitted["evidence_log_ids"],
                "final_payments_health": submitted["final_payments_health"],
                "final_timeout_ms": submitted["final_timeout_ms"],
            },
            indent=2,
        )
        + "\n"
    )
    return score(scenario, audit_path, resolution_path)


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
    dpo_adapter = REPO / "experiments" / "sft" / "runs" / "kaggle-t4-dpo-green" / "adapter"
    train_prompt = REPO / "experiments" / "sft" / "data" / "held_out_prompt.json"
    shift_prompt = REPO / "experiments" / "sft" / "data" / "held_out_prompt_shift.json"
    shift_scenario = REPO / "experiments" / "sft" / "data" / "scenario_shift.json"
    desk_app = REPO / "tasks" / "incident-tool-desk" / "environment" / "app"
    train_scenario = desk_app / "data" / "scenario.json"
    assert sft_adapter.is_dir() and dpo_adapter.is_dir()
    assert shift_prompt.is_file() and shift_scenario.is_file()

    OUT.mkdir(parents=True, exist_ok=True)
    err = None
    rc = 0
    try:
        oracle = oracle_pass_shift(shift_scenario, desk_app, OUT / "oracle_shift")
        print("ORACLE_SHIFT", json.dumps(oracle), flush=True)
        if not oracle.get("pass"):
            raise RuntimeError("shift_scenario_not_oracle_solvable")

        green = load_green()
        results = {"oracle_shift": oracle, "models": {}}
        for name, adapter in [("sft_green", sft_adapter), ("dpo_green", dpo_adapter)]:
            train = green.eval_adapter(
                adapter=adapter,
                held_path=train_prompt,
                desk_app=desk_app,
                out_dir=OUT / name / "train_ticket",
                scenario_path=train_scenario,
            )
            shift = green.eval_adapter(
                adapter=adapter,
                held_path=shift_prompt,
                desk_app=desk_app,
                out_dir=OUT / name / "shift_ticket",
                scenario_path=shift_scenario,
            )
            row = {
                "train_ticket": train,
                "shift_ticket": shift,
                "train_pass": bool(train.get("generation", {}).get("pass")),
                "shift_pass": bool(shift.get("generation", {}).get("pass")),
            }
            results["models"][name] = row
            print(name, "train", row["train_pass"], "shift", row["shift_pass"], flush=True)

        (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        (WORK / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print("SUMMARY", json.dumps(results), flush=True)
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
        META.write_text(json.dumps(meta, indent=2) + "\n")
        print("META", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
