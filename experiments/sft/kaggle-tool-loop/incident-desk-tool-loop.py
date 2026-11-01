"""Kaggle T4: interactive tool-loop decode on multi (and green) SFT adapters."""

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
OUT = WORK / "tool-loop"
META = WORK / "run_meta.json"


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def load_tool_loop():
    path = REPO / "experiments" / "sft" / "tool_loop_eval.py"
    spec = importlib.util.spec_from_file_location("tool_loop_eval", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

    # Local self-check (no GPU model) before heavy eval.
    sh([sys.executable, str(REPO / "experiments" / "sft" / "tool_loop_eval.py"), "--self-check"])

    tl = load_tool_loop()
    desk_app = REPO / "tasks" / "incident-tool-desk" / "environment" / "app"
    train_prompt = REPO / "experiments" / "sft" / "data" / "held_out_prompt.json"
    shift_prompt = REPO / "experiments" / "sft" / "data" / "held_out_prompt_shift.json"
    unseen_prompt = REPO / "experiments" / "sft" / "data" / "held_out_prompt_unseen.json"
    train_scenario = desk_app / "data" / "scenario.json"
    shift_scenario = REPO / "experiments" / "sft" / "data" / "scenario_shift.json"
    unseen_scenario = REPO / "experiments" / "sft" / "data" / "scenario_unseen.json"

    adapters = {
        "sft_multi": REPO / "experiments" / "sft" / "runs" / "kaggle-t4-sft-multi" / "adapter",
        "sft_green": REPO / "experiments" / "sft" / "runs" / "kaggle-t4-sft-green" / "adapter",
    }
    evals_spec = [
        ("train_ticket", train_prompt, train_scenario),
        ("shift_ticket", shift_prompt, shift_scenario),
        ("unseen_ticket", unseen_prompt, unseen_scenario),
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    err = None
    rc = 0
    try:
        summary = {"decode": "tool_loop", "models": {}}
        for model_name, adapter in adapters.items():
            assert adapter.is_dir(), adapter
            rows = {}
            for eval_name, prompt, scenario in evals_spec:
                out_dir = OUT / model_name / eval_name
                print("EVAL", model_name, eval_name, flush=True)
                result = tl.eval_adapter_tool_loop(
                    adapter=adapter,
                    held_path=prompt,
                    desk_app=desk_app,
                    out_dir=out_dir,
                    scenario_path=scenario,
                )
                rows[eval_name] = {
                    "result": result,
                    "pass": bool(result.get("generation", {}).get("pass")),
                }
                print(
                    model_name,
                    eval_name,
                    "pass",
                    rows[eval_name]["pass"],
                    "steps",
                    result.get("n_steps"),
                    flush=True,
                )
            summary["models"][model_name] = rows

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
