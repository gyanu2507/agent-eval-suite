"""Kaggle GPU: preference NLL + generation verifier for base / SFT / DPO."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = Path("/tmp/agent-eval-suite")
OUT = WORK / "gen-eval"
META = WORK / "run_meta.json"


def sh(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> None:
    t0 = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    import torch

    print("cuda", torch.cuda.is_available(), "n_visible", torch.cuda.device_count(), flush=True)
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
            "peft>=0.13",
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

    OUT.mkdir(parents=True, exist_ok=True)
    eval_py = REPO / "experiments" / "sft" / "eval_generation.py"
    assert eval_py.is_file(), eval_py

    rc = 0
    err = None
    try:
        sh(
            [
                sys.executable,
                str(eval_py),
                "--out",
                str(OUT),
                "--runs",
                "base",
                "sft",
                "dpo",
            ]
        )
    except Exception as e:
        rc = 1
        err = repr(e)
        print("EVAL_FAILED", err, flush=True)
        raise

    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    wall = round(time.time() - t0, 1)
    meta = {
        "rc": rc,
        "wall_s": wall,
        "cuda": True,
        "gpu": torch.cuda.get_device_name(0),
        "summary": summary,
        "error": err,
    }
    META.write_text(json.dumps(meta, indent=2) + "\n")
    # Also park summary at working root for easy download
    if summary is not None:
        (WORK / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("RUN_META", json.dumps(meta), flush=True)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
