#!/usr/bin/env python3
"""Score base / SFT / DPO on incident-tool-desk.

Two protocols:

1. **generation** — sample a tool transcript, replay on the desk, run the
   programmatic verifier (pass/fail).
2. **preference** — teacher-forced NLL of oracle vs premature assistant
   traces given the shared prompt. Lower NLL on oracle means the checkpoint
   ranks the good trace higher (useful when generation still fails format).

Loads one checkpoint at a time for 8 GB machines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parents[2]
DESK_APP = ROOT / "tasks" / "incident-tool-desk" / "environment" / "app"
SCENARIO = DESK_APP / "data" / "scenario.json"
DATA = ROOT / "experiments" / "sft" / "data"
DEFAULT_OUT = ROOT / "experiments" / "sft" / "runs" / "gen-eval"


def _role_text(messages: list[dict]) -> str:
    return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages)


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
        if not isinstance(args, dict):
            continue
        calls.append((name.strip(), args))
    return calls


def replay_and_score(calls: list[tuple[str, dict]], out_dir: Path) -> dict:
    sys.path.insert(0, str(DESK_APP))
    from desk import Desk  # noqa: WPS433
    from verifier import score  # noqa: WPS433

    desk = Desk.from_scenario_path(SCENARIO)
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

    out_dir.mkdir(parents=True, exist_ok=True)
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
        result = {
            "pass": False,
            "checks": {},
            "error": "empty_audit",
            "n_calls_parsed": len(calls),
            "replay_errors": errors,
        }
    else:
        result = score(SCENARIO, audit_path, resolution_path)
        result["n_calls_parsed"] = len(calls)
        result["replay_errors"] = errors
    (out_dir / "verifier_score.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _load_model(model_id: str, adapter: Path | None):
    # float32 on CPU/MPS: float16 on MPS produced early <|endoftext|> on this Mac.
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, device


def _unload(model, device: str) -> None:
    del model
    if device == "mps":
        torch.mps.empty_cache()


def mean_nll(model, tok, prompt: str, completion: str, device: str) -> float:
    """Mean token NLL of completion given prompt (teacher forced)."""
    prompt_ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    comp_ids = tok(completion, add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = torch.cat([prompt_ids, comp_ids], dim=1).to(device)
    labels = input_ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=labels).loss
    return float(loss.item())


def generate_text(model, tok, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    with torch.no_grad():
        out = model.generate(**inputs, generation_config=cfg)
    new = out[0][inputs["input_ids"].shape[1] :]
    return tok.decode(new, skip_special_tokens=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--runs", nargs="+", default=["base", "sft", "dpo"])
    p.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only run preference NLL (faster).",
    )
    args = p.parse_args()

    held = json.loads((DATA / "held_out_prompt.json").read_text())
    prompt = _role_text(held["messages"]) + "<|assistant|>\n"
    oracle = json.loads((DATA / "oracle.jsonl").read_text().splitlines()[0])
    premature = json.loads((DATA / "premature.jsonl").read_text().splitlines()[0])
    oracle_comp = oracle["messages"][2]["content"]
    premature_comp = premature["messages"][2]["content"]

    adapters = {
        "base": None,
        "sft": ROOT / "experiments" / "sft" / "runs" / "kaggle-t4-sft-lora" / "adapter",
        "dpo": ROOT / "experiments" / "sft" / "runs" / "kaggle-t4-dpo-lora" / "adapter",
    }

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    summary: dict[str, dict] = {}
    for name in args.runs:
        adapter = adapters[name]
        if adapter is not None and not adapter.is_dir():
            raise SystemExit(f"missing adapter: {adapter}")
        print(f"=== {name} ===", flush=True)
        model, device = _load_model(args.model, adapter)

        nll_oracle = mean_nll(model, tok, prompt, oracle_comp, device)
        nll_premature = mean_nll(model, tok, prompt, premature_comp, device)
        pref = {
            "nll_oracle": nll_oracle,
            "nll_premature": nll_premature,
            "prefers_oracle": nll_oracle < nll_premature,
            "nll_gap_premature_minus_oracle": nll_premature - nll_oracle,
        }
        print("preference", json.dumps(pref), flush=True)

        gen_result = None
        if not args.skip_generation:
            text = generate_text(model, tok, prompt, device, args.max_new_tokens)
            run_dir = args.out / name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "generation.txt").write_text(text + "\n")
            calls = parse_tool_calls(text)
            (run_dir / "parsed_calls.json").write_text(json.dumps(calls, indent=2) + "\n")
            print(f"parsed {len(calls)} tool calls", flush=True)
            score = replay_and_score(calls, run_dir)
            gen_result = {
                "pass": score.get("pass"),
                "checks": score.get("checks"),
                "n_calls_parsed": score.get("n_calls_parsed"),
                "audit_len": score.get("audit_len"),
                "replay_errors": score.get("replay_errors"),
                "error": score.get("error"),
            }
            print("generation", json.dumps(gen_result), flush=True)

        _unload(model, device)
        summary[name] = {"preference": pref, "generation": gen_result}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
