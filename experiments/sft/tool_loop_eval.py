#!/usr/bin/env python3
"""Interactive tool-loop decode against the live desk.

Open-loop SFT invents `->` observations. This loop:
1. generates one `tool=... args=...` line (strips any hallucinated observation),
2. executes it on a real Desk,
3. appends the real `->` observation into the context,
4. repeats until submit_resolution or max steps.

Works with adapters trained on full oracle transcripts (same line format).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESK = ROOT / "tasks" / "incident-tool-desk" / "environment" / "app"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_tool_line(line: str) -> tuple[str, dict] | None:
    raw = line.strip()
    if not raw.startswith("tool="):
        return None
    body = raw[len("tool=") :]
    if " args=" not in body:
        return None
    name, rest = body.split(" args=", 1)
    args_s = rest.split(" -> ", 1)[0].strip()
    try:
        args = json.loads(args_s)
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    return name.strip(), args


def format_tool_line(name: str, args: dict, result: object) -> str:
    return (
        f"tool={name} args={json.dumps(args, sort_keys=True)} "
        f"-> {json.dumps(result, sort_keys=True)}"
    )


def run_tool_loop(
    *,
    model,
    tok,
    messages: list[dict],
    desk,
    max_steps: int = 24,
    max_new_tokens: int = 192,
    device: str = "cuda",
) -> dict:
    """Generate+execute loop. `desk` is a live Desk instance."""
    import torch
    from transformers import GenerationConfig

    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    accepted: list[str] = []
    calls: list[tuple[str, dict]] = []
    errors: list[str] = []
    raw_generations: list[str] = []
    cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )

    for step in range(max_steps):
        context = prompt + ("\n".join(accepted) + ("\n" if accepted else ""))
        inputs = tok(context, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, generation_config=cfg)
        new_text = tok.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        raw_generations.append(new_text)
        # First non-empty line only; ignore anything after (incl. hallucinated ->).
        line = ""
        for piece in new_text.splitlines():
            if piece.strip():
                line = piece.strip()
                break
        if not line:
            errors.append(f"empty_generation_step_{step}")
            break

        parsed = parse_tool_line(line)
        if parsed is None:
            errors.append(f"parse_fail_step_{step}:{line[:160]}")
            break
        name, args = parsed
        fn = getattr(desk, name, None)
        if fn is None:
            errors.append(f"unknown_tool:{name}")
            break
        try:
            result = fn(**args)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}:{type(exc).__name__}:{exc}")
            break

        accepted.append(format_tool_line(name, args, result))
        calls.append((name, args))
        if name == "submit_resolution":
            break
    else:
        errors.append("max_steps")

    return {
        "accepted_lines": accepted,
        "calls": calls,
        "errors": errors,
        "raw_generations": raw_generations,
        "transcript": "\n".join(accepted) + ("\n" if accepted else ""),
    }


def score_loop_result(
    *,
    desk,
    scenario_path: Path,
    desk_app: Path,
    out_dir: Path,
    loop: dict,
    ticket_id: str,
) -> dict:
    if str(desk_app) not in sys.path:
        sys.path.insert(0, str(desk_app))
    from verifier import score  # noqa: WPS433

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generation.txt").write_text(loop["transcript"])
    (out_dir / "raw_steps.json").write_text(
        json.dumps(loop["raw_generations"], indent=2) + "\n"
    )
    (out_dir / "parsed_calls.json").write_text(
        json.dumps(loop["calls"], indent=2) + "\n"
    )

    audit_path = out_dir / "audit.jsonl"
    with audit_path.open("w") as f:
        for row in desk.audit:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    submitted = desk._submitted
    if submitted is None:
        resolution = {
            "ticket_id": ticket_id,
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
            "n_calls_parsed": len(loop["calls"]),
            "replay_errors": loop["errors"],
        }
    else:
        gen = score(scenario_path, audit_path, resolution_path)
        gen["n_calls_parsed"] = len(loop["calls"])
        gen["replay_errors"] = loop["errors"]
    (out_dir / "verifier_score.json").write_text(json.dumps(gen, indent=2) + "\n")
    return gen


def eval_adapter_tool_loop(
    *,
    adapter: Path,
    held_path: Path,
    desk_app: Path,
    out_dir: Path,
    scenario_path: Path | None = None,
    model_id: str = MODEL_ID,
    max_steps: int = 24,
) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(desk_app))
    from desk import Desk

    held = json.loads(held_path.read_text())
    messages = held["messages"]
    scenario = scenario_path or (desk_app / "data" / "scenario.json")
    ticket_id = json.loads(scenario.read_text())["ticket"]["id"]

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.to("cuda")
    model.eval()

    desk = Desk.from_scenario_path(scenario)
    loop = run_tool_loop(
        model=model, tok=tok, messages=messages, desk=desk, max_steps=max_steps
    )
    gen = score_loop_result(
        desk=desk,
        scenario_path=scenario,
        desk_app=desk_app,
        out_dir=out_dir,
        loop=loop,
        ticket_id=ticket_id,
    )
    del model
    torch.cuda.empty_cache()
    return {
        "generation": gen,
        "n_steps": len(loop["calls"]),
        "n_gen_chars": len(loop["transcript"]),
        "scenario": str(scenario),
        "ticket_id": ticket_id,
        "decode": "tool_loop",
    }


def _self_check() -> None:
    """Parse/format + oracle-style scripted loop without torch."""
    sys.path.insert(0, str(DEFAULT_DESK))
    from desk import Desk
    from verifier import score

    line = 'tool=get_ticket args={"ticket_id": "INC-2048"} -> {"id": "FAKE"}'
    parsed = parse_tool_line(line)
    assert parsed == ("get_ticket", {"ticket_id": "INC-2048"})

    scenario = DEFAULT_DESK / "data" / "scenario.json"
    desk = Desk.from_scenario_path(scenario)
    # Scripted args matching oracle order (observations come from desk).
    script = [
        ("get_ticket", {"ticket_id": "INC-2048"}),
        ("get_logs", {"service": "payments", "since_ts": 1000}),
        ("get_config", {"key": "payments.timeout_ms"}),
        ("set_config", {"key": "payments.timeout_ms", "value": 5000}),
        ("restart_service", {"service": "payments"}),
        ("health", {"service": "payments"}),
        ("advance_clock", {"seconds": 1}),
        ("health", {"service": "payments"}),
        ("advance_clock", {"seconds": 1}),
        ("health", {"service": "payments"}),
        ("advance_clock", {"seconds": 1}),
        ("health", {"service": "payments"}),
        ("advance_clock", {"seconds": 1}),
        ("health", {"service": "payments"}),
        ("advance_clock", {"seconds": 1}),
        ("health", {"service": "payments"}),
        (
            "submit_resolution",
            {
                "ticket_id": "INC-2048",
                "root_cause": "config_drift:payments.timeout_ms",
                "actions": ["set_config", "restart_service"],
                "evidence_log_ids": ["L-9", "L-11"],
            },
        ),
    ]
    accepted = []
    for name, args in script:
        result = getattr(desk, name)(**args)
        accepted.append(format_tool_line(name, args, result))
    assert "L-9" in accepted[1] and "L-11" in accepted[1]
    import tempfile

    td = Path(tempfile.mkdtemp())
    audit = td / "audit.jsonl"
    with audit.open("w") as f:
        for row in desk.audit:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    res = td / "resolution.json"
    sub = desk._submitted
    res.write_text(
        json.dumps(
            {
                "ticket_id": sub["ticket_id"],
                "root_cause": sub["root_cause"],
                "actions": sub["actions"],
                "evidence_log_ids": sub["evidence_log_ids"],
                "final_payments_health": sub["final_payments_health"],
                "final_timeout_ms": sub["final_timeout_ms"],
            },
            indent=2,
        )
        + "\n"
    )
    out = score(scenario, audit, res)
    assert out["pass"], out
    print("self_check_ok", len(accepted))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--adapter", type=Path)
    p.add_argument("--held", type=Path)
    p.add_argument("--scenario", type=Path)
    p.add_argument("--desk-app", type=Path, default=DEFAULT_DESK)
    p.add_argument("--out", type=Path, required=False)
    args = p.parse_args()
    if args.self_check:
        _self_check()
        return
    if not args.adapter or not args.held or not args.out:
        raise SystemExit("need --adapter --held --out (or --self-check)")
    summary = eval_adapter_tool_loop(
        adapter=args.adapter,
        held_path=args.held,
        desk_app=args.desk_app,
        out_dir=args.out,
        scenario_path=args.scenario,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
