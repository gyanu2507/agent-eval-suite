#!/usr/bin/env python3
"""Build step-level SFT rows: assistant emits tool calls only; observations are user turns.

No `->` observations inside assistant content. Each training example is one next-call
step with real desk results already in the prior user turns.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "tasks" / "incident-tool-desk" / "environment" / "app"
OUT = Path(__file__).resolve().parent / "data"

sys.path.insert(0, str(APP))
from desk import Clock, Desk  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_multi_scenario import (  # noqa: E402
    TRAIN_SPECS,
    instruction_for,
    make_scenario,
    system_for,
)

sys.path.insert(0, str(ROOT / "tasks" / "incident-tool-desk" / "solution"))
import policy as oracle_policy  # noqa: E402


def format_call(name: str, args: dict) -> str:
    return f"tool={name} args={json.dumps(args, sort_keys=True)}"


def format_obs(result: object) -> str:
    return f"OBS {json.dumps(result, sort_keys=True)}"


def trajectory_steps(scenario: dict) -> list[tuple[str, dict, object]]:
    desk = Desk(scenario=deepcopy(scenario), clock=Clock(now=int(scenario["clock_start"])))
    oracle_policy.run_policy(desk)
    steps = []
    for row in desk.audit:
        steps.append((row["tool"], dict(row["args"]), row["result"]))
    return steps


def expand_step_rows(ticket_id: str, baseline: int, steps: list[tuple[str, dict, object]]) -> list[dict]:
    base = [
        {"role": "system", "content": system_for(ticket_id)},
        {"role": "user", "content": instruction_for(ticket_id, baseline)},
    ]
    rows = []
    history: list[dict] = []
    for i, (name, args, result) in enumerate(steps):
        call = format_call(name, args)
        messages = base + history + [{"role": "assistant", "content": call}]
        rows.append(
            {
                "label": "chosen",
                "ticket_id": ticket_id,
                "step": i,
                "tool": name,
                "messages": messages,
            }
        )
        history = history + [
            {"role": "assistant", "content": call},
            {"role": "user", "content": format_obs(result)},
        ]
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    meta = {"n_scenarios": 0, "n_steps": 0, "tickets": []}

    for spec in TRAIN_SPECS:
        scenario = make_scenario(**spec)
        steps = trajectory_steps(scenario)
        rows = expand_step_rows(spec["ticket_id"], spec["baseline"], steps)
        all_rows.extend(rows)
        meta["tickets"].append(
            {
                "ticket_id": spec["ticket_id"],
                "n_steps": len(steps),
                "tools": [s[0] for s in steps],
            }
        )
        print("steps", spec["ticket_id"], len(steps))

    out_path = OUT / "step_oracle.jsonl"
    with out_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    meta["n_scenarios"] = len(TRAIN_SPECS)
    meta["n_steps"] = len(all_rows)
    (OUT / "multi" / "step_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", out_path, "n", len(all_rows))
    # Sanity: assistant contents never contain ' -> '
    bad = [r for r in all_rows if " -> " in r["messages"][-1]["content"]]
    assert not bad, bad[:1]
    print("no_arrow_in_assistant_ok")


if __name__ == "__main__":
    main()
