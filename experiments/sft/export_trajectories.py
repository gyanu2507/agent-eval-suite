#!/usr/bin/env python3
"""Export incident-tool-desk trajectories to jsonl for SFT/DPO."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "tasks" / "incident-tool-desk" / "environment" / "app"
FIX = ROOT / "docs" / "fixtures" / "incident-tool-desk"
OUT = Path(__file__).resolve().parent / "data"
INSTRUCTION = (ROOT / "tasks" / "incident-tool-desk" / "instruction.md").read_text()

SYSTEM = (
    "You are an incident-response policy. Call desk tools to resolve INC-2048. "
    "Obey the instruction. Prefer numeric log timestamps and wait for health ok."
)


def traj_to_record(traj_path: Path, label: str) -> dict:
    traj = json.loads(traj_path.read_text())
    lines = []
    for step in traj:
        lines.append(
            f"tool={step['tool']} args={json.dumps(step['arguments'], sort_keys=True)} "
            f"-> {json.dumps(step['observation'], sort_keys=True)}"
        )
    assistant = "\n".join(lines)
    return {
        "label": label,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTION.strip()},
            {"role": "assistant", "content": assistant},
        ],
        "source": str(traj_path.relative_to(ROOT)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping = {
        "oracle": (FIX / "oracle-pass" / "trajectory.json", "chosen"),
        "starter": (FIX / "starter-fail" / "trajectory.json", "rejected"),
        "premature": (FIX / "premature-submit" / "trajectory.json", "rejected"),
    }
    for name, (path, label) in mapping.items():
        rec = traj_to_record(path, label)
        out = OUT / f"{name}.jsonl"
        out.write_text(json.dumps(rec) + "\n")
        print("wrote", out, "label", label)

    # Held-out prompt file (same instruction; used for qualitative eval notes)
    held = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTION.strip()},
        ]
    }
    (OUT / "held_out_prompt.json").write_text(json.dumps(held, indent=2) + "\n")
    print("wrote", OUT / "held_out_prompt.json")


if __name__ == "__main__":
    main()
