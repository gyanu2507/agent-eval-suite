#!/usr/bin/env python3
"""Run policy against the desk; write audit + resolution artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from desk import Desk  # noqa: E402
from policy import run_policy  # noqa: E402

SCENARIO = ROOT / "data" / "scenario.json"
OUT = ROOT / "artifacts"


def main() -> None:
    desk = Desk.from_scenario_path(SCENARIO)
    submitted = run_policy(desk)
    OUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUT / "audit.jsonl"
    with audit_path.open("w") as f:
        for row in desk.audit:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    resolution = {
        "ticket_id": submitted["ticket_id"],
        "root_cause": submitted["root_cause"],
        "actions": submitted["actions"],
        "evidence_log_ids": submitted["evidence_log_ids"],
        "final_payments_health": submitted["final_payments_health"],
        "final_timeout_ms": submitted["final_timeout_ms"],
    }
    (OUT / "resolution.json").write_text(json.dumps(resolution, indent=2) + "\n")
    print(json.dumps(resolution, indent=2))


if __name__ == "__main__":
    main()
