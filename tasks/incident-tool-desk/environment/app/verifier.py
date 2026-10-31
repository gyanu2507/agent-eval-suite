"""Replay audit.jsonl on a fresh desk; score against scenario truth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from desk import Desk

REQUIRED_ROOT = "config_drift:payments.timeout_ms"
REQUIRED_ACTIONS = {"set_config", "restart_service"}


def replay_audit(scenario_path: Path, audit_path: Path) -> Desk:
    desk = Desk.from_scenario_path(scenario_path)
    for line in audit_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tool = row["tool"]
        args = row.get("args") or {}
        fn = getattr(desk, tool)
        fn(**args)
    return desk


def expected_evidence(scenario: dict[str, Any]) -> list[str]:
    opened = int(scenario["ticket"]["opened_ts"])
    rows = [
        row
        for row in scenario["logs"]
        if row["service"] == "payments"
        and int(row["ts"]) >= opened
        and "timeout" in row["msg"]
    ]
    rows = sorted(rows, key=lambda r: int(r["ts"]))
    return [r["id"] for r in rows]


def score(scenario_path: Path, audit_path: Path, resolution_path: Path) -> dict[str, Any]:
    scenario = json.loads(scenario_path.read_text())
    desk = replay_audit(scenario_path, audit_path)
    resolution = json.loads(resolution_path.read_text())
    baseline = scenario["baseline_config"]["payments.timeout_ms"]
    evidence = expected_evidence(scenario)

    checks = {
        "root_cause": resolution.get("root_cause") == REQUIRED_ROOT,
        "actions": set(resolution.get("actions") or []) >= REQUIRED_ACTIONS,
        "evidence": list(resolution.get("evidence_log_ids") or []) == evidence,
        "timeout_restored": desk.scenario["config"].get("payments.timeout_ms") == baseline,
        "payments_ok": desk._payments_health_status() == "ok",
        "resolution_health": resolution.get("final_payments_health") == "ok",
        "resolution_timeout": resolution.get("final_timeout_ms") == baseline,
        "submitted": any(r.get("tool") == "submit_resolution" for r in desk.audit),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "expected_evidence": evidence,
        "audit_len": len(desk.audit),
    }
