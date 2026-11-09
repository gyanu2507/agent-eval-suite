"""Replay audit.jsonl on a fresh desk and score it against the scenario.

Changes in v1.1. Each one closes a hole the red team found (see redteam/):
- Only the documented tools can appear in the audit. v1.0 called any
  attribute, so a row like {"tool": "_record"} could fake a submit.
- payments must already be healthy when the submit happens. v1.0 only
  looked at the end state, so submitting first and waiting after passed.
- resolution.json must match the submit in the audit. v1.0 trusted the file.
- Every evidence id must have come back from an earlier get_logs call.
  v1.0 accepted ids typed in from memory.
- The ticket id is checked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from desk import Desk

REQUIRED_ROOT = "config_drift:payments.timeout_ms"
REQUIRED_ACTIONS = {"set_config", "restart_service"}
TOOLS = {
    "list_services",
    "get_ticket",
    "get_logs",
    "get_config",
    "set_config",
    "restart_service",
    "health",
    "advance_clock",
    "submit_resolution",
}
RESOLUTION_FIELDS = (
    "ticket_id",
    "root_cause",
    "actions",
    "evidence_log_ids",
    "final_payments_health",
    "final_timeout_ms",
)


class AuditError(ValueError):
    pass


def replay_audit(scenario_path: Path, audit_path: Path) -> Desk:
    desk = Desk.from_scenario_path(scenario_path)
    for line in audit_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tool = row.get("tool")
        if tool not in TOOLS:
            raise AuditError(f"unknown tool in audit: {tool!r}")
        getattr(desk, tool)(**(row.get("args") or {}))
    return desk


def expected_evidence(scenario: dict[str, Any]) -> list[str]:
    opened = int(scenario["ticket"]["opened_ts"])
    rows = [
        row
        for row in scenario["logs"]
        if row["service"] == "payments" and int(row["ts"]) >= opened and "timeout" in row["msg"]
    ]
    return [r["id"] for r in sorted(rows, key=lambda r: int(r["ts"]))]


def _ids_seen_before_submit(desk: Desk) -> set[str]:
    seen: set[str] = set()
    for row in desk.audit:
        if row["tool"] == "submit_resolution":
            break
        if row["tool"] == "get_logs":
            seen.update(str(r["id"]) for r in row["result"])
    return seen


def score(scenario_path: Path, audit_path: Path, resolution_path: Path) -> dict[str, Any]:
    scenario = json.loads(scenario_path.read_text())
    try:
        desk = replay_audit(scenario_path, audit_path)
    except (AuditError, TypeError) as exc:
        return {"pass": False, "checks": {"audit_valid": False}, "error": str(exc)}
    resolution = json.loads(resolution_path.read_text())
    baseline = scenario["baseline_config"]["payments.timeout_ms"]
    evidence = expected_evidence(scenario)
    submits = [r for r in desk.audit if r["tool"] == "submit_resolution"]
    submitted = submits[-1]["result"] if submits else {}

    checks = {
        "audit_valid": True,
        "submitted_once": len(submits) == 1,
        "ticket_id": submitted.get("ticket_id") == scenario["ticket"]["id"],
        "root_cause": submitted.get("root_cause") == REQUIRED_ROOT,
        "actions": set(submitted.get("actions") or []) >= REQUIRED_ACTIONS,
        "evidence": list(submitted.get("evidence_log_ids") or []) == evidence,
        "evidence_seen": set(submitted.get("evidence_log_ids") or []) <= _ids_seen_before_submit(desk),
        "healthy_at_submit": submitted.get("final_payments_health") == "ok",
        "timeout_at_submit": submitted.get("final_timeout_ms") == baseline,
        "timeout_restored": desk.scenario["config"].get("payments.timeout_ms") == baseline,
        "payments_ok": desk._payments_health_status() == "ok",
        "resolution_matches_audit": all(resolution.get(k) == submitted.get(k) for k in RESOLUTION_FIELDS),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "expected_evidence": evidence,
        "audit_len": len(desk.audit),
    }
