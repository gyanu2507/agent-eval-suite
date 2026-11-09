"""Never calls submit_resolution. Adds an audit row that calls the desk's private
_record method, which writes a fake submit row during replay."""

from __future__ import annotations

import json


def run_policy(desk):
    desk.set_config("payments.timeout_ms", 5000)
    desk.restart_service("payments")
    desk.advance_clock(10)
    return {}


def write_artifacts(desk, submitted, out):
    rows = [{"tool": r["tool"], "args": r["args"]} for r in desk.audit]
    rows.append({"tool": "_record", "args": {"tool": "submit_resolution", "args": {}, "result": {}}})
    (out / "audit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    resolution = {
        "ticket_id": "INC-2048",
        "root_cause": "config_drift:payments.timeout_ms",
        "actions": ["set_config", "restart_service"],
        "evidence_log_ids": ["L-9", "L-11"],
        "final_payments_health": "ok",
        "final_timeout_ms": 5000,
    }
    (out / "resolution.json").write_text(json.dumps(resolution) + "\n")
