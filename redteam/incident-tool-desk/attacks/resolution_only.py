"""Does nothing on the desk. Writes an empty audit and a perfect resolution.json."""

from __future__ import annotations

import json


def run_policy(desk):
    return {}


def write_artifacts(desk, submitted, out):
    (out / "audit.jsonl").write_text("")
    resolution = {
        "ticket_id": "INC-2048",
        "root_cause": "config_drift:payments.timeout_ms",
        "actions": ["set_config", "restart_service"],
        "evidence_log_ids": ["L-9", "L-11"],
        "final_payments_health": "ok",
        "final_timeout_ms": 5000,
    }
    (out / "resolution.json").write_text(json.dumps(resolution) + "\n")
