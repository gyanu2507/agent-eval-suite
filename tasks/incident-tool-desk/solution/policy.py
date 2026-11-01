"""Oracle policy: config rollback → payments restart → wait → evidenced submit."""

from __future__ import annotations

import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "environment" / "app"
sys.path.insert(0, str(APP))

from desk import Desk  # noqa: E402


def run_policy(desk: Desk) -> dict:
    ticket_id = desk.scenario["ticket"]["id"]
    ticket = desk.get_ticket(ticket_id)
    opened = int(ticket["opened_ts"])

    pay_logs = desk.get_logs("payments", since_ts=opened)
    pay_logs = sorted(pay_logs, key=lambda row: int(row["ts"]))
    evidence = [
        row["id"]
        for row in pay_logs
        if "timeout" in row["msg"] and row["service"] == "payments"
    ]

    baseline = desk.scenario["baseline_config"]["payments.timeout_ms"]
    current = desk.get_config("payments.timeout_ms")
    if current != baseline:
        desk.set_config("payments.timeout_ms", baseline)

    desk.restart_service("payments")
    while True:
        h = desk.health("payments")
        if h.get("status") == "ok":
            break
        desk.advance_clock(1)

    return desk.submit_resolution(
        ticket_id=ticket_id,
        root_cause="config_drift:payments.timeout_ms",
        actions=["set_config", "restart_service"],
        evidence_log_ids=evidence,
    )
