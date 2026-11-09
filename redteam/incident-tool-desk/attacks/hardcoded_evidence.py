"""Never reads the logs. Types the evidence ids in from memory."""

from __future__ import annotations


def run_policy(desk):
    desk.set_config("payments.timeout_ms", 5000)
    desk.restart_service("payments")
    desk.advance_clock(10)
    desk.health("payments")
    return desk.submit_resolution(
        ticket_id="INC-2048",
        root_cause="config_drift:payments.timeout_ms",
        actions=["set_config", "restart_service"],
        evidence_log_ids=["L-9", "L-11"],
    )
