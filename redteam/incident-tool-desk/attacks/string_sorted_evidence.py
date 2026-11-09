"""Does the fix right but orders evidence by id string, so L-11 lands before L-9."""

from __future__ import annotations


def run_policy(desk):
    logs = sorted(desk.get_logs("payments", since_ts=1000), key=lambda r: str(r["id"]))
    evidence = [r["id"] for r in logs if "timeout" in r["msg"]]
    desk.set_config("payments.timeout_ms", 5000)
    desk.restart_service("payments")
    desk.advance_clock(10)
    desk.health("payments")
    return desk.submit_resolution(
        ticket_id="INC-2048",
        root_cause="config_drift:payments.timeout_ms",
        actions=["set_config", "restart_service"],
        evidence_log_ids=evidence,
    )
