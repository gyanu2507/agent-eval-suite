"""A correct policy in a different order: check what's running, fix the config,
restart, jump the clock straight to ready_at, then read the logs and submit."""

from __future__ import annotations


def run_policy(desk):
    desk.list_services()
    ticket = desk.get_ticket("INC-2048")
    timeout = desk.get_config("payments.timeout_ms")
    if timeout != 5000:
        desk.set_config("payments.timeout_ms", 5000)

    restart = desk.restart_service("payments")
    wait = restart["ready_at"] - desk.advance_clock(0)["now"]
    desk.advance_clock(max(wait, 0))
    if desk.health("payments")["status"] != "ok":
        raise RuntimeError("payments did not come back")

    logs = desk.get_logs("payments", since_ts=int(ticket["opened_ts"]))
    evidence = [r["id"] for r in sorted(logs, key=lambda r: int(r["ts"])) if "timeout" in r["msg"]]
    return desk.submit_resolution(
        ticket_id=ticket["id"],
        root_cause="config_drift:payments.timeout_ms",
        actions=["set_config", "restart_service"],
        evidence_log_ids=evidence,
    )
