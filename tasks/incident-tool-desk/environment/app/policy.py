"""Buggy starter policy: looks decisive, fails the contract."""

from __future__ import annotations

from desk import Desk


def run_policy(desk: Desk) -> dict:
    ticket = desk.get_ticket("INC-2048")
    hint = ticket.get("service_hint", "api")

    # Lexicographic id order trap: L-10 before L-2 when sorted as strings.
    logs = desk.get_logs(hint, since_ts=ticket.get("opened_ts", 0))
    logs_sorted = sorted(logs, key=lambda row: str(row["id"]))
    first = logs_sorted[0] if logs_sorted else {"id": "none", "msg": ""}

    # Wrong root cause: blames api from first line / hint.
    desk.restart_service("api")
    desk.advance_clock(1)  # too short for payments settle anyway
    _ = desk.health("api")

    # Submits before payments is healthy and without config fix.
    return desk.submit_resolution(
        ticket_id="INC-2048",
        root_cause="api_error:" + first.get("msg", "")[:40],
        actions=["restart_service"],
        evidence_log_ids=[first.get("id", "none")],
    )
