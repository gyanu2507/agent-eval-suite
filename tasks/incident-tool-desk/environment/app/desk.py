"""Ops desk: tools + injected clock. Scenario file is read-only for agents."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Clock:
    now: int

    def advance(self, seconds: int) -> int:
        if seconds < 0:
            raise ValueError("seconds must be >= 0")
        self.now += int(seconds)
        return self.now


@dataclass
class Desk:
    scenario: dict[str, Any]
    clock: Clock
    audit: list[dict[str, Any]] = field(default_factory=list)
    _restart_ready_at: dict[str, int] = field(default_factory=dict)
    _submitted: dict[str, Any] | None = None

    @classmethod
    def from_scenario_path(cls, path: Path) -> Desk:
        data = json.loads(path.read_text())
        return cls(scenario=deepcopy(data), clock=Clock(now=int(data["clock_start"])))

    def _record(self, tool: str, args: dict[str, Any], result: Any) -> Any:
        self.audit.append({"tool": tool, "args": args, "result": result, "ts": self.clock.now})
        return result

    def list_services(self) -> dict[str, str]:
        out = {name: meta["status"] for name, meta in self.scenario["services"].items()}
        return self._record("list_services", {}, out)

    def get_ticket(self, ticket_id: str) -> dict[str, Any]:
        ticket = self.scenario["ticket"]
        if ticket["id"] != ticket_id:
            res: dict[str, Any] = {"error": "unknown_ticket"}
        else:
            res = dict(ticket)
        return self._record("get_ticket", {"ticket_id": ticket_id}, res)

    def get_logs(self, service: str, since_ts: int) -> list[dict[str, Any]]:
        rows = [
            dict(row)
            for row in self.scenario["logs"]
            if row["service"] == service and int(row["ts"]) >= int(since_ts)
        ]
        # Caller must sort numerically; desk returns unsorted on purpose for ts order traps
        # when filtered subset still has L-10 before L-2 if agent sorts by id string.
        return self._record(
            "get_logs", {"service": service, "since_ts": int(since_ts)}, rows
        )

    def get_config(self, key: str) -> Any:
        cfg = self.scenario["config"]
        res = cfg.get(key, None)
        return self._record("get_config", {"key": key}, res)

    def set_config(self, key: str, value: Any) -> dict[str, Any]:
        self.scenario["config"][key] = value
        res = {"ok": True, "key": key, "value": value}
        return self._record("set_config", {"key": key, "value": value}, res)

    def restart_service(self, service: str) -> dict[str, Any]:
        services = self.scenario["services"]
        if service not in services:
            res = {"error": "unknown_service"}
            return self._record("restart_service", {"service": service}, res)
        settle = int(services[service]["restart_settle_sec"])
        self._restart_ready_at[service] = self.clock.now + settle
        services[service]["status"] = "restarting"
        res = {"ok": True, "ready_at": self._restart_ready_at[service]}
        return self._record("restart_service", {"service": service}, res)

    def health(self, service: str) -> dict[str, Any]:
        services = self.scenario["services"]
        if service not in services:
            res = {"status": "unknown"}
            return self._record("health", {"service": service}, res)
        ready_at = self._restart_ready_at.get(service)
        if ready_at is None:
            # never restarted this run: report current label
            status = services[service]["status"]
            # degraded payments is not ok
            mapped = "ok" if status == "up" else status
            res = {"status": mapped}
            return self._record("health", {"service": service}, res)
        if self.clock.now >= ready_at:
            services[service]["status"] = "up"
            res = {"status": "ok"}
        else:
            services[service]["status"] = "restarting"
            res = {"status": "restarting", "ready_at": ready_at}
        return self._record("health", {"service": service}, res)

    def advance_clock(self, seconds: int) -> dict[str, Any]:
        now = self.clock.advance(seconds)
        res = {"now": now}
        return self._record("advance_clock", {"seconds": int(seconds)}, res)

    def _payments_health_status(self) -> str:
        ready_at = self._restart_ready_at.get("payments")
        status = self.scenario["services"]["payments"]["status"]
        if ready_at is not None and self.clock.now >= ready_at:
            self.scenario["services"]["payments"]["status"] = "up"
            return "ok"
        if ready_at is not None and self.clock.now < ready_at:
            return "restarting"
        return "ok" if status == "up" else status

    def submit_resolution(
        self,
        ticket_id: str,
        root_cause: str,
        actions: list[str],
        evidence_log_ids: list[str],
    ) -> dict[str, Any]:
        timeout = self.scenario["config"].get("payments.timeout_ms")
        res = {
            "accepted": True,
            "ticket_id": ticket_id,
            "root_cause": root_cause,
            "actions": list(actions),
            "evidence_log_ids": list(evidence_log_ids),
            "final_payments_health": self._payments_health_status(),
            "final_timeout_ms": timeout,
            "submitted_at": self.clock.now,
        }
        self._submitted = res
        return self._record(
            "submit_resolution",
            {
                "ticket_id": ticket_id,
                "root_cause": root_cause,
                "actions": list(actions),
                "evidence_log_ids": list(evidence_log_ids),
            },
            res,
        )


def load_scenario(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())
