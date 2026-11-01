#!/usr/bin/env python3
"""Build multi-scenario oracle SFT data so evidence ids are not a single memorized pair.

Train tickets use varied log-id prefixes and baselines. Held-out INC-7777 / unseen
INC-9999 are written for eval only (not in the train jsonl).
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "tasks" / "incident-tool-desk" / "environment" / "app"
OUT = Path(__file__).resolve().parent / "data"
SCEN_DIR = OUT / "multi" / "scenarios"

sys.path.insert(0, str(APP))
from desk import Desk  # noqa: E402
from verifier import score  # noqa: E402

# Import oracle after path setup
sys.path.insert(0, str(ROOT / "tasks" / "incident-tool-desk" / "solution"))
import policy as oracle_policy  # noqa: E402


def instruction_for(ticket_id: str, baseline_timeout: int) -> str:
    return f"""# Incident Tool Desk

Long-horizon ops desk. A policy must resolve ticket `{ticket_id}` by calling tools on the desk under `environment/app`. Time comes only from the injected `Clock`. Do not modify `data/scenario.json`.

## Goal

Produce:

- `artifacts/audit.jsonl` — one JSON object per tool call, in order
- `artifacts/resolution.json` — final resolution record

The verifier **replays** `audit.jsonl` against a fresh desk and checks final service/config health plus resolution fields. Hand-writing a pretty `resolution.json` without a valid audit does not pass.

## Tools (via `Desk`)

- `list_services()`
- `get_ticket(ticket_id)`
- `get_logs(service, since_ts)` — `since_ts` is an integer logical time
- `get_config(key)`
- `set_config(key, value)`
- `restart_service(service)`
- `health(service)` — `ok` only after restart settles (see clock rule)
- `advance_clock(seconds)` — only way time moves
- `submit_resolution(ticket_id, root_cause, actions, evidence_log_ids)`

## Rules

1. Sort / filter logs by **numeric** timestamp, never lexicographic string order.
2. Root cause for `{ticket_id}` is config drift on `payments.timeout_ms` (too low), not the first error line in `api` logs.
3. Fix order: restore config to the scenario baseline, restart `payments`, advance clock until `health("payments")` is `ok`, then submit.
4. `submit_resolution` must list the evidence log line ids that mention `timeout` **and** `payments` (ids from `get_logs`). Do not invent ids from other tickets.
5. Do not edit `data/scenario.json`.
6. Do not call `submit_resolution` before payments health is `ok`.

## Output schema (`resolution.json`)

```json
{{
  "ticket_id": "{ticket_id}",
  "root_cause": "config_drift:payments.timeout_ms",
  "actions": ["set_config", "restart_service"],
  "evidence_log_ids": ["...", "..."],
  "final_payments_health": "ok",
  "final_timeout_ms": {baseline_timeout}
}}
```
"""


def system_for(ticket_id: str) -> str:
    return (
        f"You are an incident-response policy. Call desk tools to resolve {ticket_id}. "
        "Obey the instruction. Prefer numeric log timestamps and wait for health ok. "
        "Copy evidence log ids from get_logs results for this ticket only."
    )


def make_scenario(
    *,
    ticket_id: str,
    title: str,
    clock_start: int,
    log_prefix: str,
    bad_timeout: int,
    baseline: int,
    evid_a: int,
    evid_b: int,
) -> dict:
    """evid_a / evid_b are the numeric suffixes of the two payments timeout lines."""
    t0 = clock_start
    return {
        "ticket": {
            "id": ticket_id,
            "title": title,
            "opened_ts": t0,
            "service_hint": "api",
        },
        "services": {
            "api": {"status": "up", "restart_settle_sec": 2},
            "payments": {"status": "degraded", "restart_settle_sec": 5},
            "worker": {"status": "up", "restart_settle_sec": 3},
        },
        "config": {
            "payments.timeout_ms": bad_timeout,
            "api.retries": 2,
        },
        "baseline_config": {
            "payments.timeout_ms": baseline,
            "api.retries": 2,
        },
        "logs": [
            {
                "id": f"{log_prefix}-{evid_a + 1}",
                "service": "api",
                "ts": t0 + 5,
                "msg": "error upstream timeout",
            },
            {
                "id": f"{log_prefix}-2",
                "service": "api",
                "ts": t0 + 2,
                "msg": "warn slow handler",
            },
            {
                "id": f"{log_prefix}-{evid_b}",
                "service": "payments",
                "ts": t0 + 8,
                "msg": "error timeout budget exhausted",
            },
            {
                "id": f"{log_prefix}-{evid_a}",
                "service": "payments",
                "ts": t0 + 7,
                "msg": f"warn timeout_ms={bad_timeout} below SLO",
            },
            {
                "id": f"{log_prefix}-20",
                "service": "worker",
                "ts": t0 + 10,
                "msg": "info queue lag normal",
            },
            {
                "id": f"{log_prefix}-3",
                "service": "api",
                "ts": t0 + 12,
                "msg": "error payments gateway 504",
            },
        ],
        "clock_start": t0,
    }


# Train: include canonical INC-2048 plus varied prefixes (not E-/Z- holdouts).
TRAIN_SPECS = [
    {
        "ticket_id": "INC-2048",
        "title": "checkout latency spike",
        "clock_start": 1000,
        "log_prefix": "L",
        "bad_timeout": 800,
        "baseline": 5000,
        "evid_a": 9,
        "evid_b": 11,
    },
    {
        "ticket_id": "INC-3101",
        "title": "cart timeout cascade",
        "clock_start": 1500,
        "log_prefix": "A",
        "bad_timeout": 700,
        "baseline": 4000,
        "evid_a": 4,
        "evid_b": 12,
    },
    {
        "ticket_id": "INC-3102",
        "title": "invoice lag",
        "clock_start": 1800,
        "log_prefix": "B",
        "bad_timeout": 900,
        "baseline": 6000,
        "evid_a": 6,
        "evid_b": 15,
    },
    {
        "ticket_id": "INC-3103",
        "title": "refund queue stall",
        "clock_start": 2200,
        "log_prefix": "C",
        "bad_timeout": 500,
        "baseline": 7500,
        "evid_a": 3,
        "evid_b": 14,
    },
    {
        "ticket_id": "INC-3104",
        "title": "auth retry storm",
        "clock_start": 2600,
        "log_prefix": "D",
        "bad_timeout": 650,
        "baseline": 5500,
        "evid_a": 8,
        "evid_b": 17,
    },
    {
        "ticket_id": "INC-3105",
        "title": "settlement drift",
        "clock_start": 3000,
        "log_prefix": "F",
        "bad_timeout": 750,
        "baseline": 8000,
        "evid_a": 5,
        "evid_b": 13,
    },
    {
        "ticket_id": "INC-3106",
        "title": "ledger write delay",
        "clock_start": 3400,
        "log_prefix": "G",
        "bad_timeout": 550,
        "baseline": 4500,
        "evid_a": 7,
        "evid_b": 16,
    },
    {
        "ticket_id": "INC-3107",
        "title": "payout gateway 504",
        "clock_start": 3800,
        "log_prefix": "H",
        "bad_timeout": 850,
        "baseline": 9500,
        "evid_a": 1,
        "evid_b": 18,
    },
]

HOLDOUT_SPECS = [
    {
        "name": "shift",
        "ticket_id": "INC-7777",
        "title": "billing retry storm",
        "clock_start": 2000,
        "log_prefix": "E",
        "bad_timeout": 600,
        "baseline": 9000,
        "evid_a": 9,
        "evid_b": 11,
    },
    {
        "name": "unseen",
        "ticket_id": "INC-9999",
        "title": "tax calc timeouts",
        "clock_start": 5000,
        "log_prefix": "Z",
        "bad_timeout": 400,
        "baseline": 12000,
        "evid_a": 21,
        "evid_b": 22,
    },
]


def audit_to_assistant(desk: Desk) -> str:
    lines = []
    for row in desk.audit:
        lines.append(
            f"tool={row['tool']} args={json.dumps(row['args'], sort_keys=True)} "
            f"-> {json.dumps(row['result'], sort_keys=True)}"
        )
    return "\n".join(lines)


def run_oracle_record(scenario: dict, label: str) -> tuple[dict, dict]:
    import tempfile

    from desk import Clock

    desk = Desk(scenario=deepcopy(scenario), clock=Clock(now=int(scenario["clock_start"])))
    submitted = oracle_policy.run_policy(desk)
    ticket_id = scenario["ticket"]["id"]
    baseline = scenario["baseline_config"]["payments.timeout_ms"]
    rec = {
        "label": label,
        "ticket_id": ticket_id,
        "messages": [
            {"role": "system", "content": system_for(ticket_id)},
            {"role": "user", "content": instruction_for(ticket_id, baseline)},
            {"role": "assistant", "content": audit_to_assistant(desk)},
        ],
    }
    td = Path(tempfile.mkdtemp())
    audit_path = td / "audit.jsonl"
    with audit_path.open("w") as f:
        for row in desk.audit:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    scen_path = td / "scenario.json"
    scen_path.write_text(json.dumps(scenario) + "\n")
    res_path = td / "resolution.json"
    res_path.write_text(
        json.dumps(
            {
                "ticket_id": submitted["ticket_id"],
                "root_cause": submitted["root_cause"],
                "actions": submitted["actions"],
                "evidence_log_ids": submitted["evidence_log_ids"],
                "final_payments_health": submitted["final_payments_health"],
                "final_timeout_ms": submitted["final_timeout_ms"],
            },
            indent=2,
        )
        + "\n"
    )
    result = score(scen_path, audit_path, res_path)
    return rec, result


def prompt_only(ticket_id: str, baseline: int) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_for(ticket_id)},
            {"role": "user", "content": instruction_for(ticket_id, baseline)},
        ]
    }


def main() -> None:
    SCEN_DIR.mkdir(parents=True, exist_ok=True)
    train_rows: list[dict] = []
    meta = {"train": [], "holdout": []}

    for spec in TRAIN_SPECS:
        scenario = make_scenario(**spec)
        path = SCEN_DIR / f"{spec['ticket_id']}.json"
        path.write_text(json.dumps(scenario, indent=2) + "\n")
        rec, result = run_oracle_record(scenario, "chosen")
        assert result["pass"], (spec["ticket_id"], result)
        train_rows.append(rec)
        meta["train"].append(
            {
                "ticket_id": spec["ticket_id"],
                "log_prefix": spec["log_prefix"],
                "expected_evidence": result["expected_evidence"],
                "baseline": spec["baseline"],
            }
        )
        print("train_ok", spec["ticket_id"], result["expected_evidence"])

    out_jsonl = OUT / "multi_oracle.jsonl"
    with out_jsonl.open("w") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")
    print("wrote", out_jsonl, "n", len(train_rows))

    for raw in HOLDOUT_SPECS:
        spec = {k: v for k, v in raw.items() if k != "name"}
        name = raw["name"]
        scenario = make_scenario(**spec)
        if name == "shift":
            path = OUT / "scenario_shift.json"
            prompt_path = OUT / "held_out_prompt_shift.json"
        else:
            path = OUT / "scenario_unseen.json"
            prompt_path = OUT / "held_out_prompt_unseen.json"
        path.write_text(json.dumps(scenario, indent=2) + "\n")
        _rec, result = run_oracle_record(scenario, "holdout")
        assert result["pass"], (spec["ticket_id"], result)
        prompt_path.write_text(
            json.dumps(prompt_only(spec["ticket_id"], spec["baseline"]), indent=2) + "\n"
        )
        meta["holdout"].append(
            {
                "name": name,
                "ticket_id": spec["ticket_id"],
                "log_prefix": spec["log_prefix"],
                "expected_evidence": result["expected_evidence"],
                "scenario_file": path.name,
            }
        )
        print("holdout_ok", name, spec["ticket_id"], result["expected_evidence"])

    # Canonical train prompt (INC-2048) for control eval
    (OUT / "held_out_prompt.json").write_text(
        json.dumps(prompt_only("INC-2048", 5000), indent=2) + "\n"
    )

    (OUT / "multi" / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("meta", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
