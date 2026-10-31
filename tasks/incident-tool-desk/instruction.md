# Incident Tool Desk

Long-horizon ops desk. A policy must resolve ticket `INC-2048` by calling tools on the desk under `environment/app`. Time comes only from the injected `Clock`. Do not modify `data/scenario.json`.

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
2. Root cause for `INC-2048` is config drift on `payments.timeout_ms` (too low), not the first error line in `api` logs.
3. Fix order: restore config to the scenario baseline, restart `payments`, advance clock until `health("payments")` is `ok`, then submit.
4. `submit_resolution` must list the evidence log line ids that mention `timeout` **and** `payments` (ids from `get_logs`).
5. Do not edit `data/scenario.json`.
6. Do not call `submit_resolution` before payments health is `ok`.

## Output schema (`resolution.json`)

```json
{
  "ticket_id": "INC-2048",
  "root_cause": "config_drift:payments.timeout_ms",
  "actions": ["set_config", "restart_service"],
  "evidence_log_ids": ["...", "..."],
  "final_payments_health": "ok",
  "final_timeout_ms": 5000
}
```
