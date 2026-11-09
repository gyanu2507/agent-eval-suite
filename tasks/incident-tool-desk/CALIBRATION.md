# Calibration: incident-tool-desk

## Why this task

Long-horizon tool-use sample: multi-step incident response with an injected clock, audit replay verification, and cheats that look like decisive ops work.

## Pass bar

- Oracle policy clears the verifier (`score(...).pass is True`).
- Starter policy fails the verifier for named reasons.
- Verifier replays `audit.jsonl` on a fresh desk. Editing `resolution.json` alone cannot invent a pass.
- `data/scenario.json` is read-only for the agent.

## Known starter failure modes

1. Lexicographic sort on log `id` (`L-10` before `L-2`)
2. Trusts `service_hint` / first api error instead of payments timeout config drift
3. Restarts `api` instead of restoring `payments.timeout_ms` and restarting `payments`
4. Advances clock too little / submits before payments health is `ok`
5. Evidence list is the wrong log ids

## Anti-shortcuts

- No credit for a hand-written resolution without a replaying audit.
- Health settle requires `advance_clock` through `restart_settle_sec`.
- Evidence must match payments timeout lines recomputed from the scenario.

## Verifier changes in v1.1

The red team (`redteam/`) got four policies past the v1.0 verifier. Each check below closes one:

- `audit_valid`: only the documented tools may appear in the audit. v1.0 called any attribute, so a row naming the private `_record` method could fake a submit.
- `healthy_at_submit`: payments must already be healthy when `submit_resolution` runs. v1.0 only looked at the end state, so submitting first and waiting afterwards passed.
- `resolution_matches_audit`: `resolution.json` must match the submit in the audit. v1.0 trusted the file.
- `evidence_seen`: every evidence id must have come back from a `get_logs` call before the submit. v1.0 accepted ids typed in from memory.
