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
