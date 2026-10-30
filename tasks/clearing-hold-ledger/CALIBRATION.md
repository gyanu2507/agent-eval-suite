# Calibration: clearing-hold-ledger

## Why this task

Public analogue of multi-turn repair on an event-sourced ledger. Private work used VOID/idempotency/ordering bugs. This version uses HOLD/RELEASE with the same failure classes so the craft is visible without shipping private fixtures.

## Pass bar

- Oracle solution clears every unit test.
- Broken starter fails `test_starter_fails_oracle_expectations` and several property tests.
- Agent must edit `clearing/replay.py` (or equivalent) so a full replay of `data/events.json` matches the oracle state.

## Known starter failure modes

1. Lexicographic `seq` sort (`"10"` before `"2"`)
2. Prefix collision on `dedupe_key`
3. Partial HOLD when funds are short
4. RELEASE matched by wallet, not `hold_key`
5. Zero-balance wallets dropped from output

## Anti-shortcuts

- `data/events.json` is read-only for the agent.
- Tests recompute expectations from the event stream, not from a checked-in golden file the agent can overwrite.
- Hidden property tests cover ordering / dedupe / hold semantics beyond the fixture file.
