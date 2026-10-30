# Clearing Hold Ledger

The service under `environment/app` rebuilds wallet balances by replaying immutable clearing events from `data/events.json` into `clearing_state.json`. The starter is wrong under ordering, idempotency, and hold/release edge cases. Repair the implementation. Do not change files under `data/`.

## Event types

- `POST` — credit `wallet` by `amount`
- `DRAW` — debit `wallet` by `amount`
- `MOVE` — transfer `amount` from `src` to `dst`
- `HOLD` — freeze `amount` on `wallet` (reduces available balance, tracked separately)
- `RELEASE` — release a prior `HOLD` identified by `hold_key`

## Rules

1. Replay in deterministic numeric `seq` order (not lexicographic string order).
2. Process each event once per full `dedupe_key`. Keys that share a prefix are not duplicates.
3. `RELEASE` only frees a previously applied `HOLD` with matching `hold_key`.
4. Releasing an unknown, already-released, or non-HOLD key is a no-op.
5. A `HOLD` that exceeds available funds is a no-op (do not partially hold).
6. `MOVE` updates both wallets.
7. Output must include every wallet id that appears anywhere in the stream, even at `0.0`.
8. Do not modify `data/events.json`.

## Output

Write `clearing_state.json` with exactly:

- `available`: wallet id → float available balance
- `held`: wallet id → float currently held
- `applied`: list of applied events in replay order (`seq`, `type`, `dedupe_key`)
- `released`: sorted list of hold keys successfully released
