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
- `tests/grade.py` pins the fixture's SHA-256.
- v1.1 compares the submission with the reference on 300 generated streams (`grade.generate_stream`). v1.0 only had the fixture and three small public cases, so a lookup table of their answers passed, and so did two real bugs: a RELEASE that frees the same hold twice, and a HOLD that overwrites an open hold with the same key. See `redteam/`.

## Spec gaps the generator steps around

The instruction doesn't say whether a failed HOLD burns its `dedupe_key`, or whether a RELEASE adds its own `wallet` field to the output. The reference says no to both. The generator only duplicates POST, DRAW and MOVE events, which always apply, so these two cases don't decide a grade.
