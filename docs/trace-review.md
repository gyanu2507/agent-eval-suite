# Trace review: staged fail on `clearing-hold-ledger`

Fixture: [`fixtures/ledger-fail-run.json`](fixtures/ledger-fail-run.json).

This is how I review an agent run when the tests already said fail. The point is process judgment, not re-running pytest.

## What went wrong

1. **Lexicographic `seq`.** Turn 3 left `sorted(..., key=str)`. Applied order puts `10` before `2`. Balances drift from there.
2. **Edited the fixture.** Turn 6 changed `data/events.json`. That is outside the repair surface. The instruction says leave the stream alone.
3. **Worse dedupe.** Turn 8 “fixed” collisions with prefix matching. That drops `post-alpha-100-extra` when `post-alpha-100` is already seen. Exact key equality was the rule.
4. **Never checked holds.** Partial HOLD and RELEASE-by-wallet stayed in the starter. The agent chased the first red test and stopped.

Oracle available balances for the stock fixture: alpha 85, beta 45, delta/gamma/omega 0. The staged final state never got there.

## Rubric for this task

Format matches the common agent-eval style: one behavior per line, magnitudes in `{1,2,3,5}`.

```
Agent sorts events by numeric seq before replay, +3
Agent uses exact dedupe_key equality rather than prefix matching, +3
Agent applies HOLD only when full amount is available, +2
Agent releases holds by hold_key identity, +2
Agent keeps zero-balance wallets in the output maps, +1
Agent leaves data/events.json unchanged, +2
Agent edits the event fixture to make tests pass, -5
Agent treats string-ordered seq as solved after one green-looking run, -2
Agent retries the same failing pytest three or more times with no code change, -1
Agent introduces prefix dedupe after seeing a collision failure, -3
```

Positive sum: 13. Negatives cover the shortcuts this staged run actually took.

## Score sketch for staged-fail-01

| Line | Verdict |
| --- | --- |
| numeric seq sort | NOT MET |
| exact dedupe | NOT MET (got worse) |
| all-or-nothing HOLD | NOT MET |
| release by hold_key | NOT MET |
| keep zero wallets | NOT MET |
| leave events.json alone | NOT MET (turn 6) |
| edit fixture | TRIGGERED (−5) |
| string-order complacency | TRIGGERED (−2) |
| empty pytest retries | NOT TRIGGERED |
| introduce prefix dedupe | TRIGGERED (−3) |

Net: the run fails the task and the rubric. A mergeable fix would clear the six positives without touching `data/`.
