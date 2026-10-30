# agent-eval-suite

Public samples of how I design agent evaluation tasks.

These are **original** tasks. Same craft as the private eval work (repair trajectories, systems contracts, shift + calibration gates). No private packages shipped here.

## Tasks

| Task | Lane | What the agent must get right |
| --- | --- | --- |
| [`clearing-hold-ledger`](tasks/clearing-hold-ledger) | Event-sourcing repair | Fix a broken event replay path: ordering, idempotency, hold/release semantics |
| [`slot-ring-buffer`](tasks/slot-ring-buffer) | Systems | Bring a packet ring buffer to a harder contract (capacity, discard, dual cursors) |
| [`site-shift-meter`](tasks/site-shift-meter) | Distribution shift | Train under site shift, report an honest control, clear a calibrated gate |

Each task has:

- `instruction.md` — agent-facing prompt
- `environment/` — starter code / data
- `tests/` — verifier
- `solution/` — reference fix
- `CALIBRATION.md` — why the pass bar sits where it does

## How to run locally

```bash
# Ledger
python3 tasks/clearing-hold-ledger/environment/app/clearing_main.py
pytest -q tasks/clearing-hold-ledger/tests

# Ring buffer
pytest -q tasks/slot-ring-buffer/tests

# Site-shift meter (CPU, tiny synthetic data)
python3 tasks/site-shift-meter/environment/app/train_and_eval.py
pytest -q tasks/site-shift-meter/tests
```

## What this is meant to prove

I build evaluations that:

1. Fail a stock / buggy starter for a known reason
2. Pass a careful oracle
3. Recompute scores from artifacts (not vibes)
4. Document calibration and anti-shortcuts

Private corpora stay private. This suite is the public, attribution-safe slice.
