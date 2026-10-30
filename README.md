# agent-eval-suite

Public samples of how I design agent evaluation tasks. Original code and synthetic data only.

Read the [case study](docs/case-study.md) for pass-bar reasoning, and the [trace review](docs/trace-review.md) for how I score a failing agent run.

## Tasks

| Task | Lane | What the agent must get right |
| --- | --- | --- |
| [`clearing-hold-ledger`](tasks/clearing-hold-ledger) | Event-sourcing repair | Fix a broken event replay path: ordering, idempotency, hold/release semantics |
| [`slot-ring-buffer`](tasks/slot-ring-buffer) | Systems | Bring a packet ring buffer to a harder contract (capacity, discard, dual cursors) |
| [`site-shift-meter`](tasks/site-shift-meter) | Distribution shift | Train under site shift, report an honest control, clear a calibrated gate |

Each task has `instruction.md`, a buggy starter, tests, an oracle, and `CALIBRATION.md`.

## How to run locally

```bash
pytest -q -p no:randomly tasks/clearing-hold-ledger/tests
pytest -q -p no:randomly tasks/slot-ring-buffer/tests
pytest -q -p no:randomly tasks/site-shift-meter/tests
```

## Design bar

A task ships when the starter fails for a reason I can name, the oracle passes, the verifier recomputes the score from artifacts, and the calibration note says which shortcuts are blocked.
