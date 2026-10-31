# agent-eval-suite

Public samples of how I design agent evaluation tasks. Original code and synthetic data only.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23829461.svg)](https://doi.org/10.5281/zenodo.23829461)

**Tech report:** [When the delta is fake](docs/tech-report.md) ([HTML](docs/tech-report.html), [PDF](docs/tech-report.pdf))  
**Cite:** concept DOI [10.5281/zenodo.23829461](https://doi.org/10.5281/zenodo.23829461) · [`CITATION.cff`](CITATION.cff)

Short forms: [case study](docs/case-study.md) (site-shift pass bar), [trace review](docs/trace-review.md) (scored ledger fail).

Author: Gyanu Mayank · B.Tech CSE, NIT Patna · Pune, India

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

## Related open-source activity

Harness and task drafts in the same craft (not part of the Zenodo bibliographic references):

- [Harbor #4003](https://github.com/harbor-framework/harbor/pull/4003): write target trajectory when a user agent is present
- [Terminal-Bench #2480](https://github.com/harbor-framework/terminal-bench/pull/2480): `lease-fencing-reaper`
