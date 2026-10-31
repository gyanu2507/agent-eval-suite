# agent-eval-suite

Pass-bar design for agent evaluation: tasks with named starter failures, oracles, and artifact-backed scoring.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23829461.svg)](https://doi.org/10.5281/zenodo.23829461)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Technical report:** [When the delta is fake](docs/tech-report.md) ([HTML](docs/tech-report.html) · [PDF](docs/tech-report.pdf))  
**Cite:** [10.5281/zenodo.23829461](https://doi.org/10.5281/zenodo.23829461) · [`CITATION.cff`](CITATION.cff)

Coverage spans event-sourced repair, systems contracts, and distribution-shift reporting. Fixtures are synthetic. Short notes: [case study](docs/case-study.md), [trace review](docs/trace-review.md).

**Author:** Gyanu Mayank · Independent Researcher · B.Tech CSE, NIT Patna · Pune, India

## Tasks

| Task | Domain | Required behavior |
| --- | --- | --- |
| [`clearing-hold-ledger`](tasks/clearing-hold-ledger) | Event-sourced repair | Correct ordering, idempotency, and hold/release semantics |
| [`slot-ring-buffer`](tasks/slot-ring-buffer) | Systems | Dual-consumer ring with drop-incoming under slow-reader pressure |
| [`site-shift-meter`](tasks/site-shift-meter) | Distribution shift | Beat a frozen control; report recomputable calibration |

Each task provides `instruction.md`, a failing starter, tests, an oracle, and `CALIBRATION.md`.

## Pass-bar

A task is released when the starter fails for a named reason, the oracle passes, the verifier recomputes score from artifacts, and calibration documents blocked shortcuts.

## Reproduce

```bash
pytest -q -p no:randomly tasks/clearing-hold-ledger/tests
pytest -q -p no:randomly tasks/slot-ring-buffer/tests
pytest -q -p no:randomly tasks/site-shift-meter/tests
```

## Related work in the Terminal-Bench ecosystem

- [Harbor #4003](https://github.com/harbor-framework/harbor/pull/4003): target trajectory when a user agent is present
- [Terminal-Bench #2480](https://github.com/harbor-framework/terminal-bench/pull/2480): `lease-fencing-reaper`
