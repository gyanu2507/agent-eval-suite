# agent-eval-suite

**Research note + reference tasks:** catching no-op “wins” in agent evaluation.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23829461.svg)](https://doi.org/10.5281/zenodo.23829461)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Idea.** Outcome tests are not enough. A run can look green while the method is a no-op, the control was never recomputed, or the fixture was edited. This repo states a **pass-bar** for task authorship and a concrete construction: ranking AUROC is invariant to pure mean shift / constant score offset, so mean-only “adaptation” can fake progress. Pair that with a train-site label shortcut that flips at eval and a **frozen, verifier-recomputed control**. Fake deltas fail. Real methods that drop the shortcut clear a large gap.

**Report:** [When the delta is fake](docs/tech-report.md) ([HTML](docs/tech-report.html) · [PDF](docs/tech-report.pdf))  
**Cite:** [10.5281/zenodo.23829461](https://doi.org/10.5281/zenodo.23829461) · [`CITATION.cff`](CITATION.cff)

**Author:** Gyanu Mayank · Independent Researcher · B.Tech CSE, NIT Patna · Pune, India

## Reference tasks (illustrations of the pass-bar)

| Task | What it proves about the bar |
| --- | --- |
| [`site-shift-meter`](tasks/site-shift-meter) | Primary construction: fake delta vs shortcut-flip + frozen control |
| [`clearing-hold-ledger`](tasks/clearing-hold-ledger) | Artifact-backed replay; fixture edits and loose dedupe fail |
| [`slot-ring-buffer`](tasks/slot-ring-buffer) | Contract not implied by weaker single-consumer greens |

Each task: named starter failure, oracle, tests, `CALIBRATION.md`. Trace rubric: [docs/trace-review.md](docs/trace-review.md).

## Pass-bar

Nop failure · oracle success · verifier recomputes score from artifacts · calibration record of blocked shortcuts.

## Reproduce

```bash
pytest -q -p no:randomly tasks/site-shift-meter/tests
pytest -q -p no:randomly tasks/clearing-hold-ledger/tests
pytest -q -p no:randomly tasks/slot-ring-buffer/tests
python3 tasks/site-shift-meter/solution/train_and_eval.py
```

## Ecosystem

Same integrity concerns in Terminal-Bench / Harbor work: [Harbor #4003](https://github.com/harbor-framework/harbor/pull/4003), [Terminal-Bench #2480](https://github.com/harbor-framework/terminal-bench/pull/2480).
