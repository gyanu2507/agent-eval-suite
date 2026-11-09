# agent-eval-suite

Reference tasks and a technical report on catching no-op "wins" in agent evaluation.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23829461.svg)](https://doi.org/10.5281/zenodo.23829461)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Outcome tests can certify a method that does nothing. A run looks green because the method is a no-op, the control was never recomputed, or the fixture was edited. This repo defines a pass-bar for task authorship and a construction for ranking metrics that makes fake deltas fail: AUROC is invariant to a constant score offset, so mean-only "adaptation" can't move it. Plant a train-site label shortcut that flips at eval, freeze a control that the verifier recomputes from data, and gate on the delta.

Report: [When the delta is fake](docs/tech-report.md) ([HTML](docs/tech-report.html), [PDF](docs/tech-report.pdf)). Cite: [10.5281/zenodo.23829461](https://doi.org/10.5281/zenodo.23829461), [`CITATION.cff`](CITATION.cff).

## Quick start

```bash
git clone https://github.com/gyanu2507/agent-eval-suite && cd agent-eval-suite
./scripts/reproduce.sh
```

Needs Python 3.10+ and `pytest`. Everything else is stdlib, no GPU. It runs all 17 task tests, the `site-shift-meter` oracle, and the ablation below (under 10 seconds on a laptop).

## Result: site-shift-meter

![Eval AUROC by method on site-shift-meter](docs/figures/site-shift-ablation.svg)

| Method | Eval AUROC | Delta vs control | Gate (>= 0.08) |
| --- | --- | --- | --- |
| Frozen control (all features, raw) | 0.0108 | 0 | fail |
| Starter: copies control into model field | 0.0108 | 0 | fail |
| Mean-only: per-site centering, keeps shortcut `f4` | 0.0107 | -0.0001 | fail |
| Per-site z-score, keeps `f4` | 0.1068 | +0.0959 | pass |
| Drop `f4`, no adaptation | 0.9766 | +0.9658 | pass |
| Oracle: drop `f4` + per-site z-score | 0.9766 | +0.9658 | pass |

n_eval = 180 (eval site S2; train sites S0 and S1, 220 rows each). Numbers come from [`scripts/site_shift_ablation.py`](scripts/site_shift_ablation.py), which writes [`docs/figures/site-shift-ablation.json`](docs/figures/site-shift-ablation.json).

Reading it: pure mean-centering is a no-op under AUROC, as the invariance argument predicts. The control collapses to 0.011 because it latches onto `f4`, whose label association flips at eval. All of the oracle's gain comes from dropping `f4`. Per-site variance scaling adds nothing on this data.

Known gap: per-site z-scoring that keeps `f4` clears the 0.08 gate (+0.096). It doesn't drop the shortcut, so the gate threshold is loose for that method. A threshold closer to the shortcut-free delta, or a test that the model ignores `f4`, would close it.

## Reference tasks

| Task | What it shows about the pass-bar |
| --- | --- |
| [`site-shift-meter`](tasks/site-shift-meter) | Primary construction: fake delta vs shortcut flip + frozen control |
| [`clearing-hold-ledger`](tasks/clearing-hold-ledger) | Artifact-backed replay. Fixture edits and loose dedupe fail |
| [`slot-ring-buffer`](tasks/slot-ring-buffer) | A contract that single-consumer greens don't imply |
| [`incident-tool-desk`](tasks/incident-tool-desk) | Long-horizon tools and a clock, audit replay, text-judge false green |

Each task has a named starter failure, an oracle, tests, and a `CALIBRATION.md`. Trace rubric: [docs/trace-review.md](docs/trace-review.md). Judge vs verifier: [docs/judge-vs-verifier.md](docs/judge-vs-verifier.md).

## Pass-bar

| Criterion | Meaning |
| --- | --- |
| Nop failure | Starter fails for a failure mode you can state in one sentence |
| Oracle success | Reference solution clears every test |
| Artifact-backed score | Verifier recomputes the control or expected state. Report fields aren't trusted |
| Calibration record | Starter modes and blocked shortcuts are written down |

## Post-training on incident-tool-desk

LoRA SFT and DPO runs on Kaggle T4s, scored by the task's verifier rather than a text judge. Step-level SFT (train only on assistant tool calls, observations come back as user turns) passes the train ticket, a shifted ticket, and an unseen ticket. Full-transcript SFT invents observations and fails out of distribution. Details, kernels and the full table: [experiments/sft/RESULTS.md](experiments/sft/RESULTS.md). Adapters are under `experiments/sft/runs/` (about 28MB in total).

## Author

Gyanu Mayank, independent researcher. B.Tech CSE, NIT Patna.

Related upstream work on eval integrity: [Harbor #4003](https://github.com/harbor-framework/harbor/pull/4003), [Terminal-Bench #2480](https://github.com/harbor-framework/terminal-bench/pull/2480).
