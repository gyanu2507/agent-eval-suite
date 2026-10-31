# When the delta is fake

**Pass-bar design for agent evaluation: nop failures, frozen controls, and scored traces**

Gyanu Mayank  
Independent Researcher  
B.Tech, Computer Science and Engineering, National Institute of Technology Patna  
Pune, India · gyanumayank0@gmail.com · https://github.com/gyanu2507  
31 October 2026 · Version 0.1.3  

**DOI (concept, preferred):** https://doi.org/10.5281/zenodo.23829461  
**Software:** https://github.com/gyanu2507/agent-eval-suite

---

## Abstract

Outcome-based agent benchmarks such as Terminal-Bench specify instruction, containerized environment, tests, and a reference solution [Merrill et al., 2026]. Harnesses such as Harbor and Inspect execute those units at scale. They do not, by themselves, guarantee that a task’s pass condition is resistant to cosmetic repairs, fixture edits, or self-reported metrics.

This technical note states a pass-bar for task authorship and releases three public tasks that implement it across event-sourced repair, concurrent systems contracts, and distribution-shift reporting. Each task includes a starter with named failure modes, an oracle that clears the suite, and a verifier that recomputes scores from artifacts. A distribution-shift case study shows that mean-only feature normalization need not change ranking AUROC for a linear scorer, and that a train-site label shortcut which flips at evaluation forces an honest control to fail unless the method drops the shortcut. A staged fail transcript is scored with an explicit positive/negative rubric to separate contract repair from fixture negotiation.

Scope is methodological: synthetic fixtures, three tasks, no frontier-model leaderboard. The contribution is a reusable design criteria set and calibrated examples for agent-evaluation work.

---

## 1. Positioning

| Layer | Provides | Leaves to the task author |
| --- | --- | --- |
| [Terminal-Bench](https://arxiv.org/abs/2601.11868) | Hard terminal tasks, reviewed oracles, outcome tests, Harbor as harness | Failure-mode design inside a new task |
| [Harbor](https://github.com/harbor-framework/harbor) | Parallel trials, trajectories, verifier rewards | Whether the reward is gameable |
| [Inspect](https://inspect.aisi.org.uk/) | Dataset, solver/agent, and scorer composition with sandboxes | The same question at sample level |

The Terminal-Bench unit (instruction, environment, final-state tests, oracle) is taken as the container. The criteria below address reward integrity: whether a green run implies the intended capability rather than a reporting trick.

The released suite is small and public by design. Domain analogues from confidential evaluation work are omitted. The failure classes transferred here are ordering and idempotency under event replay, dual-consumer buffer contracts, and metrics that remain green while the underlying method is a no-op.

---

## 2. Pass-bar criteria

A task is considered release-ready when:

1. **Nop failure.** The starter fails for a failure mode that can be stated in one sentence.
2. **Oracle success.** The reference solution clears every unit test.
3. **Artifact-backed score.** The verifier recomputes the control or expected state. Editing a report field or a checked-in golden file cannot invent a pass.
4. **Calibration record.** `CALIBRATION.md` lists starter failure modes and blocked shortcuts.

Default anti-shortcuts:

- Fixture data and frozen baselines are read-only to the agent.
- Property tests extend beyond the stock fixture.
- Occupancy, balances, and controls are derived quantities, not agent-supplied counters.

These criteria are stricter than “tests exist” and weaker than a multi-reviewer benchmark admission process. The three tasks demonstrate the criteria in code. They are not proposed as a replacement for Terminal-Bench.

---

## 3. Tasks

Results below use the checked-in fixtures and oracles (31 October 2026):

```bash
pytest -q -p no:randomly \
  tasks/clearing-hold-ledger/tests \
  tasks/slot-ring-buffer/tests \
  tasks/site-shift-meter/tests
```

Twelve tests pass.

### 3.1 `clearing-hold-ledger` (event-sourced repair)

**Objective.** Replay an immutable clearing stream into wallet balances and holds. Repair `clearing/replay.py`. Leave `data/events.json` unchanged.

**Starter failure modes.**

1. Lexicographic `seq` sort (`"10"` before `"2"`)
2. Prefix collision on `dedupe_key`
3. Partial HOLD when funds are insufficient
4. RELEASE matched by wallet rather than `hold_key`
5. Zero-balance wallets omitted from output

**Fixture.** 16 events. Oracle available balances: alpha 85, beta 45, delta/gamma/omega 0.

**Motivation.** Repair agents often chase the first failing test, then edit fixtures, then loosen deduplication. Tests recompute expected state from the stream. Prefix matching is strictly worse than the starter once `post-alpha-100-extra` collides with `post-alpha-100`.

### 3.2 `slot-ring-buffer` (systems contracts)

**Objective.** Fixed-capacity packet ring with two independent consumers. Under pressure from the slowest consumer, drop incoming writes. Never overwrite unread slots for the slow reader.

**Starter failure modes.** Incorrect dual-consumer drop policy, occupancy derived from a stale size counter, missed capacity edges at N=1 and N=2.

**Motivation.** Systems-lane evaluation without proprietary netlist or HDL fixtures. The contract concerns preservation of unread packets. Single-consumer tests can remain green while the dual-consumer contract fails.

### 3.3 `site-shift-meter` (distribution-shift reporting)

**Objective.** Train fault detection on sites S0/S1. Evaluate on S2. Exceed a frozen control and report calibration fields that match a recomputed baseline.

**Oracle (checked-in CSVs).**

| Quantity | Value |
| --- | --- |
| `control_auroc` | 0.0108 |
| `model_auroc` | 0.9766 |
| `delta_auroc` | 0.9658 |
| Gate | `delta_auroc >= 0.08` and control match within `1e-6` |
| `n_eval` | 180 |

**Starter.** Fits logistic regression on raw features, then copies that AUROC into both `control_auroc` and `model_auroc`. Reported delta is zero while the JSON schema appears complete.

Section 4 analyses this task in detail.

---

## 4. Case study: mean shift without ranking change

Training uses sites S0/S1. Evaluation uses S2 with shifted feature means. A common first intervention is per-site z-scoring followed by the same logistic model.

Under mean-only shift, ranking AUROC for a linear scorer is largely invariant to a constant offset on scores. Order is preserved. Per-site normalization can therefore look like domain adaptation while leaving delta against the control near zero.

The released fixture instead places a strong label shortcut in feature `f4` on training sites and flips the association on evaluation (anti-correlated noise). The frozen control in `baseline.py` relies on `f4` and collapses on S2 (control AUROC ≈ 0.01). A method that drops `f4` and z-scores `f1`-`f3` per site, estimating evaluation-site moments from unlabeled evaluation features, recovers a large delta (model AUROC ≈ 0.98 on the checked-in CSVs).

The acceptance gate (`delta_auroc >= 0.08`) is set so that (i) control copying and cosmetic mean-only z-scoring fail once the shortcut is present, and (ii) a correct method clears without requiring a near-perfect classifier.

**Pass conditions.**

1. Starter delta is zero.
2. Verifier recomputes the control from `baseline.py` and the CSVs.
3. `data/` and `baseline.py` are read-only.
4. `n_eval` equals the evaluation row count (180).

**Blocked shortcuts.** Inflating delta without beating the frozen control. Weakening the baseline in-place. Mean-only normalization that does not change ranking metrics. Editing goldens instead of changing the method.

The transferable pattern is: nop failure, oracle success, artifact-backed score, and an honest control even when that control is weak.

---

## 5. Trace review under an explicit rubric

Fixture: `docs/fixtures/ledger-fail-run.json`. The transcript is staged to match starter behavior. It is used to illustrate process scoring after unit tests have already failed.

**Failure sequence.**

1. **Lexicographic `seq`.** Turn 3 retains `sorted(..., key=str)`. Event 10 is applied before event 2. Balances diverge.
2. **Fixture edit.** Turn 6 modifies `data/events.json`, outside the repair surface.
3. **Weaker dedupe.** Turn 8 introduces prefix matching. `post-alpha-100-extra` is dropped once `post-alpha-100` is seen. The required rule is exact key equality.
4. **Incomplete hold semantics.** Partial HOLD and wallet-based RELEASE remain. Effort concentrates on the first red test.

**Rubric** (one behavior per line; magnitudes in `{1,2,3,5}`):

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

Positive mass: 13. Negatives encode the shortcuts observed in this transcript.

**Score sketch (`staged-fail-01`).** Numeric seq: not met. Exact dedupe: not met (regressed). All-or-nothing HOLD: not met. Release by `hold_key`: not met. Retain zero wallets: not met. Leave events unchanged: not met. Fixture edit: triggered (−5). String-order complacency: triggered (−2). Empty pytest retries: not triggered. Prefix dedupe introduction: triggered (−3).

The run fails both the task tests and the rubric. A correct repair clears the six positive criteria without modifying `data/`.

This rubric format separates outcome failure from process failure: whether the agent restored the contract or negotiated with fixtures.

---

## 6. Limitations

- Three tasks; synthetic or small fixtures only.
- No multi-model agent sweep is reported. The ledger transcript is constructed.
- Confidential evaluation corpora are excluded.
- Gate values are calibrated to these CSVs and are not claimed as universal thresholds.
- No safety or capability claims about frontier systems.

Natural extensions include timing tasks with an injected deterministic clock, and automated rubric application over trajectories. Neither is included in 0.1.3.

---

## 7. Reproduction

```bash
git clone https://github.com/gyanu2507/agent-eval-suite
cd agent-eval-suite
python3 -m pytest -q -p no:randomly \
  tasks/clearing-hold-ledger/tests \
  tasks/slot-ring-buffer/tests \
  tasks/site-shift-meter/tests
python3 tasks/site-shift-meter/solution/train_and_eval.py
```

See also `tasks/*/CALIBRATION.md`, `docs/case-study.md`, and `docs/trace-review.md`.

---

## References

Merrill, M. A., Shaw, A. G., Carlini, N., et al. (2026). *Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces*. arXiv:2601.11868. https://doi.org/10.48550/arXiv.2601.11868

Harbor Framework. https://github.com/harbor-framework/harbor

UK AI Security Institute. *Inspect*. https://inspect.aisi.org.uk/

Mayank, G. (2026). *When the delta is fake: Pass-bar design for agent evaluation* (v0.1.3). Zenodo. https://doi.org/10.5281/zenodo.23829461

---

## Citation

Use the concept DOI so citations resolve to the latest version.

```bibtex
@techreport{mayank2026deltafake,
  author       = {Mayank, Gyanu},
  title        = {When the delta is fake: Pass-bar design for agent evaluation},
  year         = {2026},
  month        = {10},
  institution  = {Independent},
  version      = {0.1.3},
  doi          = {10.5281/zenodo.23829461},
  url          = {https://doi.org/10.5281/zenodo.23829461},
  note         = {B.Tech CSE, National Institute of Technology Patna}
}
```
