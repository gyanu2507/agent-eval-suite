# When the delta is fake

**A construction for catching no-op “wins” in agent evaluation**

Gyanu Mayank  
Independent Researcher  
B.Tech, Computer Science and Engineering, National Institute of Technology Patna  
Pune, India · gyanumayank0@gmail.com · https://github.com/gyanu2507  
31 October 2026 · Version 0.1.4  

**DOI (concept, preferred):** https://doi.org/10.5281/zenodo.23829461  
**Software:** https://github.com/gyanu2507/agent-eval-suite

---

## Abstract

Agent benchmarks increasingly score autonomous repair and tool-use by final state: instruction, environment, tests, oracle [Merrill et al., 2026]. That unit is necessary. It is not sufficient. A run can satisfy a schema, print a large “delta,” or turn tests green while the underlying method is a no-op, the fixture has been edited, or the reported control was never recomputed.

This note proposes a small integrity layer for agent-task authorship, call it a **pass-bar**, and a concrete construction that makes fake improvement hard to hide. The core observation is metric-theoretic: for ranking AUROC under a linear scorer, a pure mean shift of features (or a constant offset of scores) need not change order. Domain-adaptation theater that only recenters features can therefore look serious and still leave delta at zero. Pairing that invariance with a **train-only label shortcut that flips at evaluation**, plus a **frozen, verifier-recomputed control**, forces any passing method to drop the shortcut rather than rewrite the report.

Three public tasks instantiate the pass-bar (nop starter, oracle, artifact-backed score, calibration record). One staged transcript is scored with an explicit process rubric to separate contract repair from fixture negotiation. The claim is methodological: a reusable way to author agent tasks so “green” implies capability, not cosmetics. Synthetic fixtures only. No frontier leaderboard.

---

## 1. The problem

Outcome harnesses (Terminal-Bench via Harbor, Inspect-style task/solver/scorer stacks) answer: did the final state match the tests? They do not answer: was the pass earned by solving the intended difficulty, or by negotiating the measurement?

Observed failure modes in agent evaluation practice:

1. **Schema pass, method nop.** Artifacts match a JSON shape while control and model fields are identical.
2. **Cosmetic adaptation.** Normalization or “site shift” code that cannot change the scored ranking metric.
3. **Control capture.** The agent weakens or replaces the baseline instead of beating a frozen one.
4. **Fixture negotiation.** Editing goldens or event streams so tests agree with a broken implementation.
5. **First-red fixation.** Repair stops at the first failing assertion while deeper contract bugs remain.

Benchmarks that only publish pass rate under these modes overstate progress. The contribution below is a design response, not a new 100-task corpus.

---

## 2. Proposed pass-bar

Release a task only when:

| Criterion | Meaning |
| --- | --- |
| Nop failure | Starter fails for a failure mode stateable in one sentence |
| Oracle success | Reference solution clears every test |
| Artifact-backed score | Verifier recomputes control or expected state; report fields are not authoritative |
| Calibration record | Documented starter modes and blocked shortcuts |

Defaults: fixtures and frozen baselines read-only to the agent; property tests beyond the stock fixture; derived quantities (balances, occupancy, controls) rather than agent-written counters.

Relative to Terminal-Bench’s unit, this is an authorship integrity layer. It does not replace human review or scale. It makes a class of false greens expensive.

---

## 3. The construction: fake deltas under ranking metrics

### 3.1 Ranking invariance

Let scores be \(s(x)\) for a linear model. Adding a constant \(c\) to every score leaves pairwise order unchanged, hence leaves AUROC unchanged. A covariate mean shift that is absorbed into the bias term behaves the same way for ranking. Therefore **mean-only “adaptation” is not evidence of distribution-shift competence** when the gate is AUROC delta against a control.

### 3.2 Shortcut flip + frozen control

To make the gate bite:

1. Plant a strong label association in a feature on training sites.
2. Flip that association on the evaluation site.
3. Freeze a control that fits on all raw features (including the shortcut) and **recompute it in the verifier**.
4. Require `model_auroc - control_auroc` above a calibrated threshold, with control match within a tight tolerance.

A method that only z-scores means fails. A method that drops the shortcut and adapts the remaining features can clear a large delta. Editing the report cannot lower the control.

### 3.3 Checked-in instance (`site-shift-meter`)

| Quantity | Value |
| --- | --- |
| `control_auroc` | 0.0108 |
| `model_auroc` | 0.9766 |
| `delta_auroc` | 0.9658 |
| Gate | `delta_auroc >= 0.08`, control match `1e-6` |
| `n_eval` | 180 |

Starter copies control into the model field (delta 0). Oracle drops `f4`, z-scores `f1`-`f3` per site using unlabeled eval moments.

This is the crack the title names: **a green-looking delta can be fake; the construction makes the fake fail.**

---

## 4. Instantiations

The construction is metric-specific. The pass-bar is not. Two further tasks show the same integrity pattern outside AUROC.

### 4.1 Event replay (`clearing-hold-ledger`)

Immutable clearing stream. Starter fails on lexicographic `seq`, prefix dedupe, partial HOLD, wallet-based RELEASE, dropped zero wallets. Tests recompute from the stream. Fixture is read-only. Agents that edit events or loosen dedupe are caught.

### 4.2 Dual-consumer ring (`slot-ring-buffer`)

Unread packets for the slow consumer must survive under drop-incoming pressure. Occupancy is derived from cursors. Single-consumer greens do not imply the contract.

### 4.3 Process rubric after outcome fail

Staged transcript `docs/fixtures/ledger-fail-run.json` scored with signed magnitudes (positive contract behaviors, negative fixture negotiation). Separates “tests failed” from “how the agent failed.” Positive mass 13. The staged run triggers fixture edit (−5), string-order complacency (−2), and prefix-dedupe regression (−3).

Full rubrics and calibrations: `docs/trace-review.md`, `tasks/*/CALIBRATION.md`.

---

## 5. Relation to existing stacks

| Stack | Role here |
| --- | --- |
| Terminal-Bench [Merrill et al., 2026] | Outcome unit (instruction, env, tests, oracle) |
| Harbor | Execution harness |
| Inspect | Dataset / agent / scorer composition |

The note adds **authorship constraints** those stacks leave open. Adjacent open contributions in that ecosystem (harness trajectory context; a lease-fencing task draft) are listed in the repository README. They are not bibliographic claims of this DOI.

---

## 6. Limitations

- One primary construction (ranking AUROC + shortcut flip). Other metrics need their own invariance arguments.
- Three illustrative tasks, synthetic data, no multi-model sweep.
- Staged transcript, not a live frontier-agent log.
- Gate threshold calibrated to these CSVs.
- No capability or safety claims about models.

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

---

## References

Merrill, M. A., Shaw, A. G., Carlini, N., et al. (2026). *Terminal-Bench: Benchmarking Agents on Hard, Realistic Tasks in Command Line Interfaces*. arXiv:2601.11868. https://doi.org/10.48550/arXiv.2601.11868

Harbor Framework. https://github.com/harbor-framework/harbor

UK AI Security Institute. *Inspect*. https://inspect.aisi.org.uk/

Mayank, G. (2026). *When the delta is fake* (v0.1.4). Zenodo. https://doi.org/10.5281/zenodo.23829461

---

## Citation

```bibtex
@techreport{mayank2026deltafake,
  author       = {Mayank, Gyanu},
  title        = {When the delta is fake: A construction for catching no-op wins in agent evaluation},
  year         = {2026},
  month        = {10},
  institution  = {Independent},
  version      = {0.1.4},
  doi          = {10.5281/zenodo.23829461},
  url          = {https://doi.org/10.5281/zenodo.23829461},
  note         = {B.Tech CSE, National Institute of Technology Patna}
}
```
