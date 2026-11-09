# Construction instance: mean shift without ranking change

Primary claim of the [tech report](tech-report.md) ([DOI](https://doi.org/10.5281/zenodo.23829461)): fake deltas under ranking AUROC, caught by shortcut flip + frozen control.

Task: `site-shift-meter`. Train on sites S0/S1, evaluate on S2.

## Starter

Trains logistic regression on raw features and submits its scores, which is the control itself. Delta is zero. In v1.0 it also copied that AUROC into both `control_auroc` and `model_auroc`, so the report schema looked complete.

## Why mean-only adaptation is insufficient

Per-site normalization followed by the same logistic is a common first intervention. Under mean-only synthetic shift, ranking AUROC for a linear scorer is largely invariant to a constant score offset. Per-site mean-centering scores 0.0107 here, the same as the control.

Z-scoring also rescales each feature, and that does move the ranking: 0.0108 to 0.1068 while still leaning on `f4`. That cleared the v1.0 gate of 0.08. v1.1 raises the gate to 0.5 (see the task's `CALIBRATION.md`).

## What breaks the control

A strong label shortcut in `f4` on training sites is flipped on evaluation (anti-correlated noise). The frozen control in `baseline.py` relies on `f4` and collapses on S2. Dropping `f4` and z-scoring `f1`-`f3` per site (evaluation-site moments from unlabeled evaluation features) recovers a large delta.

On the checked-in CSVs: control AUROC ≈ 0.01, reference method ≈ 0.98. Gate in v1.1: the verifier scores the submission on hidden S2 labels and needs AUROC >= 0.9 and delta >= 0.5. v1.0 used delta >= 0.08 and read the model number from the report. That gate was loose for a second reason: with the control at 0.011, a constant score already gives +0.489.

## Pass conditions

1. Starter fails (delta 0).
2. Verifier recomputes the control from `baseline.py` and the CSVs.
3. `data/` and `baseline.py` are read-only.
4. `n_eval` equals the evaluation row count (180).

## Blocked shortcuts

Inflating delta without beating the frozen control (v1.1 recomputes model AUROC from scores). Weakening the baseline in-place. Mean-only normalization that does not change ranking metrics. Editing goldens instead of changing the method. Scoring with the eval labels (v1.1 keeps them out of `eval.csv`). Random or constant scores (v1.1 has an absolute AUROC floor).

---

# Post-training on a verifier desk

Second construction: `incident-tool-desk` plus a small LoRA loop on Qwen2.5-0.5B-Instruct (Kaggle T4). Full table: [`experiments/sft/RESULTS.md`](../experiments/sft/RESULTS.md). Judge-vs-verifier fixtures: [`docs/judge-vs-verifier.md`](judge-vs-verifier.md).

The desk only passes when an audit of tool calls replays to a healthy payments config. Pretty `resolution.json` without a valid audit fails.

## Open-loop failure

Full-transcript SFT (assistant emits `tool=… args=… -> {observation}`) can look green on the train ticket (`INC-2048`, evidence `L-9`/`L-11`). On a shifted ticket (`INC-7777`, evidence `E-*`) open-loop decode invents observations, then copies those fake log ids into `submit_resolution`. Multi-ticket cloning alone did not fix that. Train pass rate is not OOD competence.

## What changed the gate

1. **Tool-loop decode.** Generate one call, run it on the live desk, append the real observation, continue. Shift went green even for single-ticket SFT. The model stops needing to invent `->` payloads.
2. **Step-level SFT.** Train only on assistant tool-call turns. Real results sit in user turns as `OBS …`. Decode with the same chat tool-loop.

Step-SFT + chat tool-loop (Kaggle, ~728 s train+eval, loss ≈ 0.010):

| Eval | Verifier | Evidence | Baseline timeout |
| --- | --- | --- | --- |
| train `INC-2048` | pass | L-9, L-11 | 5000 |
| shift `INC-7777` | pass | E-9, E-11 | 9000 |
| unseen `INC-9999` | pass | Z-21, Z-22 | 12000 |

DPO on top of green transcript SFT stayed train-ticket green when continuing the SFT LoRA with a merged-SFT reference. Fresh LoRA after merge wiped tool format. Epoch/rank cuts (5 ep, r=8) still passed the train ticket under open-loop.

## Blocked shortcuts (post-training)

Reporting train loss or preference NLL as the headline. Open-loop gen that never talks to the desk. Memorizing one ticket's evidence ids. Claiming scenario transfer without a held-out ticket and a programmatic replay verifier.
