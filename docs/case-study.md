# Construction instance: mean shift without ranking change

Primary claim of the [tech report](tech-report.md) ([DOI](https://doi.org/10.5281/zenodo.23829461)): fake deltas under ranking AUROC, caught by shortcut flip + frozen control.

Task: `site-shift-meter`. Train on sites S0/S1, evaluate on S2.

## Starter

Trains logistic regression on raw features, then copies that AUROC into both `control_auroc` and `model_auroc`. Reported delta is zero. The JSON schema appears complete.

## Why mean-only adaptation is insufficient

Per-site z-scoring followed by the same logistic is a common first intervention. Under mean-only synthetic shift, ranking AUROC for a linear scorer is largely invariant to a constant score offset. Order is preserved, so the adapted model clears almost nothing against the control.

## What breaks the control

A strong label shortcut in `f4` on training sites is flipped on evaluation (anti-correlated noise). The frozen control in `baseline.py` relies on `f4` and collapses on S2. Dropping `f4` and z-scoring `f1`-`f3` per site (evaluation-site moments from unlabeled evaluation features) recovers a large delta.

On the checked-in CSVs: control AUROC ≈ 0.01, reference method ≈ 0.98. Gate: `delta_auroc >= 0.08`, with control recomputed within `1e-6`.

## Pass conditions

1. Starter fails (delta 0).
2. Verifier recomputes the control from `baseline.py` and the CSVs.
3. `data/` and `baseline.py` are read-only.
4. `n_eval` equals the evaluation row count (180).

## Blocked shortcuts

Inflating delta without beating the frozen control. Weakening the baseline in-place. Mean-only normalization that does not change ranking metrics. Editing goldens instead of changing the method.

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

Reporting train loss or preference NLL as the headline. Open-loop gen that never talks to the desk. Memorizing one ticket’s evidence ids. Claiming scenario transfer without a held-out ticket and a programmatic replay verifier.
