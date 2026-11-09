# Changelog

## 0.2.0 (2026-11-09)

Red team and verifier fixes. Every task moves from version 1.0 to 1.1 in its `task.toml`. The tech report (DOI 10.5281/zenodo.23829461) describes the 1.0 tasks and isn't changed.

### Added

- `tasks/<task>/tests/grade.py` for all four tasks. Each one grades any submission, under the 1.0 or the 1.1 rules.
- `redteam/`: 23 attacks, 4 alternate correct solutions, and `run.py`, which grades all of them under both rule sets. 14 attacks passed 1.0. 1 passes 1.1.
- `scripts/reproduce.sh` runs the red team after the tests.

### Changed

site-shift-meter
- Eval labels moved out of `eval.csv` into `tests/eval_labels.csv`, so the agent no longer sees them. In 1.0 a script could score with the labels.
- The submission writes `artifacts/scores.csv`, one score per eval row, and `report.json` shrinks to `n_eval`, `method` and `notes`. The verifier computes both AUROCs itself. 1.0 read `model_auroc` from the report, so a typed-in 0.99 passed.
- Gate is now AUROC >= 0.9 and delta >= 0.5. 1.0 asked for delta >= 0.08 only. Because the control sits at 0.011, a constant score already gave +0.489 and random scores cleared 0.08 every time. Z-scoring that keeps the `f4` shortcut scored +0.096 and cleared it too.
- `baseline.py` gains `control_scores()` and `load_labels()`, and `load_csv()` accepts a file without labels. The control's numbers are unchanged.
- The verifier checks that `data/` and `baseline.py` are unchanged.

clearing-hold-ledger
- The grader compares the submission with the reference on 300 generated event streams and pins the fixture's SHA-256. 1.0 had only public cases, so a lookup table passed, along with a double-release bug and a hold-key overwrite bug.
- CALIBRATION.md claimed hidden property tests that didn't exist in 1.0. They exist now.

slot-ring-buffer
- The grader replays 500 random write/read sequences against the reference, including seq wrap. This catches a ring that counts empty reads and one that rejects a wrapped seq.

incident-tool-desk (`verifier.py`)
- Only the documented tools may appear in the audit.
- payments must be healthy when `submit_resolution` runs. 1.0 only checked the end state.
- `resolution.json` must match the submit recorded in the audit.
- Every evidence id must come from a `get_logs` result earlier in the audit.
- The ticket id is checked, and there must be exactly one submit.

### Known open

- site-shift-meter: negating the control's scores passes (AUROC 0.989). It needs to know that `f4` flips at S2, which the agent can't see but this repo's docs state. A second hidden eval site where `f4` doesn't flip would close it.
