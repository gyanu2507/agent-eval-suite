# Changelog

## 0.2.0 (unreleased)

Red team and verifier fixes. Every task moves from version 1.0 to 1.1 in its `task.toml`. The tech report (DOI 10.5281/zenodo.23829461) describes the 1.0 tasks and isn't changed.

### Added

- `tasks/<task>/tests/grade.py` for all four tasks. Each one grades any submission, under the 1.0 or the 1.1 rules.
- `redteam/`: 21 attacks, 4 alternate correct solutions, and `run.py`, which grades all of them under both rule sets. 12 attacks passed 1.0. 1 passes 1.1.
- `scripts/reproduce.sh` runs the red team after the tests.

### Changed

site-shift-meter
- The submission writes `artifacts/scores.csv`, one score per eval row. The verifier recomputes model AUROC from it and rejects a report that disagrees. 1.0 read `model_auroc` from the report, so a typed-in 0.99 passed.
- Gate raised from 0.08 to 0.5. Per-site z-scoring that keeps the `f4` shortcut scored +0.096 and cleared 0.08.
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

- site-shift-meter ships eval labels in the agent-visible `eval.csv`. `redteam/site-shift-meter/attacks/label_leak.py` scores with them and passes.
