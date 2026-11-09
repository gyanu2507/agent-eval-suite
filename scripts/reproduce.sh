#!/usr/bin/env sh
# One-command reproduction. Needs Python 3.10+ and pytest. No other deps, no GPU.
set -eu
cd "$(dirname "$0")/.."

for t in site-shift-meter clearing-hold-ledger slot-ring-buffer incident-tool-desk; do
  echo "== tests: $t"
  python3 -m pytest -q -p no:randomly "tasks/$t/tests"
done

echo "== site-shift-meter oracle report"
python3 tasks/site-shift-meter/solution/train_and_eval.py

echo "== site-shift-meter ablation (writes docs/figures/site-shift-ablation.{json,svg})"
python3 scripts/site_shift_ablation.py
