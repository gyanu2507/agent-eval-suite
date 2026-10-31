# Judge vs verifier: `incident-tool-desk`

Long-horizon tool env. The **verifier** replays `audit.jsonl` and checks world state. A naive **rubric judge** that only reads `resolution.json` text can green-light cheats.

Fixtures: `docs/fixtures/incident-tool-desk/`.

## Rubric judge (text-only, intentionally weak)

Score +1 for each:

1. `root_cause` equals `config_drift:payments.timeout_ms`
2. `actions` include both `set_config` and `restart_service`
3. `evidence_log_ids` equals `["L-9","L-11"]`
4. `final_payments_health` is `ok`
5. `final_timeout_ms` is `5000`

Pass threshold: score >= 4 / 5.

This judge never replays the audit or the clock.

## Verifier (artifact-backed)

`environment/app/verifier.py` replays the audit on a fresh desk and checks config, payments health after settle, evidence recomputed from scenario, and resolution fields.

## Results (checked-in fixtures)

| Trace | Verifier pass | Judge score | Judge pass (>=4) | What happened |
| --- | --- | --- | --- | --- |
| `oracle-pass` | yes | 5/5 | yes | Config restore, payments restart, clock settle, evidenced submit |
| `starter-fail` | no | 0/5 | no | Lexicographic log sort, restarts `api`, wrong cause |
| `premature-submit` | no | 5/5 | **yes** | Right fields written, but clock not advanced through settle. Payments still restarting. |

## Takeaway

`premature-submit` is the interesting row: the resolution JSON is perfect and a text judge passes it. The verifier fails `payments_ok` / `resolution_health` after audit replay. Same class of bug as fake deltas: the report looks finished while the world is not.

Reproduce:

```bash
python3 - <<'PY'
from pathlib import Path
import json, sys
sys.path.insert(0, 'tasks/incident-tool-desk/environment/app')
from verifier import score
root = Path('docs/fixtures/incident-tool-desk')
sc = Path('tasks/incident-tool-desk/environment/app/data/scenario.json')
for name in ['oracle-pass','starter-fail','premature-submit']:
    d = root/name
    print(name, score(sc, d/'audit.jsonl', d/'resolution.json')['pass'])
PY
```
