"""Grade any incident-desk policy. The other tests here only exercise the reference.

The candidate is a module with run_policy(desk). It may also define
write_artifacts(desk, submitted, out_dir) to write its own audit and
resolution, the way an agent with shell access could. The candidate runs in a
subprocess, then the verifier scores what it left behind.

version="1.0" scores with the original verifier rules (kept here as
_score_v10). version="1.1" uses environment/app/verifier.py.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

__test__ = False  # helpers, not tests (pytest-relaxed collects every public function)

TASK = Path(__file__).resolve().parents[1]
APP = TASK / "environment" / "app"
SCENARIO = APP / "data" / "scenario.json"
PROTECTED = [SCENARIO, APP / "desk.py", APP / "verifier.py"]

HARNESS = r"""
import importlib.util, json, sys
from pathlib import Path
app, candidate, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(app))
from desk import Desk
spec = importlib.util.spec_from_file_location("candidate_policy", candidate)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
desk = Desk.from_scenario_path(app / "data" / "scenario.json")
submitted = mod.run_policy(desk)
out.mkdir(parents=True, exist_ok=True)
if hasattr(mod, "write_artifacts"):
    mod.write_artifacts(desk, submitted, out)
else:
    with (out / "audit.jsonl").open("w") as f:
        for row in desk.audit:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    keys = ["ticket_id", "root_cause", "actions", "evidence_log_ids", "final_payments_health", "final_timeout_ms"]
    (out / "resolution.json").write_text(json.dumps({k: submitted[k] for k in keys}, indent=2) + "\n")
"""


def _load(path: Path, name: str):
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _score_v10(scenario_path: Path, audit_path: Path, resolution_path: Path) -> dict[str, Any]:
    """The verifier as it shipped in v1.0, unchanged."""
    desk_mod = _load(APP / "desk.py", "desk_v10")
    scenario = json.loads(scenario_path.read_text())
    desk = desk_mod.Desk.from_scenario_path(scenario_path)
    for line in audit_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        getattr(desk, row["tool"])(**(row.get("args") or {}))
    resolution = json.loads(resolution_path.read_text())
    baseline = scenario["baseline_config"]["payments.timeout_ms"]
    opened = int(scenario["ticket"]["opened_ts"])
    evidence = [
        r["id"]
        for r in sorted(scenario["logs"], key=lambda r: int(r["ts"]))
        if r["service"] == "payments" and int(r["ts"]) >= opened and "timeout" in r["msg"]
    ]
    checks = {
        "root_cause": resolution.get("root_cause") == "config_drift:payments.timeout_ms",
        "actions": set(resolution.get("actions") or []) >= {"set_config", "restart_service"},
        "evidence": list(resolution.get("evidence_log_ids") or []) == evidence,
        "timeout_restored": desk.scenario["config"].get("payments.timeout_ms") == baseline,
        "payments_ok": desk._payments_health_status() == "ok",
        "resolution_health": resolution.get("final_payments_health") == "ok",
        "resolution_timeout": resolution.get("final_timeout_ms") == baseline,
        "submitted": any(r.get("tool") == "submit_resolution" for r in desk.audit),
    }
    return {"pass": all(checks.values()), "checks": checks}


def grade(candidate: Path, version: str = "1.1") -> dict[str, Any]:
    snapshot = {p: p.read_bytes() for p in PROTECTED}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "artifacts"
        run = subprocess.run(
            [sys.executable, "-c", HARNESS, str(APP), str(candidate), str(out)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if run.returncode != 0:
            return {"pass": False, "checks": {"policy_ran": False}, "error": run.stderr[-400:]}
        untouched = all(p.read_bytes() == b for p, b in snapshot.items())
        if not untouched:
            for p, b in snapshot.items():
                p.write_bytes(b)
        audit, resolution = out / "audit.jsonl", out / "resolution.json"
        try:
            if version == "1.0":
                result = _score_v10(SCENARIO, audit, resolution)
            else:
                verifier = _load(APP / "verifier.py", "verifier_v11")
                result = verifier.score(SCENARIO, audit, resolution)
                result["checks"]["fixtures_unchanged"] = untouched
                result["pass"] = result["pass"] and untouched
        except Exception as exc:
            return {"pass": False, "checks": {"replay": False}, "error": repr(exc)}
    return result
