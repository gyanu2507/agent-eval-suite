from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
APP = TASK / "environment" / "app"
SCENARIO = APP / "data" / "scenario.json"
SOLUTION_POLICY = TASK / "solution" / "policy.py"
STARTER_POLICY = APP / "policy.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Ensure desk importable for policy modules
    sys.path.insert(0, str(APP))
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _desk():
    desk_mod = _load(APP / "desk.py", "desk_mod")
    return desk_mod.Desk.from_scenario_path(SCENARIO)


def _write_artifacts(desk, submitted, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "audit.jsonl").open("w") as f:
        for row in desk.audit:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    resolution = {
        "ticket_id": submitted["ticket_id"],
        "root_cause": submitted["root_cause"],
        "actions": submitted["actions"],
        "evidence_log_ids": submitted["evidence_log_ids"],
        "final_payments_health": submitted["final_payments_health"],
        "final_timeout_ms": submitted["final_timeout_ms"],
    }
    (out / "resolution.json").write_text(json.dumps(resolution, indent=2) + "\n")


def test_starter_fails_verifier(tmp_path: Path):
    desk = _desk()
    starter = _load(STARTER_POLICY, "starter_policy")
    submitted = starter.run_policy(desk)
    out = tmp_path / "starter"
    _write_artifacts(desk, submitted, out)
    ver = _load(APP / "verifier.py", "verifier_starter")
    result = ver.score(SCENARIO, out / "audit.jsonl", out / "resolution.json")
    assert result["pass"] is False
    assert result["checks"]["root_cause"] is False


def test_oracle_passes_verifier(tmp_path: Path):
    desk = _desk()
    oracle = _load(SOLUTION_POLICY, "oracle_policy")
    submitted = oracle.run_policy(desk)
    out = tmp_path / "oracle"
    _write_artifacts(desk, submitted, out)
    ver = _load(APP / "verifier.py", "verifier_oracle")
    result = ver.score(SCENARIO, out / "audit.jsonl", out / "resolution.json")
    assert result["pass"] is True
    assert result["checks"]["evidence"] is True
    assert result["expected_evidence"] == ["L-9", "L-11"]


def test_pretty_resolution_without_audit_fails(tmp_path: Path):
    out = tmp_path / "fake"
    out.mkdir(parents=True)
    (out / "audit.jsonl").write_text("")
    fake = {
        "ticket_id": "INC-2048",
        "root_cause": "config_drift:payments.timeout_ms",
        "actions": ["set_config", "restart_service"],
        "evidence_log_ids": ["L-9", "L-11"],
        "final_payments_health": "ok",
        "final_timeout_ms": 5000,
    }
    (out / "resolution.json").write_text(json.dumps(fake) + "\n")
    ver = _load(APP / "verifier.py", "verifier_fake")
    result = ver.score(SCENARIO, out / "audit.jsonl", out / "resolution.json")
    assert result["pass"] is False
    assert result["checks"]["payments_ok"] is False


def test_scenario_unchanged_bytes():
    before = SCENARIO.read_bytes()
    desk = _desk()
    oracle = _load(SOLUTION_POLICY, "oracle_policy_bytes")
    oracle.run_policy(desk)
    assert SCENARIO.read_bytes() == before


def test_lexicographic_log_trap():
    desk = _desk()
    logs = desk.get_logs("api", since_ts=1000)
    by_id = sorted(logs, key=lambda r: str(r["id"]))
    by_ts = sorted(logs, key=lambda r: int(r["ts"]))
    assert by_id[0]["id"] == "L-10"
    assert by_ts[0]["id"] == "L-2"
