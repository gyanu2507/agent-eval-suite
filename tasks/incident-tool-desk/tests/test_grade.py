from __future__ import annotations

import importlib.util
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]


def _grade():
    spec = importlib.util.spec_from_file_location("grade_desk", Path(__file__).with_name("grade.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.grade


def test_reference_passes_grader():
    assert _grade()(TASK / "solution/policy.py")["pass"] is True


def test_starter_fails_grader():
    assert _grade()(TASK / "environment/app/policy.py")["pass"] is False
