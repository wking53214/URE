"""The integrity gate reads both shapes ghost-buster's --json has printed.

Up to ghost_tools 1.9.0 it was a bare list of findings. From ghost_tools #97 it
is a report object with the list under ``findings`` and an ``error`` key. The
gate read only the list, so with Ghost Tools V1.9.1 it stopped at "expected a
JSON list of findings" on every run.
"""
import importlib.util
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[1] / "tools" / "integrity_gate.py"
spec = importlib.util.spec_from_file_location("integrity_gate", GATE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

FINDING = {"severity": "major", "detector": "dead_code", "summary": "x"}


def test_the_old_bare_list_is_still_read():
    assert gate.findings_from([FINDING]) == [FINDING]


def test_the_new_report_object_is_read_through_its_findings_key():
    report = {"findings": [FINDING], "error": None, "exit_code": 1}
    assert gate.findings_from(report) == [FINDING]


def test_a_report_that_says_the_scan_errored_stops_the_gate():
    with pytest.raises(SystemExit) as stopped:
        gate.findings_from({"findings": [], "error": "target missing"})
    assert stopped.value.code == 2


@pytest.mark.parametrize("data", [{"no": "findings"}, "text", 3, {"findings": "not a list"}])
def test_anything_else_stops_the_gate_rather_than_passing_it(data):
    with pytest.raises(SystemExit) as stopped:
        gate.findings_from(data)
    assert stopped.value.code == 2


def test_new_findings_still_block_through_the_report_object():
    findings = gate.findings_from({"findings": [FINDING], "error": None})
    assert gate.report(findings) == 1
