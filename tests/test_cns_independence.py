"""URE is independent of CNS, and these tests hold whether or not CNS is installed.

Each runs in a fresh interpreter in which ``cns`` is blocked outright
(``sys.modules['cns'] = None`` makes any import of it fail), so the result does not
depend on what the test environment happens to contain. They never skip.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ure_engine"
BLOCK = "import sys; sys.modules['cns'] = None; sys.modules['cns.gate'] = None\n"
ENV = {
    "PYTHONPATH": str(ROOT),
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "PYTHONDONTWRITEBYTECODE": "1",
}


def run(code: str, *, block: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", (BLOCK if block else "") + textwrap.dedent(code)],
        capture_output=True, text=True, env=ENV, timeout=120, check=False,
    )


def ok(done: subprocess.CompletedProcess[str]) -> None:
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().endswith("ok"), done.stdout


def test_the_package_and_its_connector_import_with_cns_blocked():
    ok(run("import ure_engine, ure_engine.cns_connector; print('ok')"))


def test_the_engine_still_does_its_job_with_cns_blocked():
    ok(run("""
        from ure_engine import UREEngine
        a = UREEngine().observe({"latency_ms": 50, "error_rate": 0.0}, now=1000.0)
        assert a.decision.value == "ALLOW", a.decision
        print('ok')
    """))


def test_the_connector_says_what_is_missing_when_cns_is_blocked():
    ok(run("""
        from ure_engine.cns_connector import CnsNotInstalled, UREGate, cns_available
        assert cns_available() is False
        try:
            UREGate()
        except CnsNotInstalled as e:
            assert "pip install 'ure-engine[cns]'" in str(e)
        else:
            raise SystemExit("constructed without cns")
        print('ok')
    """))


def test_importing_the_package_does_not_load_cns_even_when_it_is_installed():
    ok(run("""
        import sys
        import ure_engine, ure_engine.cns_connector
        assert not [m for m in sys.modules if m == "cns" or m.startswith("cns.")], "cns was loaded"
        print('ok')
    """, block=False))


def test_no_module_of_the_package_imports_cns_by_statement():
    offenders: dict[str, list[str]] = {}
    for path in PACKAGE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            bad = [n for n in names if n == "cns" or n.startswith("cns.")]
            if bad:
                offenders.setdefault(path.name, []).extend(bad)
    assert offenders == {}


def test_the_package_declares_no_runtime_dependency_and_pins_the_cns_extra():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert meta["dependencies"] == []
    pin = "cns @ git+https://github.com/wking53214/CNS.git@v"
    assert any(dep.startswith(pin) for dep in meta["optional-dependencies"]["cns"])
