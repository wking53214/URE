#!/usr/bin/env python3
"""
integrity_gate.py -- turn a ghost_buster scan into a pass/fail CI gate.

Why this exists rather than just checking ghost-buster's exit code
------------------------------------------------------------------
``ghost-buster`` exits non-zero whenever it has anything to say, including
INFORMATIONAL findings. On a clean tree it still reports its own blind spots
(checks that have not run here, such as ``boundary`` and ``kernel``, both of
which are documented as not applicable in ``docs/AUDIT.md``). Gating on the raw
exit code would therefore fail every build for reasons that are not defects,
and a gate that always fails gets switched off within a week.

So the gate reads the JSON finding set instead and blocks on severity. The
baseline does the rest: ``--json`` already suppresses anything recorded in
``.ghost_baseline.json``, so what reaches this script is by definition *new*.

What blocks
-----------
Any finding at ``major`` or ``critical``. In practice that means:

* a new ``vacuous_check`` -- a test that mutation analysis proved cannot fail,
  which is the finding this repository added the gate for;
* ``swallowed_exception``, ``hollow_contract``, ``dead_code`` and friends;
* ``regressed_finding`` -- something previously fixed has come back.

``minor`` and ``informational`` are reported and do not block. A new minor
finding is a prompt to look, and the right response is usually to fix it or to
baseline it with a written reason in ``docs/AUDIT.md``, not to fail the build.

What this does NOT catch
------------------------
Measured, not assumed. ``--mutate`` selects tests "shaped like they check
nothing" and then tries to break the code they call. That selection is a
heuristic over the whole test, so a weak assertion sitting inside an otherwise
strong test is invisible to it: adding one substantive assertion to a test is
enough to stop it being considered a candidate at all, even if a vacuous
assertion remains beside it.

Verified against this repository: reverting
``test_learning_raises_the_score`` to a bare ``assert ... > 0.0`` while leaving
a neighbouring ``assert len(bve) == 4`` in place produced zero findings, where
the original fully-vacuous form was caught as MAJOR.

So this gate raises the floor; it does not prove the suite is sound. Treat a
green gate as "no new finding of a kind ghost_buster looks for", never as "the
tests are load-bearing".

Why ``unmerged_branch`` is filtered rather than the check disabled
-----------------------------------------------------------------
The ``unmerged_branch`` detector reports MAJOR when a branch has commits not in
the default branch. On a pull request that is the definition of a pull request,
not a defect, and the detector says so itself: it "has no visibility into
GitHub pull-request state". Left blocking, it would fail every PR by
construction.

The obvious move is ``--no-branches``, and it is wrong. ghost_buster tracks
checks that never run and escalates them to MAJOR after five consecutive
skips, on the correct principle that a check which never runs is
indistinguishable from one that ran and found nothing. So suppressing the
branch check makes the gate fail on its own suppression a few runs later: a
self-defeating loop, invisible in CI only because the ledger does not persist
across fresh checkouts, and reproducible immediately on a developer machine.

So the check RUNS, and this gate filters the single detector whose MAJOR is
structurally meaningless here. The blind spot never accrues, and the false
positive never blocks.

The filter also has to follow derived findings. ghost_buster's ledger watches
for a finding that was fixed and came back and reports it one severity step
higher, which is right in general and wrong for this one: a branch legitimately
has unmerged commits, then does not once it merges, then does again on the next
branch. That oscillation is the normal shape of development, and the ledger
reports it as a CRITICAL regression. Derived findings name their origin in
``attributes.source_detector``, so the filter follows that link rather than
matching on the summary text, which would break the moment the wording changed.

Usage
-----
::

    python tools/integrity_gate.py              # scan . and gate
    python tools/integrity_gate.py --secrets-binary /path/to/gitleaks
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter

#: Severities that fail the build. Everything else is reported only.
BLOCKING: frozenset[str] = frozenset({"major", "critical"})

#: Detectors whose findings are structurally meaningless for this gate. Kept
#: deliberately tiny: every entry is a hole, and each one is justified in
#: docs/AUDIT.md.
EXEMPT_DETECTORS: frozenset[str] = frozenset({"unmerged_branch"})


def is_exempt(finding: dict[str, object]) -> bool:
    """True when this finding, or the finding it derives from, is exempt.

    Checking only ``detector`` would miss the ledger's ``regressed_finding``,
    which carries its own detector name and points at the original through
    ``attributes.source_detector``.
    """
    if str(finding.get("detector", "")) in EXEMPT_DETECTORS:
        return True
    attributes = finding.get("attributes")
    if isinstance(attributes, dict):
        return str(attributes.get("source_detector", "")) in EXEMPT_DETECTORS
    return False


def run_scan(
    path: str, secrets_binary: str | None, mutate: bool, mutate_timeout: int
) -> list[dict[str, object]]:
    """Run ghost-buster and return its finding set.

    ``--json`` emits findings already filtered against the baseline, so
    anything returned here is new since the baseline was accepted.
    """
    command = [
        "ghost-buster",
        path,
        "--trust",  # consent to executing this repository's own test suite
        "--json",
    ]
    if secrets_binary:
        command += ["--secrets", "--secrets-binary", secrets_binary]
    else:
        command += ["--no-secrets"]
    if mutate:
        command += ["--mutate", "--mutate-timeout", str(mutate_timeout)]

    completed = subprocess.run(command, capture_output=True, text=True)

    # A non-zero exit is expected whenever there are findings of any severity,
    # so it is not itself an error. Empty stdout is, because that means the
    # scan did not produce a finding set at all.
    if not completed.stdout.strip():
        sys.stderr.write(
            "integrity gate: ghost-buster produced no JSON output "
            f"(exit {completed.returncode}).\n{completed.stderr}\n"
        )
        raise SystemExit(2)

    try:
        findings = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"integrity gate: could not parse scan output: {exc}\n")
        raise SystemExit(2) from exc

    if not isinstance(findings, list):
        sys.stderr.write("integrity gate: expected a JSON list of findings\n")
        raise SystemExit(2)
    return findings


def report(findings: list[dict[str, object]]) -> int:
    """Print a summary and return the process exit code."""
    counts = Counter(str(finding.get("severity", "unknown")) for finding in findings)
    blocking = [
        finding for finding in findings
        if str(finding.get("severity", "")) in BLOCKING and not is_exempt(finding)
    ]

    summary = ", ".join(f"{n} {sev}" for sev, n in sorted(counts.items())) or "none"
    print(f"integrity gate: {len(findings)} finding(s) beyond baseline ({summary})")

    if not blocking:
        print("integrity gate: PASS -- nothing at major or critical severity")
        return 0

    print(f"\nintegrity gate: FAIL -- {len(blocking)} blocking finding(s)\n")
    for finding in blocking:
        severity = str(finding.get("severity", "?")).upper()
        detector = finding.get("detector", "?")
        print(f"  [{severity:8}] {detector}")
        print(f"      {finding.get('summary', '')}")
        evidence = finding.get("evidence")
        if isinstance(evidence, str) and evidence:
            print(f"      {evidence.splitlines()[0]}")
        print()

    print(
        "Fix the finding, or -- if it is a deliberate decision -- accept it into\n"
        "the baseline with `ghost-buster . --accept` and record the reasoning in\n"
        "docs/AUDIT.md. Silent suppression is what this gate exists to prevent."
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=".", help="directory to scan")
    parser.add_argument(
        "--secrets-binary",
        default=None,
        help="path to gitleaks; omitted means the secrets scan is skipped",
    )
    parser.add_argument(
        "--mutate",
        action="store_true",
        help="include mutation analysis (~9s on this suite)",
    )
    parser.add_argument("--mutate-timeout", type=int, default=120)
    args = parser.parse_args()
    return report(
        run_scan(args.path, args.secrets_binary, args.mutate, args.mutate_timeout)
    )


if __name__ == "__main__":
    raise SystemExit(main())
