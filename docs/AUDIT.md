# Integrity Audit

URE is scanned with [ghost_tools](https://github.com/wking53214/ghost_tools)
`ghost_buster` v1.5.0, the same integrity toolkit used on `sentinel_os` and
`innovation_os`. This file records the current state, what was fixed because of
it, and what is deliberately left alone.

`.ghost_baseline.json` is tracked, matching the convention in the sibling
repositories. `.ghost_ledger.json` is run history and is gitignored.

## Reproducing

```bash
ghost-buster . --trust --secrets                # standard scan
ghost-buster . --mutate --mutate-timeout 120    # mutation analysis
```

`--trust` records consent for the tool to execute this repository's test suite.
`--secrets` needs gitleaks on PATH.

## Current state

| | |
|---|---|
| Files scanned | 29 |
| MAJOR findings | **0** |
| MINOR findings | 8, all baselined (see below) |
| Tests | 278 collected, 278 passed, 0 reruns |
| Committed secrets | 0 (gitleaks 8.21.2) |
| Structure | 25 modules, 2 entry points, 0 findings |
| Surviving mutants | **0** |
| Serum candidacy | **CANDIDATE**, all six criteria met |

The six candidacy criteria: parses completely, tests run and pass, no committed
secrets, not a drifted copy, no swallowed-everything handlers, no hollow
contracts.

## What the scan caught, and what changed

### Mutation analysis found two vacuous tests

This is the finding worth reading. Both were rated MAJOR `vacuous_check`, and
both were mine:

| Test | Mutation that survived |
|---|---|
| `TestImmunization::test_interleaved_markers_still_match` | 9 numeric constants in `BehavioralVaccineEngine.immunize` moved by +100 |
| `TestAdaptationScore::test_learning_raises_the_score` | 4 numeric constants in `BehavioralVaccineEngine.adaptation_score` moved by +100 |

Both tests asserted only a direction (`> 0.0`). Because a vaccine synthesized
from purely adverse observations has potency exactly 1.0, the values under test
were already saturated at the clamp ceiling, so inflating every constant in the
function by 100 changed nothing the assertions could see. The tests passed for
a reason unrelated to whether the code was correct.

Both now pin the real arithmetic: exact pressure, `best_match`, the
`preemptive` flag, the activated vaccine ids, and `adaptation_score` against
its derived value of 0.85 (`0.70 x 1.0 + 0.30 x 0.5`). Re-running mutation
analysis afterwards: **0 survived, 1 killed**, and neither test is flagged as a
weak candidate any more.

This is the concrete instance of the claim in
[`RECONSTRUCTION.md`](../RECONSTRUCTION.md) section 6: a suite that agrees with
the code is not evidence the code is right. 278 tests passed throughout; two of
them were proving nothing, and only mutation testing could tell the difference.

### Two loop-invariant calls fixed

`examples/incident_walkthrough.py` rebuilt `list(ATTACK_MARKERS)` on every
iteration of two timeline loops. `ATTACK_MARKERS` is already an immutable tuple
and the telemetry layer normalizes it to a tuple regardless, so the call was
pure waste. Now passed directly. Demo output is byte-identical.

## What is baselined, and why

All 8 are MINOR. None are suppressed silently; each is a judgment recorded here.

### `long_function` x4

| Location | Lines |
|---|---|
| `ure_engine/engine.py:212` `observe` | 113 |
| `ure_engine/governance_adapter.py:177` `decide` | 123 |
| `ure_engine/recovery.py:162` `_propose` | 139 |
| `examples/incident_walkthrough.py:37` `phases` | 82 |

Threshold is 80 lines, and the tool is explicit that this is a line-count proxy
rather than cyclomatic complexity: "a long function that's mostly a flat
sequence of simple statements is a weaker signal than a short function with
deep branching."

All four are flat sequences, and in three of them the flatness is the design:

* `observe` is the ten-step pipeline that `docs/ARCHITECTURE.md` section 2
  documents in order. Splitting it would hide the ordering, and the ordering is
  load-bearing (AMX and BVE must run before classification).
* `decide` is a precedence ladder with four documented tiers. The precedence is
  the contract; breaking it into helpers would scatter it.
* `_propose` is a regime-to-action dispatch, one branch per regime, each
  constructing a `RecoveryVector` with its rationale.

Kept as is. Revisit if branching depth grows rather than length.

### `intra_function_duplicate_block` x2

`governance_adapter.py:290` and `incident_walkthrough.py:50`. Single repeated
statements with differing literals, inside branches that are deliberately
parallel in shape. Extracting a helper for one statement would cost more
indirection than it saves.

### `name_disagreement` x2

* `ure_engine/vaccines.py:222`, parameter `trigger` versus local `observation`.
  `trigger` names the parameter's role (the observation that triggered
  synthesis), which is more informative at the callee than `observation`.
  Kept.
* `ure_engine/regimes.py:163`, parameter `x` versus local `normalized_energy`.
  `x` is the parameter of the generic math helpers `_bump` and `_gate`, which
  are deliberately domain-agnostic. Naming it `normalized_energy` would bind a
  general function to one caller's meaning. Kept.

Both are genuine 1:1 disagreements and the tool is right to surface them. The
decision in each case is that the callee's name is the better one.

## The CI gate

`.github/workflows/ci.yml` runs an `integrity` job on every push and pull
request. It installs ghost_tools and gitleaks, then runs:

```bash
python tools/integrity_gate.py . --secrets-binary /tmp/gitleaks --mutate --mutate-timeout 120
```

The gate blocks on any **new** finding at `major` or `critical`. Baselined
findings are suppressed by ghost_buster itself, so anything reaching the gate
is new since this baseline was accepted. Mutation analysis runs inside the same
step, measured at about 9 seconds on this suite, which is cheap enough to run
on every push rather than on a schedule.

Verified in a clean checkout, both directions:

| Condition | Result |
|---|---|
| Unmodified tree | `0 finding(s) beyond baseline`, exit 0 |
| Injected `swallowed_exception` | `FAIL -- 1 blocking finding(s)`, exit 1 |

The gate deliberately does **not** use ghost_buster's exit code. That is
non-zero whenever the tool has anything at all to say, INFORMATIONAL blind
spots included, so gating on it would fail every build for non-defects, and a
gate that always fails gets switched off.

### What the gate does not catch

Measured, not assumed. `--mutate` selects tests "shaped like they check
nothing". That judgment is made over the whole test, so a weak assertion inside
an otherwise strong test is invisible to it. Reverting
`test_learning_raises_the_score` to a bare `assert ... > 0.0` while leaving the
neighbouring `assert len(bve) == 4` in place produced **zero** findings, where
the original fully-vacuous form had been caught as MAJOR.

A green gate therefore means "no new finding of a kind ghost_buster looks for".
It is not evidence that the suite is load-bearing.

## Checks that do not run here, and why

The tool flags checks that never run, on the grounds that "a check that never
runs and says nothing is indistinguishable in the output from a check that ran
and found nothing." Recording the reasons, as it asks:

| Check | Status | Why |
|---|---|---|
| `boundary` | Not applicable | Single repository reaching for no unprovided packages. URE has zero runtime dependencies, so there is no boundary to scan. |
| `kernel` | Not applicable | Opt-in, requires `--kernel PATH` pointing at a kernel spec. URE defines no kernel contract. |
| `correlate` | Nothing to connect | No second finding source to correlate against. |
| `branches` | Disabled in the gate | `unmerged_branch` reports MAJOR when a branch has commits not in the default branch. On a pull request that is the definition of a pull request, and the detector states it "has no visibility into GitHub pull-request state". Left on, it would fail every PR by construction. Run `ghost-buster .` without `--no-branches` locally when you do want the check. |

These are configuration facts, not defects, which is why they are documented
here rather than carried in the baseline: they depend on run history and would
otherwise churn it on every scan.

## Trajectory

Trajectory assessment needs three comparable runs. The ledger was reset when
this baseline was written, so the first meaningful trend reading is three scans
from now.
