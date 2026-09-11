# Provenance

URE existed in concept across an extended design effort before it existed as
code. This document records where each part of it came from, what was
recovered verbatim, what was corrected, and what is new, so the lineage is
auditable rather than asserted.

**Status of this repository:** a reconstruction. The design is recovered from
primary sources; the implementation is written against that design. Where the
archive contained working code, it was carried forward. Where the archive
contained a specification without an implementation, the specification is cited
below and the implementation is new.

---

## 1. Primary sources

All sources are conversation transcripts in the author's own archive
repositories, reachable by the identifiers below.

| # | Source | Identifier | Contribution |
|---|---|---|---|
| S1 | `wking53214/ChatGPT_History` | `transcripts/6a35c605-ca08-83ea-b911-b8119f947c15.md`: *"GSA URE Architecture Continuation"*, 2026-06-19 | **The authoritative specification.** Continuation prompt for the GSA → URE evolution project: AEGIS hierarchy, design objective, all eight core components, integration contracts, the `UREAssessment` canonical contract, the target module layout, execution flow. |
| S2 | `wking53214/ChatGPT_History` | `transcripts/6a35c4a6-92d0-83ea-bd81-9d52aa36f1c4.md`: *"GSA.zip Analysis Request"*, 2026-06-19 | Attack Memory, Behavioral Vaccine Engine, Adaptive Threshold Controller v2 with its three calibration anchors, the Recovery Engine action ladder, the `RecoveryVector` example. |
| S3 | `wking53214/ChatGPT_History` | `transcripts/6a5e3542-be20-83ea-b6c0-992c086bc48e.md`: *"Module Architecture Recommendation"*, 2026-07-20 | **The hardened `ure_engine.py`** (297 lines, recovered verbatim) and the `ure_compat.py` shim. Contains the energy-recentering fix and its rationale. |
| S4 | `wking53214/ChatGPT_History` | `transcripts/6a679bd9-8ec0-83ea-bcf1-59e1fc6e6669.md`: *"Sentinel OS for Drones"*, 2026-07-29 | The **UGPIS-Ω v1 engine**: `SystemResilienceConfig`, `OperationalRegime`, `SystemMetricsTelemetry`, `IntegratedResilienceOrchestrator`. The predecessor URE. |
| S5 | `wking53214/ARCHIVE` | `TRANSCRIPT.md`, `README.md`, `artifact_*.json` | Registry entries identifying URE as `SYS-URE-001` / `ARCH-URE-001` and its domain classification. |

Identifiers S1–S5 are used throughout this document.

---

## 2. What URE is, per the archive

From S5, the registry description:

> **URE (Universal Resilience Engine) module**
> *Domain:* Nonlinear Dynamical Systems and Risk Modeling.
> *Scope:* A predictive analytics engine designed to detect operational regimes
> and track systemic stability vectors using statistical modeling.
> *Core Mechanics:* Evaluates system telemetry using Lyapunov energy metrics,
> tracks mathematical probability fields (Shannon entropy), isolates operational
> profiles, and computes real-time volatility thresholds.

From S1, the design objective:

> URE becomes the authoritative system state engine.
> Instead of computing: `risk`
> it computes: `stability`, `trajectory`, `regime`, `resilience`, `recovery`,
> `adaptation`.

---

## 3. The originating gap

GSA's gateway (S1, S3) contained:

```python
from ure_engine import ClassificationResult, GatewayHealthEngine  # UGPIS-Ω regime engine
```

S1 records the finding directly:

> No `ure_engine.py` existed in the archive.
> The goal is to replace this missing dependency with a next-generation
> Universal Resilience Engine rather than a simple health monitor.

And the architectural conclusion (S1):

> `gsa_gateway.py` is the governance shell.
> `policy_api.py` is the architectural contract.
> `ure_engine.py` is the missing resilience subsystem.

This repository is that subsystem. `ure_engine/__init__.py` binds
`GatewayHealthEngine` and `ClassificationResult` so the original import
resolves unchanged.

---

## 4. Recovered verbatim

### 4.1 Architecture and naming, S1

Carried through without alteration:

- The AEGIS hierarchy (GSA · URE · LTE · AMX · BVE · FORTRESS · WS3 · Dashboard).
- The six-dimensional `StateVector` and its exact field names.
- The seven-member `SystemRegime` enum and its member names.
- The `UREAssessment` field list and field order, S1 specifies this explicitly
  "to prevent drift across future implementations", and `assessment.py`
  reproduces it exactly.
- The `GatewayHealthEngine` protocol signature: `async def evaluate(self, state:
  StateVector) -> UREAssessment`.
- The module layout (`state_vector.py` through `ure_engine.py`) and the
  per-module policy-dependency table, reproduced in `ARCHITECTURE.md` §13.
- The execution flow, reproduced in `ARCHITECTURE.md` §2.

### 4.2 The hardened engine, S3

S3 contains a complete, corrected 297-line `ure_engine.py`. Carried forward:

- `V(x) = 1.5 · tanh(Σ w·p² / 2)` and the weight tuple
  `(1.4, 0.8, 1.3, 0.7, 1.5, 0.9)`.
- `ENERGY_MAX = 1.5`.
- `SystemRegime.health_status`, the three-state projection onto GSA's
  vocabulary.
- The `TelemetryAdapter` pressure mappings, including the `circuit_state ==
  "OPEN"` → +0.5 failure step.
- The regime calibration bands, retained as the anchor points the affinity
  functions in `regimes.py` are fitted to.
- The `__main__` reachability sweep, carried forward as
  `test_full_state_space_sweep`.

### 4.3 Component specifications, S1, S2

`AttackProfile` fields, the AMX capability set (`remember` / `lookup` / `decay`
/ `forget`), `BehavioralVaccine` fields, the `Vaccine-N` naming convention, the
worked vaccine example (`ignore previous instructions` → `developer override` →
`show prompt`), the five-action recovery ladder, the `RecoveryVector` control
keys (`reduce_rate_limit`, `disable_learning`, `freeze_vaccines`), the
resilience index range and its five contributing terms, and the adaptive
threshold anchors NOMINAL 0.85 / ATTACKED 0.45 / CASCADING 0.20.

### 4.4 The compatibility contract, S3

The exact legacy surface, from the `ure_compat.py` docstring in S3:

```
summary = health_engine.assess(reject_rate=, retry_rate=, latency_ms=, backlog=)
summary.result_status / dominant_regime / composite_risk_score
summary.calculated_energy / regime_entropy / regime_confidence
summary.rationale_statement
```

`compat.py` reproduces this field-for-field, including S3's rule that the shim
must never report calmer than the engine.

---

## 5. Defects recovered from the archive, and their fixes

### D1, Energy had a non-zero floor

**Recorded in S3, verbatim:**

> ENERGY RECENTERED. The redesign used `1.5*sigmoid(Σ pressure²)`, whose floor
> is 0.75 at zero load, which made NOMINAL and RECOVERING mathematically
> unreachable.

**Fix (S3, carried forward):** `1.5 · tanh(Σp²/2)`: algebraically the same
sigmoid with its 0.5 baseline removed. Exactly 0 at rest.

**Guarded by:** `test_zero_state_has_exactly_zero_energy`,
`test_full_state_space_sweep`.

### D2, Sustained rejections never left NOMINAL

**Recorded in S3, verbatim:**

> feeding reject_rate only into blocked_rate under-weights it (0.3) and the
> regime never leaves NOMINAL even at a 60% block rate.

**Fix:** `GSATelemetryAdapter` drives threat pressure from `reject_rate`
directly. From a gateway's point of view a surge of rejections *is* the threat
signal.

**Guarded by:** `TestTheDefectTheShimExistedToWorkAround`: five tests,
including monotonic status escalation across the rejection range.

### D3, AMX, BVE and the recovery planner were dead code

**Recorded in S3, verbatim:**

> DEAD SUBSYSTEMS REMOVED. AttackMemoryExchange / BehavioralVaccineEngine /
> RecoveryPlanner were instantiated but never called.

**S3's decision was to delete them**, and that was correct for that build: a
dead subsystem is worse than an absent one. But the reason they were never
called is that nothing fed them: the engine had no signature or marker channel.

**Fix:** they are restored **wired**. `TelemetryFrame` carries opaque
`signatures` and `markers`; AMX and BVE run *upstream* of regime classification
so their output raises `adversarial_pressure` and changes the assessment; the
recovery vector gates the engine's own learning.

**Guarded by:** `TestMemoryIsActuallyWired`: five tests asserting the
subsystems change outcomes, not merely that they exist.

### D4, Trend computed twice

**Recorded in S3:** *"Trend is computed once (TrajectoryEngine), not twice."*

**Fix:** `LyapunovTrajectoryEngine` owns the smoothed regression slope used for
decisions; `TrajectoryEngine` owns raw finite-difference motion used for
explanation. Distinct responsibilities, one source each.

---

## 6. Defects found while building this implementation

These were not in the archive. They were found by writing tests and by running
the demo, and are recorded here because they are the kind of defect that
survives review.

### D5, NaN energy reported NOMINAL

`clamp()` maps NaN to its floor, which is the right defensive choice for a
*pressure* and catastrophic for *energy*: a NaN energy normalized to 0.0 and
the classifier confidently returned NOMINAL for a computation that had broken.

**Fix:** non-finite energy, derivative or volatility returns `UNKNOWN`.
Failing visibly beats failing reassuringly.

**Found by:** `test_non_finite_energy_yields_unknown`.

### D6, Hysteresis could never release

The recovery planner recorded *held* actions into its history, so every hold
re-asserted the elevated action as evidence for continuing to hold it. A system
that entered QUARANTINE could never leave, a permanent lockout.

**Fix:** the planner records what conditions **proposed**, tracked separately
from what is **applied**.

**Found by:** `test_controls_are_released_once_calm_persists`.

### D7, Held vectors carried the wrong controls

A held recommendation carried the *proposed* action's control map. An executor
would receive `action=QUARANTINE` alongside `controls={...resume admission...}`
and obey the controls, silently undoing the hold.

**Fix:** held vectors carry the applied action's controls.

**Found by:** inspecting `examples/incident_walkthrough.py` output.

### D8, A traffic surge was classified as CASCADING

A steep but legitimate load ramp is indistinguishable from a cascade on every
signal the affinity function used, rising energy, high failure and resource
pressure, so the demo quarantined a system that was merely busy.

**Fix:** CASCADING is *gated* below 0.65 normalized energy rather than weighted,
restoring S3's calibrated `energy >= 1.0` band as a hard precondition.

**Found by:** running `examples/incident_walkthrough.py`.

---

## 7. What is new in this implementation

Specified in the archive but never implemented, or added here for soundness:

| Component | Basis |
|---|---|
| Soft affinity classification with confidence, entropy, runner-up and margin | S1 requires classification use energy, derivative, memory and trajectory "not static thresholds"; the predecessor used a threshold cascade and a hard-coded `mock_distribution`. |
| Least-squares `dV/dt` with epoch re-basing | New. Required for a trend signal stable enough to gate regime transitions. |
| Oscillation detection | New. Closes the case where a flapping system has a mean slope of zero. |
| `time_to_threshold` early warning | New. Turns "rising" into an actionable interval. |
| AMX decay, probabilistic union, ambient pressure, fleet exchange | S1/S2 specify `decay()`; the half-life model, union aggregation, ambient term and exchange format are new. |
| BVE discrimination threshold, subsequence matching, Laplace-smoothed waning | S1/S2 specify the vaccine fields; the synthesis criterion and the feedback loop are new. |
| Recovery hysteresis | New. S1/S2 specify the action ladder but not damping. |
| Regime ceiling on the resilience index | New. Prevents a reassuring number during an outage. |
| `PolicyTelemetry` | **Specified in S1** as the single recommended extension for URE v2, implemented here as specified. |
| Mechanical policy-seam enforcement | New. S1 states the invariant; `test_ure_does_not_import_policy_modules` enforces it. |

The v1 engine's `mock_distribution` is worth noting explicitly. S4's
`SystemRegimeClassifier.classify_current_regime` contained a hard-coded
probability table with the comment *"Simulated distribution mapping function"*.
It was never a working classifier. `regimes.py` replaces it with a real one.

---

## 8. Naming reconciliation

Two regime vocabularies appear in the archive. S4 (UGPIS-Ω v1) used
`OperationalRegime`: `STABLE`, `SURGE`, `RESOURCE_OVERLOAD`,
`ANOMALOUS_CRITICAL`, `SATURATED`, `CONTESTED`. S1 (the v2 target) specifies
`SystemRegime`: `NOMINAL`, `STRESSED`, `ATTACKED`, `CASCADING`, `RECOVERING`,
`ADAPTING`, `UNKNOWN`.

This implementation uses the **S1 v2 vocabulary**, which S1 names as the
explicit replacement target. The v1 names are not retained; the v1
`ClassificationResult` three-state vocabulary (`NEUTRAL` / `REGRESSIVE` /
`RISK_INCREASING`) *is* retained, in `compat.py` only, because GSA reads it.

---

## 9. Verification

| | |
|---|---|
| Tests | 278, all passing |
| Runtime dependencies | none |
| Python | 3.11+ (S1 specifies 3.12+; the implementation targets 3.11 and runs on both) |
| Archive defects with regression tests | D1, D2, D3 |
| Build-time defects with regression tests | D5, D6, D7 |
| Reachability sweep | 5⁶ states × 5 derivatives × 3 volatilities |

The reachability sweep from S3's `__main__` block is the single most important
inherited test: it is what exposed D1 originally, and it will expose any
recurrence of that class of defect.

---

## 10. Honest limitations

- **Energy saturation loses trajectory.** Once `V(x)` is pinned near 1.5 there
  is no headroom left for a derivative to register, so a system already at the
  ceiling cannot be seen to be "rising". CASCADING detection depends on
  catching the climb, not the plateau.
- **Regime affinity weights are calibrated, not learned.** They are fitted to
  the archive's calibration points and validated by sweep and scenario tests.
  They are not derived from production data, because there is none.
- **AMX and BVE depend on the integrator's signature quality.** URE never sees
  request content by design, so the discriminating power of a signature or
  marker is entirely the gateway's responsibility.
- **The v1 `SystemResilienceConfig` tuning constants are not carried forward.**
  They were fitted to a different (unnormalized, mixed-unit) state model and do
  not transfer.
- **No production deployment has exercised this.** Every number in the README
  and docs is from the test suite or `examples/incident_walkthrough.py`.
