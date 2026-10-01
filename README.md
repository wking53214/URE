# URE: Universal Resilience Engine

**SYS-URE-001 · v2.0 Reference Implementation · the resilience subsystem of AEGIS**

URE sits behind a governance gateway and answers a different question than a
risk scorer does.

A risk scorer asks *how dangerous is this request?* URE asks:

> Given everything happening to this system right now and everything it
> remembers, what state is it in, where is it heading, how much can it still
> absorb, and what should be done about it?

It computes **stability, trajectory, regime, resilience, recovery and
adaptation**: not a risk score.

```
278 tests · 0 runtime dependencies · Python 3.11+
```

---

## Why this repository exists

The GSA (Governed Secure AI Gateway) archive contained a gateway, a policy
seam, an attestation service, an audit ledger, and this line:

```python
from ure_engine import ClassificationResult, GatewayHealthEngine
```

There was no `ure_engine.py`. The governance shell had been built around a
resilience subsystem that was never written. An architectural review of that
archive put it plainly:

> `gsa_gateway.py` is the governance shell.
> `policy_api.py` is the architectural contract.
> `ure_engine.py` is the missing resilience subsystem.

This is that subsystem. The legacy import above works, unchanged, against the
full v2 engine. See [`docs/PROVENANCE.md`](docs/PROVENANCE.md) for where every
component came from.

URE had been dead for four months and had never possessed a repository at any
point in its life. It was reconstructed from 863 archived design conversations.
[`RECONSTRUCTION.md`](RECONSTRUCTION.md) records how, and what that took.

---

## Quick start

```python
from ure_engine import UREEngine

engine = UREEngine()

assessment = engine.observe({
    "risk_score": 0.95,
    "blocked_rate": 0.85,
    "retry_rate": 0.10,
    "latency_ms": 50,
    "queue_saturation": 0.10,
    "signatures": ["probe-7f3a"],                       # opaque, for AMX
    "markers": ["ignore_previous", "developer_override"],  # opaque, for BVE
})

assessment.regime                # SystemRegime.ATTACKED
assessment.resilience_index      # 0.45
assessment.recommended_action    # RecoveryAction.ISOLATE
assessment.recommended_threshold # 0.18  (vs 0.81 when nominal)
print(assessment.explanation)
```

See the whole thing work through a five-phase incident:

```bash
python examples/incident_walkthrough.py
```

---

## The central claim

Load and malice look similar in aggregate telemetry and call for **opposite**
responses. Throttling an overloaded system relieves it. Throttling an *attacked*
system delivers exactly the denial of service the attacker was trying to cause.

A scalar risk score cannot make that distinction. The regime model is what
earns its own complexity:

| Scenario | Energy | Regime | Recommendation |
|---|---|---|---|
| Legitimate traffic surge | 0.96 | `STRESSED` | `THROTTLE` (shed load) |
| Probing campaign | 1.16 | `ATTACKED` | `ISOLATE` (narrow *who* is served, do not throttle) |
| Attack inducing failures | 1.46 | `CASCADING` | `QUARANTINE` (stop admission, preserve evidence) |

Comparable energy. Different answers. That is the whole point.

---

## Architecture

```
AEGIS
 ├── GSA        Governance Gateway
 ├── URE        Universal Resilience Engine   ← this package
 ├── LTE        Lyapunov Trajectory Engine
 ├── AMX        Attack Memory Exchange
 ├── BVE        Behavioral Vaccine Engine
 ├── FORTRESS   Recovery Authority
 ├── WS3        Universal Telemetry Layer
 └── Executive Dashboard
```

The pipeline, per observation:

```
telemetry
   ↓  TelemetryAdapter          project onto six normalized pressures
   ↓  AMX.recall()              does memory recognize this?
   ↓  BVE.immunize()            does it match a learned escalation?
   ↓  LTE                       V(x) and dV/dt
   ↓  TrajectoryEngine          velocity, acceleration, volatility
   ↓  RegimeClassifier          soft classification + confidence + entropy
   ↓  ResilienceIndex           the single executive metric
   ↓  RecoveryPlanner           what to do, with hysteresis
   ↓  AdaptiveThresholdController
UREAssessment
```

**AMX and BVE run before classification, not after.** Their output raises
`adversarial_pressure`, which changes the energy the classifier sees. Running
them afterwards would let memory annotate an assessment but never alter one,
which is precisely how they became dead code in the predecessor build. Here
they are upstream of the decision, or they are not shipped.

Full design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## The six pressures

Everything URE reasons about is projected onto six normalized `[0, 1]`
dimensions. That projection is the only thing the engine consumes, which is
what lets the same engine sit behind an LLM gateway, a payments API, or a
control link, swap the adapter, keep the engine.

| Pressure | Meaning |
|---|---|
| `threat_pressure` | Hostile intent observed **now**: policy scores, block rates |
| `latency_pressure` | Time-domain strain, absolute latency plus its volatility |
| `failure_pressure` | Things breaking, retries, open circuits, errors |
| `drift_pressure` | Divergence from the expected operating distribution |
| `adversarial_pressure` | Hostile capability accumulated **over time**: AMX/BVE memory |
| `resource_pressure` | Substrate exhaustion, queue saturation, degraded health |

`threat` and `adversarial` are separate on purpose: one nasty request is
threat; a recognized recurring campaign is adversarial. The energy function
weights the second more heavily.

---

## Regimes

| Regime | Meaning | Response |
|---|---|---|
| `NOMINAL` | Low energy, quiet trajectory | none |
| `ADAPTING` | Mid-band, in motion, absorbing it | none |
| `RECOVERING` | Energy dissipating | `RESTORE` (do not intervene) |
| `STRESSED` | Load, not malice | `THROTTLE` / `DEGRADE` |
| `ATTACKED` | Threat or adversarial pressure dominates | `ISOLATE` |
| `CASCADING` | Failures inducing failures | `QUARANTINE` |
| `UNKNOWN` | Could not classify, treated as a **fault** | `DEGRADE` + page |

`UNKNOWN` ranks above `STRESSED` in severity. An engine that cannot classify
its own state is not in a mild condition.

---

## What was fixed from the predecessor

Three defects are documented in the archive and carry regression tests here so
they cannot return.

**1. Energy was never zero.** The original computed `V(x) = 1.5 · sigmoid(Σp²)`.
Since `sigmoid(0) = 0.5`, an idle system with zero pressure scored **0.75**
energy, above the `NOMINAL` band. `NOMINAL` and `RECOVERING` were
*mathematically unreachable* and a healthy idle gateway reported as stressed
forever. Now `V(x) = 1.5 · tanh(Σp²/2)`: exactly 0 at rest, same saturation.
Guarded by `test_zero_state_has_exactly_zero_energy` and a full state-space
reachability sweep.

**2. Rejections were under-weighted.** A sustained 60% policy rejection rate
never moved the regime off `NOMINAL`, because `reject_rate` fed only the
0.3-weighted block-rate term. It now drives threat pressure directly, because from a
gateway's point of view a surge of rejections *is* the threat signal. Guarded
by `test_sustained_sixty_percent_rejection_is_not_nominal`.

**3. AMX, BVE and the recovery planner were dead.** They were instantiated and
never called. The predecessor deleted them, correctly: a dead subsystem is
worse than an absent one. They are restored here **wired**: AMX and BVE feed
`adversarial_pressure` upstream of classification, and the recovery planner's
output gates the engine's own learning. Guarded by `TestMemoryIsActuallyWired`.

Two further defects were found by the tests written for this build:

**4. NaN energy reported `NOMINAL`.** `clamp()` maps NaN to its floor: right
for a pressure, catastrophic for energy, because a broken computation
normalized to 0.0 and the engine confidently reported health. Non-finite
inputs now return `UNKNOWN`.

**5. Hysteresis could never release.** The planner recorded *held* actions as
history, so every hold re-asserted itself as evidence for continuing to hold.
A system that entered `QUARANTINE` could never leave it. It now records what
conditions *proposed*, separately from what is applied.

---

## Legacy GSA integration

The import GSA was written against resolves, with the original call signature:

```python
from ure_engine import ClassificationResult, GatewayHealthEngine

health = GatewayHealthEngine(backlog_capacity=256)
summary = health.assess(reject_rate=0.6, retry_rate=0.1, latency_ms=50, backlog=20)

summary.result_status.value        # "RISK_INCREASING"
summary.dominant_regime.value      # "ATTACKED"
summary.composite_risk_score       # 0.26
summary.calculated_energy          # 0.385
summary.regime_entropy             # a real measurement now, not a placeholder
summary.regime_confidence
summary.rationale_statement
```

The full v2 pipeline runs underneath, trajectory analysis, AMX, BVE, adaptive
thresholds, the legacy interface simply cannot *see* it. When you're ready to
migrate, `summary.assessment` is the modern `UREAssessment`, and
`health.engine` is the `UREEngine` itself.

The compatibility layer will report a condition as **worse** than the engine
does, never calmer. A compatibility shim must not be a place where warnings go
to be lost.

---

## The policy seam

URE never knows a policy's type, implementation, domain, rules, or internals.
It consumes gateway telemetry and opaque signatures. Policy outcomes reach it
only through `PolicyTelemetry`: a score, a verdict shape, and a hash:

```python
engine.observe_policy_signal(PolicyTelemetry(
    policy_name="FinancialPIIPolicy",
    policy_version="1.2",
    input_score=0.9,
    hard_block=True,
    output_acceptable=False,
    timestamp=time.time(),
    signature="req-hash-abc",   # opaque; URE cannot reverse it
))
```

This invariant is enforced mechanically, `test_ure_does_not_import_policy_modules`
fails if anything in the package ever imports a policy module.

And absolute rules stay absolute: a policy `hard_block` bypasses the adaptive
threshold entirely. No system state makes a raw SSN acceptable.

---

## Install and develop

```bash
pip install -e ".[dev]"

pytest                              # 278 tests
python examples/incident_walkthrough.py
```

No runtime dependencies. URE is pure-stdlib arithmetic over bounded windows, so
it drops into a gateway's hot path without dragging a dependency tree into a
component whose entire job is to keep working when other things break.

---

## Design notes worth knowing

**Resilience grows with experience.** A cold engine scores ~0.74, not 1.00, because it
has no vaccines and no memory, and a system that has never met an adversary is
genuinely less resilient than one that recognizes them. The index climbs as BVE
acquires vaccines and AMX builds memory.

**Recovery recommends; it does not act.** URE detects instability, FORTRESS
decides strategy. An engine that can both declare an emergency and act on it
can isolate a system on the strength of its own misclassification with nothing
in the loop to disagree.

**Learning stays on under attack.** That is when the most valuable adversarial
signal exists. Poisoning is defended against inside BVE, by the discrimination
threshold at synthesis and the effectiveness feedback loop, not by going
blind. Learning stops only during `CASCADING`, where an engine would otherwise
learn its own collapse.

**Vaccines wane.** A pattern that coincided with three incidents and has been
wrong ever since is actively retired, not merely outvoted. This is what keeps
the engine from accumulating superstitions.

---

## Documents

| | |
|---|---|
| [`RECONSTRUCTION.md`](RECONSTRUCTION.md) | How this repository was rebuilt, and why it was possible |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The full design, decision by decision |
| [`docs/PROVENANCE.md`](docs/PROVENANCE.md) | Source-by-source lineage, with evidence classes |
| [`docs/AUDIT.md`](docs/AUDIT.md) | ghost_buster integrity scan: 0 MAJOR, 0 surviving mutants, and what is baselined |
| [`docs/INTEGRATION_SENTINEL.md`](docs/INTEGRATION_SENTINEL.md) | Proposal for wiring URE to sentinel_os: adapter, tuned constants, staged rollout, red-team caveats |

---

## License

Proprietary. Copyright (c) 2026 William N. King. All rights reserved. See [LICENSE](LICENSE).
