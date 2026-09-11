# URE Architecture

**SYS-URE-001 · ARCH-URE-001 · v2.0**

---

## 1. Position in AEGIS

```
AEGIS
 ├── GSA        Governance Gateway          policy enforcement, attestation, audit
 ├── URE        Universal Resilience Engine system state authority        ← here
 ├── LTE        Lyapunov Trajectory Engine  V(x), dV/dt                   (in URE)
 ├── AMX        Attack Memory Exchange      historical awareness          (in URE)
 ├── BVE        Behavioral Vaccine Engine   acquired immunity             (in URE)
 ├── FORTRESS   Recovery Authority          executes recovery strategy
 ├── WS3        Universal Telemetry Layer   events, metrics, anomalies
 └── Executive Dashboard
```

LTE, AMX and BVE are named as peers in the AEGIS hierarchy but ship inside this
package. They are modules, not services: each is an independent, separately
testable component with no I/O, and the whole pipeline is arithmetic over
bounded windows. Splitting them across a network boundary would add failure
modes to a component whose job is to keep working when things fail.

### Direction of dependency

```
GSA  ──imports──▶  URE
URE  ──────────▶  (nothing)
```

URE has no import path to GSA, to any policy module, or to any third-party
runtime package. GSA depends on the `GatewayHealthEngine` **protocol**, not on
URE's concrete class, so neither side needs the other installed to typecheck.

---

## 2. The governance flow

```
Inbound Request
     ↓
PolicyInputGate
     ↓
Policy.assess_input()                    ← the policy seam; URE sees none of this
     ↓
GSA Gateway
     ↓
URE.observe()
     ├── State Vector Update
     ├── AMX Correlation
     ├── BVE Activation
     ├── LTE Energy Calculation
     ├── Trajectory Update
     ├── Regime Classification
     ├── Resilience Index Computation
     ├── Recovery Recommendation
     └── Adaptive Threshold
     ↓
UREAssessment
     ↓
GSA Decision Layer
     ALLOW · REVIEW · THROTTLE · DEGRADE · ISOLATE · QUARANTINE · DENY
     ↓
Policy.inspect_output()
     ↓
Response Pipeline → Attestation → Audit Ledger
```

### Why AMX and BVE run before classification

This ordering is load-bearing. AMX recall and BVE immunization both raise
`adversarial_pressure`, which changes the energy the classifier sees and
therefore the regime it assigns.

If they ran *after* classification, memory and learning could annotate an
assessment but never alter one. That is exactly how they became dead code in
the predecessor build: instantiated, never called, correctly deleted. A
subsystem that cannot change an outcome is decoration.

---

## 3. State space

Six normalized `[0, 1]` pressures. Everything else is derived.

| Dimension | Source signals | Energy weight |
|---|---|---|
| `threat_pressure` | policy risk score, block rate | 1.4 |
| `latency_pressure` | latency vs ceiling, latency volatility | 0.8 |
| `failure_pressure` | retry rate, error rate, circuit state | 1.3 |
| `drift_pressure` | host-reported distribution drift | 0.7 |
| `adversarial_pressure` | explicit events + **AMX/BVE memory** | 1.5 |
| `resource_pressure` | queue saturation, substrate health | 0.9 |

Weights are security-biased: a compromised system is worse than a slow one.
`test_security_dimensions_weigh_more_than_operational_ones` asserts this so a
weight edit is a visible decision rather than a silent drift.

### The units problem this solves

The predecessor's `SystemMetricsTelemetry` mixed rates in `[0, 1]`
(`dropout_rate`) with unbounded counts (`backlog_depth`, 0..N) in a single
weighted sum. The count dimension dominated at arbitrary scale, and the weights
had to be re-tuned per deployment. Normalizing every dimension up front makes
the weights portable; a deployment with different capacity changes a
constructor argument, not the physics.

---

## 4. Lyapunov energy

```
V(x) = 1.5 · tanh( Σᵢ wᵢ·pᵢ² / 2 )
```

Properties, all asserted by tests:

- **`V(0) = 0` exactly.** See §8.
- **Monotone** in every dimension.
- **Bounded** by `ENERGY_MAX = 1.5`, approached but never reached.
- **Convex**: squaring means one severe pressure outweighs several moderate
  ones of the same total. That bias is what makes the engine notice
  concentrated attacks rather than averaging them into background load.

### dV/dt

The derivative is a **least-squares slope over the recent window**, not a
two-point difference. A backward difference is dominated by sampling jitter at
the millisecond timescales a gateway operates on, and "rising" has to mean
something stable enough to gate a regime transition on.

Timestamps are re-based at the window start before the regression. Epoch
seconds are ~1.7 × 10⁹; squaring them directly destroys the covariance term in
float64. `test_uses_epoch_scale_timestamps_without_precision_loss` guards it.

### Early warning

`time_to_threshold()` projects forward at the current slope, turning "energy is
rising" into "about 40 seconds of headroom", which is the form a recovery
planner can act on.

---

## 5. Trajectory

Separate from LTE on purpose. LTE produces the smoothed slope used for regime
*decisions*; the trajectory engine produces raw finite-difference motion used
for *explanation* and for oscillation detection. The predecessor computed trend
twice in two code paths that could disagree.

| Signal | Question it answers |
|---|---|
| velocity | how fast |
| acceleration | is it speeding up (a ramping attack) |
| volatility | how erratic |
| oscillations | how many direction reversals |

**Why oscillation matters.** A system flapping between 0.55 and 0.75 energy has
a mean slope of roughly zero. Any derivative-only check calls it stable, while
an operator watching the graph would immediately call it unhealthy. Flat steps
are skipped when counting reversals; otherwise an idle system producing
identical energies registers a reversal on every sample and reads as violently
unstable.

---

## 6. Regime classification

The architecture is explicit that classification should use "Lyapunov energy,
Lyapunov derivative, historical memory, and trajectory features, **not static
thresholds**". The predecessor's cascade of `if energy >= 0.60` branches was a
stopgap: correct at its calibration points, arbitrary between them, and
impossible to explain to an operator standing at a boundary.

URE scores a continuous **affinity** for each regime and normalizes:

```
NOMINAL     = (1 − e)³ · (1 − vol)
ADAPTING    = bump(e; 0.45, 0.25) · (0.35 + 0.65·vol) · (1 − hostile)
RECOVERING  = falling · (0.30 + 0.70·e) · (1 − 0.8·hostile)
STRESSED    = e · operational · (1 − 0.6·hostile) · 1.2
ATTACKED    = hostile^1.5 · (0.40 + 0.60·(1 − falling)) · (1 + 0.5·recall) · 1.6
CASCADING   = gate(e, 0.65)² · rising · (0.30 + 0.70·breaking) · 3.6
```

where `e` is normalized energy, `hostile = max(threat, adversarial)`,
`operational = max(latency, failure, resource)`, and `breaking = max(failure,
resource)`.

This buys three things the cascade could not:

1. **Confidence**: the winner's share of probability mass, so a 0.34/0.33/0.33
   split is visibly a coin flip rather than a confident answer.
2. **Entropy**: a real measure of classifier ambiguity. The legacy GSA
   interface already had a field for this that the old engine could only fill
   with a hand-wave.
3. **Explanation**: the runner-up and the margin, which is what an operator
   actually wants at a boundary.

### Two deliberate asymmetries

**`recall` enters ATTACKED multiplicatively.** AMX memory can sharpen a
hostile reading but cannot manufacture one from nothing, so a false memory
match on a quiet system does not invent an attack.
`test_recall_sharpens_attacked_but_cannot_invent_it` guards this.

**CASCADING is *gated*, not weighted.** Below 0.65 normalized energy
(≈ 0.98 absolute, matching the predecessor's calibrated `energy >= 1.0` band)
it carries **zero** evidence. A steep legitimate load ramp looks identical to a
cascade in every other respect, rising energy, high failure and resource
pressure, and calling it CASCADING quarantines a system that is only busy.
Requiring genuine proximity to collapse is what separates the two. This was
caught by running `examples/incident_walkthrough.py` and watching a traffic
surge get quarantined.

### Reachability

Every regime must be reachable somewhere in the state space, and UNKNOWN must
never appear for well-formed input. `test_full_state_space_sweep` sweeps 5⁶
states × 5 derivatives × 3 volatilities and asserts both. This sweep is what
originally exposed the uncentered-energy bug, so it is carried forward
verbatim.

### UNKNOWN is a fault

Returned only for non-finite input. It ranks **above STRESSED** in severity and
triggers `DEGRADE` plus an operator page. An engine that cannot classify its
own state is not in a mild condition, and it must never fail toward "everything
is fine".

---

## 7. Memory and learning

### AMX, Attack Memory Exchange

Signature → `AttackProfile(severity, frequency, success_rate, last_seen,
first_seen, tags)`.

- **Severity takes the max**, not the latest. An attack does not become less
  dangerous because its most recent instance scored lower.
- **Exponential decay**, 6-hour half-life. Pausing to evade a memory window is
  the obvious counter-move; a half-life makes it expensive without making
  memory permanent.
- **Probabilistic union** for aggregate pressure, `1 − Π(1 − sᵢ)`, not a sum.
  A sum saturates at 1.0 after two matches and stops discriminating.
- **Ambient pressure**: a deployment under sustained attack is in a different
  posture than an idle one regardless of what *this* request looks like.
  Capped to the strongest 8 profiles and halved, so a large memory of weak
  signatures cannot manufacture pressure by volume.
- **Exchange**: `export_profiles` / `import_profiles` round-trip so a fleet can
  pool what it has learned. Nothing in a profile is request content.

### BVE, Behavioral Vaccine Engine

An ordered marker sequence that repeatedly preceded a bad outcome becomes a
vaccine. On a later encounter, a *partial* match fires, acting at marker two
of three instead of after marker three. That is the entire value.

Three properties separate this from a blocklist:

- **Acquired**: synthesized from observed incidents, never authored.
- **Graded**: confidence × effectiveness, with partial matches conferring
  partial protection.
- **Waning**: vaccines whose predictions stop coming true lose effectiveness
  and are retired. This is what keeps the engine from accumulating
  superstitions.

Synthesis requires `P(adverse | pattern) ≥ 0.70`. A pattern that appears just
as often in benign traffic predicts nothing, and vaccinating against it would
be pure false positives. Matching is by *subsequence*: an attacker interleaving
benign markers between the steps of a known sequence is still executing that
sequence.

Effectiveness is a Laplace-smoothed success rate, so one early mistake does not
zero out a vaccine and one early success does not certify one.

### When learning stops

Learning stays **on** under `ATTACKED`: that is when the most valuable
adversarial signal exists, and an engine that stops recording exactly when it
is attacked can never become historically aware. Poisoning is defended against
inside BVE (the discrimination threshold and the effectiveness loop), not by
going blind.

Learning stops under `CASCADING`/`QUARANTINE`, where an engine would otherwise
learn its own collapse.

---

## 8. Resilience index

One number, `0.00` collapsed → `1.00` highly resilient.

| Term | Weight | Meaning |
|---|---|---|
| stability | 0.30 | energy headroom |
| trajectory | 0.25 | improving or degrading, and how erratically |
| adaptation | 0.20 | what BVE has learned |
| recovery | 0.15 | how much capability remains unspent |
| memory | 0.10 | AMX depth **minus** active recall |

This is **not** an inverted risk score. Risk asks "how bad is it now".
Resilience asks "how much can this absorb before it stops working", which is a question
about capacity, which depends on trajectory, learning, and how much capability
has already been spent staying upright.

Three consequences worth stating plainly:

- **A cold engine scores ~0.74, not 1.00.** It has no vaccines and no memory.
  A system that has never met an adversary is genuinely less resilient than one
  that recognizes them. The index climbs with experience.
- **Memory cuts both ways.** Deep memory is an asset; active recall is a
  liability. An idle deployment with rich memory scores well; the same
  deployment mid-campaign does not.
- **A regime ceiling applies.** A CASCADING system cannot report above 0.20 no
  matter how the terms average out. A metric that looks reassuring during an
  outage is worse than no metric.

The full `ResilienceBreakdown` is always reported, including `weakest_term`,
computed by weighted contribution, so it points at where the leverage is. A
single number nobody can decompose is a number nobody trusts.

---

## 9. Recovery

```
DEGRADE     shed optional work, keep serving
THROTTLE    reduce admitted rate
ISOLATE     cut the affected scope off from shared resources
QUARANTINE  stop admitting, preserve state for forensics
RESTORE     walk back toward normal
```

**The asymmetry that justifies the regime model: STRESSED throttles, ATTACKED
isolates.** Throttling an overloaded system relieves it. Throttling an attacked
system does the attacker's work: they wanted the service degraded, and the
defence delivered it. Under attack the correct move is to narrow *who* is
served, not *how much* service exists. `reduce_rate_limit` is explicitly
`False` in the ATTACKED vector.

### Hysteresis

Escalation is immediate. De-escalation requires `restore_patience` consecutive
observations agreeing that less intervention is warranted. A control loop that
releases on one good sample re-enters the bad state on the next and oscillates,
and an oscillating loop is worse than either fixed state.

Two implementation details that are easy to get wrong, and were:

- The planner records what conditions **proposed**, not what was **applied**.
  Recording held actions makes the history self-reinforcing, every hold
  re-asserts the elevated action as evidence for continuing to hold, and the
  system can never leave QUARANTINE.
- A held vector carries the **applied** action's controls, not the proposed
  action's. Otherwise an executor receives a vector whose action says
  QUARANTINE and whose controls say resume admission, and it will obey the
  controls.

### URE recommends; it does not act

FORTRESS executes. An engine that can both declare an emergency and act on it
can isolate a system on the strength of its own misclassification with nothing
in the loop to disagree.

---

## 10. Adaptive thresholds

```
threshold = f(regime, stability, attack_pressure, recovery_state)
```

Regime anchors, the first, fifth and sixth are specified by the architecture:

| Regime | Anchor |
|---|---|
| NOMINAL | 0.85 |
| ADAPTING | 0.75 |
| RECOVERING | 0.70 |
| STRESSED | 0.60 |
| UNKNOWN | 0.60 |
| ATTACKED | 0.45 |
| CASCADING | 0.20 |

The ladder falls faster than linearly because the cost of a false negative
grows much faster than the cost of a false positive once a system is in
trouble. Adjustments: low resilience tightens, attack pressure tightens,
recovery relaxes slightly but never above the anchor, relaxing during recovery
is what causes the second spike.

False-discovery-rate feedback is preserved from GSA as a **bounded additive
trim** (±0.08). Feedback can nudge the threshold; it can never override the
regime. A noisy FDR signal during an incident must not relax a threshold the
regime just tightened.

---

## 11. The policy seam

Non-negotiable, and enforced mechanically.

> URE must never know: policy type, policy implementation, policy domain,
> policy rules, policy internals. It consumes only gateway telemetry.

`test_ure_does_not_import_policy_modules` scans every module in the package and
fails if any imports a policy or gateway module. Policy outcomes reach URE only
through `PolicyTelemetry`: a score, a verdict shape, a policy *name*, and an
opaque signature URE cannot reverse.

`governance_adapter.py` is the single point of contact, which is why the
coupling is concentrated there rather than spread across the engine.

### Decision precedence

1. **Policy hard-block**: absolute rules stay absolute. No system state makes
   a raw SSN acceptable, and no resilience reading relaxes it.
2. **System recovery posture**: QUARANTINE/ISOLATE governs regardless of this
   request's score. The system is not in a position to serve it.
3. **Adaptive threshold**: otherwise compare the score against the threshold
   computed for the current regime.

---

## 12. Concurrency, failure, and cost

**Thread safety.** `UREEngine.observe` holds an `RLock` over the mutable
history so concurrent observers cannot interleave into a torn trajectory. AMX
and BVE are independently locked.

**Async.** `evaluate()` satisfies the awaitable `GatewayHealthEngine` protocol
without spawning tasks, there is no I/O to await. It exists so callers on an
event loop are not forced into a thread.

**Failure behaviour.** Partial telemetry degrades to a partial picture. `None`,
strings and NaN coerce to safe defaults. A system that is on fire does not owe
the resilience engine a complete feed. Non-finite *energy*, however, returns
UNKNOWN rather than a default, see §6.

**Cost.** Bounded windows (64 samples), no I/O, no allocation proportional to
traffic. AMX is capacity-bounded with weakest-first eviction; BVE likewise by
potency. Memory is O(capacity), not O(requests).

---

## 13. Module map

| Module | Responsibility | Depends on policy? |
|---|---|---|
| `state_vector.py` | the six pressures, `PolicyTelemetry` | score only |
| `telemetry.py` | raw signals → pressures; smoothing | no |
| `lyapunov.py` | `V(x)`, `dV/dt`, time-to-threshold | no |
| `trajectory.py` | velocity, acceleration, volatility, oscillation | no |
| `regimes.py` | soft classification, confidence, entropy | no |
| `attack_memory.py` | AMX | telemetry only |
| `vaccines.py` | BVE | telemetry only |
| `recovery.py` | action ladder, hysteresis | no |
| `resilience.py` | the executive index | no |
| `thresholds.py` | adaptive threshold controller v2 | no |
| `assessment.py` | the canonical output contract | no |
| `engine.py` | orchestration | no |
| `governance_adapter.py` | GSA ↔ URE translation | **direct** |
| `compat.py` | the legacy interface | no |

That separation is exactly what the continuation architecture requires.
