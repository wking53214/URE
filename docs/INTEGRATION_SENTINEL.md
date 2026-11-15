# Proposal: connect URE to sentinel_os

**Status:** proposal, not a decision. Nothing here has run against production
traffic.
**Audience:** Fable, who owns whether this happens and on what schedule.
**Evidence:** simulation only, none of it observed. The red-team and engine
results are reproducible from this repository (section 8). The tuning table in
section 4 is **not**: it came from a simulated load generator that is not in
the repository, and the scripts that produced it did not survive.

---

## 1. What this is

sentinel_os measures two things and draws no conclusion from either.
`CircuitBreaker.snapshot()` knows what is being rejected and what is failing.
`TransmissionQueue.stats()` knows what is backing up and how stale it has
become. Both are counters on a dashboard. Nobody is watching them at three in
the morning, and if somebody were, the counters would not tell them whether the
system is busy or under attack, which are the two conditions that call for
opposite responses.

URE is the missing half. It takes those signals, projects them onto a bounded
state space, computes a Lyapunov energy over it, and answers a question the
counters cannot: *what kind of trouble is this*. Load, or malice, or a cascade.
Then it recommends an action from a ladder, and the ladder distinguishes the
cases the counters conflate.

The one sentence that justifies the whole integration:

> **A STRESSED system should be throttled. An ATTACKED system must not be.**
> Throttling an attacked system finishes the attacker's denial of service for
> them, using your own defences.

sentinel_os's circuit breaker cannot make that distinction, because a breaker
has one response to every kind of trouble. That is not a criticism of the
breaker; it is the reason to put something in front of it that can tell the
difference.

## 2. Why it is low risk to try

**URE recommends; it does not act.** Nothing in this proposal gives URE the
ability to open a breaker, drop a message, or reject a request. It returns an
assessment. sentinel_os decides what to do with it, including nothing.

**No dependency in either direction.** URE does not import sentinel_os and
sentinel_os does not import URE. The contract is the shape of two mappings,
which sentinel_os already produces and would produce anyway. The poller is the
one piece of code that touches both, so where it lives (inside sentinel_os, or
beside it) is the decision that determines whether that stays true. I have not
read sentinel_os's dependency policy beyond its README.

**No new runtime dependencies.** URE's engine is pure standard library.

**Shadow mode is the entire first stage.** Stage 1 below runs URE alongside the
existing system with its recommendations logged and ignored. If the answer is
"this was not worth it", the cost of finding out is one deleted poller.

## 3. The adapter

Shipped as `ure_engine/sentinel_adapter.py`, tested in
`tests/test_sentinel_adapter.py`. Usage:

```python
from ure_engine import SentinelAdapter, SentinelConfig, UREConfig, UREEngine

adapter = SentinelAdapter(SentinelConfig(
    queue_capacity=128,      # ready-queue depth that means "full"
    latency_ceiling_s=60.0,  # oldest_pending_age_s that means "full"
    dlq_rate_ceiling=20.0,   # DLQ arrivals/min that mean "full"
    overdue_ceiling=20,      # processing_overdue that means "reaper is behind"
))
engine = UREEngine(UREConfig.for_sentinel_os())   # a preset, not new defaults
engine.import_dwell(saved)   # optional, from the last run's engine.export_dwell()

# once per poll, one adapter and one engine per monitored breaker.
# Do not pass now=: timestamps default to wall-clock, and the CNS gate judges
# staleness against wall-clock (section 5, Stage 3).
assessment = engine.observe(adapter.to_frame(
    breaker.snapshot(), queue.stats(),
    dlq_per_minute=dlq_rate,
    injection_hits=policy_hit_rate,   # 0.0 if there is no policy layer here
))
```

`UREConfig.for_sentinel_os()` carries the tuned values as a named preset
rather than changing URE's defaults. They were fitted to simulated traffic whose
shapes I guessed, so making them everyone's default would put simulation-fitted
numbers into behaviour nobody chose. A deployment that has measured something
else passes it as an override.

### Poll cadence, restarts, and what the engine remembers

**Dwell's time constants are in seconds, not observations.** They were written
and first tested at one second per observation, which hid that. The
false-positive guarantee holds at every cadence tested (1, 5, 30 and 60 seconds:
a deployment whose normal rate is 25% never alarms). Detection speed is the
part that moves. A sustained 30% attack against a quiet baseline is called after
about 13 seconds at 1s polls and about 60 seconds at 30s polls, which is two
observations: slow polling detects in fewer observations, not more, and has a
floor of about two. Every "13 observations" figure in this repository is at one
second per observation.

Pick the cadence against the CNS gate's `max_age` (default 60s) with room to
spare. I would not poll slower than a quarter of it. That is a judgement, not a
measurement.

**Restarts.** The learned baseline is state, and losing it costs the warmup with
detection off. Worse, an engine that comes up during an attack learns the attack
as normal, so an adversary who can crash the engine gets a free calibration
window. Persist `engine.export_dwell()` on shutdown and `engine.import_dwell()`
on startup. The imported record is untrusted: it is clamped to the engine's own
ceiling, so a forged or stale baseline can hide traffic below ceiling plus
margin and never above it, and anything malformed is refused whole. After a
confirmed incident, do the opposite: `engine.reset()` and discard the saved
state, since that baseline is the one the attack may have shaped.

`assessment.hostile_baseline` shows what a call was judged against, next to
`assessment.hostile_dwell`.

### Three things that were wrong when this was first built

Each produced a precisely computed, confidently wrong assessment out of a
correct engine. This is the characteristic way an adapter fails, and it is why
the adapter has its own tests rather than being three lines inside a poller.

**Cumulative counters were averaged instead of differentiated.** sentinel_os
counters are lifetime totals. `total_failures / total_calls` is a lifetime
average, and a lifetime average converges and then stops responding: after a
million healthy calls, a thousand consecutive failures barely move it. The
system is visibly on fire and the number is fine. The adapter holds the
previous sample and differentiates. **This is the single most important line
in the file.**

**Latency units were off by 30x.** URE's `TelemetryAdapter` treats 1000ms as
full latency pressure, correct for the LLM gateway it was written for.
`oldest_pending_age_s` is queue staleness in seconds to tens of seconds.
Passed through raw, URE saturates at about one second of queue age, and a 7%
load ramp reads STRESSED. Every ceiling on `SentinelConfig` exists to fix a
mismatch of this kind, and **a deployment that inherits the defaults without
checking them against its own queue will get assessments computed precisely
from the wrong units.**

**Rejections were routed to the wrong pressure.** URE weights threat as
`risk_score * 0.7 + blocked_rate * 0.3`, which assumes `risk_score` is a
per-request policy score that spikes hard. sentinel_os has no such score at
this surface; it has a rejection rate. Routing rejections only into
`blocked_rate` gives them a 0.3 weight, and a sustained 30% rejection rate
reads NOMINAL. From a gateway's point of view a sustained rejection rate *is*
the threat signal, so it drives `risk_score` directly.

## 4. Where the tuned constants came from

Coordinate descent over six labelled scenarios (quiet, load_surge, attack,
cascade, recovery, flapping) at seven intensities each, ranked on the **worst
cell** rather than the mean. A configuration that is excellent at nominal load
and blind during a cascade is worse than one that is merely adequate
everywhere, and ranking on the mean hides exactly that.

| | mean | worst cell |
|---|---|---|
| URE defaults | 94.9% | 64.3% |
| Tuned | 98.3% | 85.7% |
| Held-out scenarios | 97.3% | |

`slope_window` dominates: 3 gives a 43% worst cell, 12 gives 79%. If only one
constant is carried over from this document, carry that one.

**None of this table can be reproduced from this repository.** The load
generator and tuner that produced it were scratch code and did not survive. The
scenarios are described above well enough to rebuild, but a rebuild would not
reproduce these numbers exactly.

**These are not universal.** They are fitted to simulated traffic with the
shapes I guessed sentinel_os produces. Stage 1 exists to replace the guess.

## 5. Staged rollout

Each stage has an exit criterion that can fail. A stage with no failing exit
criterion is a stage that always passes, which is not a gate.

### Stage 1, shadow (2 weeks)

Poll `snapshot()` and `stats()` on the existing interval, run URE, log the
assessment, change nothing. Sample the raw inputs too, not only the verdicts:
the point is to replace the simulated traffic shapes with real ones.

**Exit criteria**
- Fewer than 1 ATTACKED or CASCADING call per week during periods
  independently known to be healthy. This is the false-positive gate and the
  one most likely to fail.
- Every incident in the window has a URE assessment that a human agrees with
  in hindsight.
- Recompute the `SentinelConfig` ceilings from observed p99 queue depth, age,
  and DLQ rate. **If the observed scales differ from the defaults by more than
  about 2x, stop and re-tune before Stage 2.**

**Abort if** the false-positive rate exceeds the gate, or the observed traffic
shapes look nothing like the simulated ones. Both mean the tuning is fitted to
fiction.

### Stage 2, advisory (2 to 4 weeks)

Surface the regime and recommended action to operators. Still no automated
control. Page on CASCADING only.

**Exit criteria**
- Operators report the regime label told them something the existing dashboard
  did not, on at least two real events.
- No page that a human judged spurious.

**Abort if** operators start ignoring it. An advisory signal that gets ignored
does not become trustworthy by being given control.

### Stage 3, one control (4 weeks minimum)

Enable exactly one automated behaviour: **ISOLATE on ATTACKED**, which is the
distinction the whole integration exists for and the one sentinel_os cannot
currently make. Leave THROTTLE, DEGRADE, and QUARANTINE advisory.

Ship the kill switch in the same change, not after it.

**Route it through the CNS gate, and do not act on the gate alone.** URE ships
`UREGate` (`ure_engine/cns_connector.py`), which restates an assessment as a
`cns.gate.GateResult`. sentinel_os already requires `cns.gate` (its README pins
v4.0.0, the same tag URE pins), so the vocabulary is shared. The mapping:

| URE decision | CNS outcome | Stage 3 |
|---|---|---|
| ALLOW | PASS | nothing |
| THROTTLE, DEGRADE, REVIEW | RETRY | advisory only |
| ISOLATE, QUARANTINE, DENY | TERMINAL_BREACH | ISOLATE acts; the rest stay advisory |

URE still only recommends. The gate restates the recommendation and grants no
authorization, and the consumer decides what to do with it.

The connector adds a hazard this proposal had not anticipated. **An assessment
older than `max_age`, dated in the future, or with a non-finite timestamp is a
`TERMINAL_BREACH`.** That is the right default for a gate, and it makes the
poller's liveness a safety dependency: a stalled or crashed poller, or clock
skew between hosts, produces the same outcome as an attack. Acting on
`TERMINAL_BREACH` alone would turn a poller outage into an automated isolation.

So the consumer must check the assessment itself, not only the gate: act only
when `assessment.regime` is ATTACKED and the assessment is fresh, and treat a
stale or future-dated one as "URE is unavailable", which pages a human and
isolates nothing. The gate's reason string says why it breached, but matching on
prose is fragile and I would not build on it.

**Exit criteria**
- Every automated isolation is reviewed within 24 hours and judged correct.
- Time to mitigation improves measurably against the Stage 2 baseline.
- Killing the poller produces an alert and no isolation. Tested, not assumed.

**Abort if** any automated isolation is judged wrong. One wrong isolation in a
population of correct ones is still a self-inflicted outage, and the review
should treat it that way.

### Stage 4, the rest of the ladder

Only after Stage 3 has held through at least one real incident. Not before.

## 6. What this does not fix, and what it costs

**URE cannot see what sentinel_os does not measure.** `drift_metric` is
hard-coded to 0.0 because sentinel_os exposes no distribution signal at this
surface. Reporting a guess as a measurement is worse than reporting none, so
one of URE's six pressures is simply dark in this integration. If drift
matters, that is a sentinel_os change, not a URE one.

**One adapter and one engine per monitored breaker.** The adapter is stateful:
it remembers the previous counter sample. Share one across scopes and the
differences are taken between unrelated systems.

**Cost is small but not zero.** One poll per interval, order 100µs of
computation, bounded memory from the ring buffers. It is not free and it is not
worth arguing about.

## 7. Red-team caveats

URE has been attacked deliberately. `tests/test_adversarial.py` holds the
attacks and now passes with none marked expected-to-fail. Where an attack
cannot be closed outright, the tests assert what it *costs* an adversary, so a
change that makes one cheap fails in CI rather than in production. Read this
section before Stage 3, not after.

**Closed.** The sub-threshold slow boil. An adversary holding a constant
hostile rate below the classifier's boundary used to be invisible at *any*
duration, measured at a knife edge of 36.2%. `ure_engine/dwell.py` integrates
exposure instead of differentiating it.

It measures excess over **this deployment's own learned normal**, not an
absolute level. That distinction is load-bearing for you rather than academic:
an earlier version used a fixed floor, and a deployment whose ordinary
rejection rate was 25% was classified ATTACKED after 25 observations with a
recommended action of ISOLATE, with nothing wrong. At Stage 3 that is an outage
caused by the defence. The learned baseline removes it by construction, and the
result is better than the original engine on both axes at once: no steady rate
below 39% alarms (the original alarmed from 36.2%), and a 30% rate held against
a quiet baseline is detected in 13 observations instead of never.

Two consequences for your rollout:

* **Stage 1 is also the calibration period.** URE needs to see your normal
  before it can judge a departure from it, and the first 32 observations are
  spent acquiring that with detection disabled.
* **An engine started during an attack learns the attack as normal**, up to
  `dwell_baseline_ceiling`. It self-heals on the first genuine lull, but after
  any confirmed incident the right move is `engine.reset()` to re-baseline
  deliberately. `assessment.hostile_dwell` and `assessment.hostile_baseline` show
  what a call was judged against. (An earlier revision of this document said
  the baseline was exposed when it was not.)

**Also closed, both memory-poisoning attacks.**

*AMX decoy flooding.* Attack memory evicted weakest-first on severity, and
severity is whatever the caller reported, so an adversary minting signatures at
0.99 evicted a real profile at 0.95. Eviction now ranks on corroboration
(recurrence across elapsed time, which must be spent rather than claimed) and
capacity is partitioned so signatures seen once can evict nothing but each
other. 20,000 minted signatures at maximum severity no longer displace one real
profile.

An adversary who pays for corroborated decoys can still displace memory, and
that half is detected rather than prevented: replacing a full memory takes
thousands of adverse events in minutes, and `AttackMemoryExchange.flood_pressure`
reports the fraction of the established (corroborated) memory recently pushed
out, which feeds adversarial pressure. The attack converts into a detection.
(The first version alarmed on a memory that was "full and mostly new". That was
a lever, not a defence, and is described in the next section.)

*BVE poisoning.* Benign evidence was counted from the same bounded ring buffer
as adverse evidence, so flooding adverse episodes flushed the evidence that
blocked synthesis, and URE vaccinated against ordinary traffic. Benign evidence
now lives in a durable, frequency-weighted ledger, and a pattern known to occur
in legitimate traffic is never vaccinated against at any ratio.

The residual is priced rather than eliminated: burying evidence seen N times
costs roughly `4096 * (N + 1)` requests that **succeed**, since only non-adverse
outcomes write to that ledger. An adversary who can push tens of thousands of
successful requests does not need this attack.

**What this changes for your rollout.** The Stage 3 caveat about leaving
`enable_learning=False` until Stage 4 was written when both of these were open.
It is now a judgement call rather than a requirement. The conservative reading
still holds, and the red-team pass above is the reason: it found that memory,
the part learning feeds, was the cheapest lever in the engine. Those holes are
closed and tested, but they were found by attacking the fixes, which is
evidence that more may be there. Learning is the part of URE under direct
attacker influence, and Stage 3 has enough to prove without it.

### Red-teaming the fixes

The fixes above were attacked in turn, and the pass found problems in them and
one that predates them. Stated in order of severity, with what was measured.

**Memory was a lever, and cheap to pull. This predates my changes.** Ambient
pressure, AMX's background contribution to adversarial pressure, was documented
as colouring an assessment rather than dominating it. Its cap of 0.5 sat exactly
where ATTACKED outvotes NOMINAL on adversarial pressure alone (measured
crossover 0.49 with every other pressure healthy). Eight unique rejected
requests held a healthy deployment in a restrictive state for **about seven
hours** after the last one. At Stage 3 that is an automated isolation bought
for eight requests. The cap is now 0.40, and the property is pinned by
behaviour, not by the constant: after any number of remembered signatures,
restriction ends within about two minutes of the attacker stopping.

My first red team marked "hysteresis lock" as held. It was, for what it tested
(maxed telemetry, no signatures), but it never exercised the path where a
remembered attack feeds back into pressure. That was a blind spot, not a
defence.

**My dwell fix made it worse.** Dwell was fed the state after AMX's ambient
echo had been folded in, so a memory of an attack counted as ongoing exposure.
One rejected request held ISOLATE for about 86 minutes. Dwell now sees only what
was observed now. After the fix the same request restricts nothing.

**My first flood detector was a lever.** It fired on a memory that was full and
mostly new, but churning one-off signatures harms nothing, since the partition
guarantees they can evict only each other. Activity anyone can generate for the
price of a few rejected requests was feeding adversarial pressure. It now fires
on harm instead: corroborated memory being displaced.

**Dwell timing is in seconds.** Covered above under poll cadence.

**Not established.** Feeding URE one observation per request, with 10% of them
hostile, holds ISOLATE. It does so identically with the flood term disabled, so
it is neither of the above, and I did not investigate whether it is right. It
does not apply to the polled, aggregated frames this adapter produces. It does
apply to a per-request integration such as a gateway, so check it before
reusing the engine that way.

**An architectural limit, not a defect.** Once every pressure clamps at 1.0,
further escalation is invisible: a 10x increase produces a byte-identical state
vector, because the information is destroyed at normalization, before the
Lyapunov function is evaluated. Bounded pressures are what make `V(x)` a
Lyapunov function at all, so this is not fixable without giving up the
guarantee. The mitigating facts are that the recommended *action* stays
restrictive throughout, and that `assessment.hostile_dwell` keeps climbing when
energy, derivative and volatility have all gone flat. **If you are building a
"still getting worse" alert on a saturated system, build it on
`hostile_dwell`.** Nothing else moves.

## 8. Reproducing every number in this document

```bash
pip install -e .
pytest                                        # engine, adapter and connector
pytest tests/test_adversarial.py -v           # the attacks, and what each costs
pytest tests/test_dwell_persistence.py -v     # baseline persistence and the preset
python tools/integrity_gate.py . --mutate     # the integrity gate
```

The tuning sweep ran against a simulated sentinel_os load generator that is not
in this repository. It was not shipped on purpose, because a scenario generator
built from my guesses about sentinel_os traffic is not evidence and should not
be shipped looking like it is, and its scripts have since been lost. The tuned
constants it produced are recorded in section 4 and the scenarios are described
there, but the tuning table itself is not reproducible from here. **Stage 1 replaces them
with measurements, which is the point of Stage 1.**

---

## Recommendation

Run Stage 1. It is a poller and a log line, it cannot affect production, and it
answers the only question that actually matters: whether the simulated traffic
shapes this was tuned against resemble the real ones. Everything after Stage 1
should be decided on that data rather than on this document.
