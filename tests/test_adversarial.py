"""
Adversarial tests: attacks on URE itself.

Every other test in this suite asks whether URE classifies honest traffic
correctly. These ask the opposite question, the one that matters for a security
component: given an adversary who knows URE is watching and knows how it works,
can they defeat it?

URE's defences are themselves attack surface, and several are inviting:

* the adaptive threshold RELAXES during RECOVERING
* hysteresis HOLDS a degraded state for ``restore_patience`` observations
* AMX evicts weakest-first at capacity
* BVE synthesises vaccines from observed patterns, then fires on them
* every pressure clamps at 1.0, destroying magnitude above full scale

Each is a deliberate design choice. Each is also a lever.

HOW THE KNOWN WEAKNESSES ARE ENCODED
------------------------------------
Two of these attacks still succeed: A4 (AMX decoy flooding) and A5b (flushing
benign evidence out of BVE's observation window). A1 is closed by
:mod:`ure_engine.dwell`. A6 turned out to be an architectural limit rather than
a defect, and is pinned as one below with a corrected diagnosis.

The ones that still succeed are marked
``xfail(strict=True)`` rather than deleted or softened, which means:

* the suite stays green while the weakness is known and accepted;
* the moment someone fixes one, the test XPASSes, ``strict=True`` turns that
  into a failure, and the fixer is forced to come here and flip the marker.

That is not hypothetical: it is how A1 came to be un-marked. The fix XPASSed
the test, strict mode failed the build, and the marker had to be removed
deliberately rather than drifting.

A known hole that silently reopens is worse than one nobody wrote down, and an
"expected failure" that quietly starts passing teaches nobody anything. This is
the encoding that makes both loud.
"""

from __future__ import annotations

import pytest

from ure_engine import RecoveryAction, SystemRegime, UREConfig, UREEngine
from ure_engine.vaccines import Observation

BENIGN_REGIMES = frozenset({"NOMINAL", "ADAPTING", "RECOVERING"})
RESTRICTIVE = frozenset({"ISOLATE", "QUARANTINE", "THROTTLE"})

#: The tuned settings from the sentinel_os integration experiment.
TUNED = dict(smoothing_alpha=0.6, slope_window=12,
             history_window=64, restore_patience=5)


def hostile(level: float) -> dict:
    """Telemetry for a sustained hostile rate, with a healthy substrate.

    Deliberately clean everywhere except threat: the adversary is not breaking
    anything, only probing. That is what makes it hard to see.
    """
    return {
        "risk_score": level,
        "blocked_rate": level,
        "retry_rate": 0.02,
        "latency_ms": 20.0,
        "queue_saturation": 0.05,
    }


BENIGN = {"risk_score": 0.0, "blocked_rate": 0.0, "retry_rate": 0.01,
          "latency_ms": 15.0, "queue_saturation": 0.03}

MAXED = {"risk_score": 1.0, "blocked_rate": 1.0, "retry_rate": 1.0,
         "latency_ms": 5000.0, "queue_saturation": 1.0,
         "substrate_health": 0.0, "circuit_state": "OPEN",
         "adversarial_events": 50.0, "drift_metric": 1.0}


# ---------------------------------------------------------------------------
# A1. Sub-threshold slow boil (EVASION)
# ---------------------------------------------------------------------------


def test_sustained_hostile_rate_is_eventually_detected() -> None:
    """A patient adversary at a constant rate should not evade forever.

    CLOSED by :mod:`ure_engine.dwell`. Before the dwell term this failed at
    every duration, because a constant has no derivative and URE had nothing
    but derivatives to reason with.
    """
    engine = UREEngine(UREConfig(**TUNED))
    telemetry = hostile(0.30)
    for step in range(80):
        assessment = engine.observe(telemetry, now=float(step))
    assert assessment.regime.value not in BENIGN_REGIMES, (
        f"30% sustained hostile rate read as {assessment.regime.value} "
        f"after 80 observations"
    )


def test_evasion_boundary_now_costs_the_adversary_time() -> None:
    """Characterises what the fix bought, in the adversary's own currency.

    The old boundary was a knife edge at 36.2% hostile pressure that duration
    could not cross: 20 observations or 200, a rate below it was invisible
    forever. The dwell term replaces that with a floor plus a clock, so the
    surviving evasion band is both much lower and much less useful.

    Measured, so a change to the affinity functions or the dwell constants that
    moves any of it shows up here rather than in production.
    """
    def steps_to_detect(level: float, cap: int = 400) -> int | None:
        engine = UREEngine(UREConfig(**TUNED))
        for step in range(cap):
            a = engine.observe(hostile(level), now=float(step))
            if a.regime.value not in BENIGN_REGIMES:
                return step + 1
        return None

    # Below the dwell floor nothing accumulates, by design: ordinary traffic
    # carries a nonzero block rate and a floor of zero would charge every
    # healthy deployment toward ATTACKED. This is the residual hole, and it is
    # now a documented constant rather than an emergent knife edge.
    assert steps_to_detect(0.15) is None

    # What used to evade indefinitely is now detected, and detection time falls
    # as the rate rises: patience has a price.
    assert steps_to_detect(0.30) is not None
    assert steps_to_detect(0.25) is not None
    slow, fast = steps_to_detect(0.25), steps_to_detect(0.40)
    assert slow is not None and fast is not None and slow > fast

    # The old knife edge is gone: 0.30 was permanently invisible before.
    assert steps_to_detect(0.30, cap=80) is not None


def test_sustained_legitimate_load_does_not_accumulate_dwell() -> None:
    """The false positive the dwell term would cause if it were careless.

    Duration is dangerous evidence. Every real system spends long stretches
    under sustained load, and a dwell term over operational pressure would turn
    every busy afternoon into an incident. This pins the restriction that makes
    the mechanism safe: only hostile pressure charges it.
    """
    heavy = {"risk_score": 0.0, "blocked_rate": 0.0, "retry_rate": 0.45,
             "latency_ms": 850.0, "queue_saturation": 0.80,
             "substrate_health": 0.75}
    engine = UREEngine(UREConfig(**TUNED))
    for step in range(200):
        a = engine.observe(heavy, now=float(step))

    assert a.hostile_dwell == 0.0, (
        f"200 observations of heavy but honest load charged dwell to "
        f"{a.hostile_dwell:.3f}"
    )
    assert a.regime is not SystemRegime.ATTACKED


# ---------------------------------------------------------------------------
# A2. Pulse attack / threshold gaming (EVASION). DEFENCE HOLDS.
# ---------------------------------------------------------------------------


def test_threshold_cannot_be_walked_up_by_inducing_recovery() -> None:
    """The RECOVERING relaxation must not be exploitable.

    The adversary strikes briefly, goes quiet to manufacture a RECOVERING
    reading and its +0.05 threshold relaxation, then strikes again, hoping to
    ratchet the threshold upward across cycles.

    This defence holds because the relaxation is capped below the regime
    anchor, so it cannot accumulate.
    """
    engine = UREEngine(UREConfig(**TUNED))
    thresholds_at_strike = []
    t = 0
    for _ in range(6):
        for _ in range(3):
            a = engine.observe(hostile(0.9), now=float(t))
            t += 1
            thresholds_at_strike.append(a.recommended_threshold)
        for _ in range(9):
            engine.observe(BENIGN, now=float(t))
            t += 1

    assert max(thresholds_at_strike) < 0.60, (
        f"threshold reached {max(thresholds_at_strike):.2f} during a strike; "
        f"the recovery relaxation is being walked upward"
    )


# ---------------------------------------------------------------------------
# A3. Hysteresis lock / defence-as-outage (DEFENCE_DoS). DEFENCE HOLDS.
# ---------------------------------------------------------------------------


def test_short_burst_does_not_lock_the_system_in_a_restricted_state() -> None:
    """Hysteresis must resist being weaponised into a denial of service.

    A brief burst should not leave benign traffic restricted for a long
    stretch afterwards, or the adversary gets an outage for the price of four
    requests.
    """
    engine = UREEngine(UREConfig(**TUNED))
    t = 0
    for _ in range(4):
        engine.observe(MAXED, now=float(t))
        t += 1

    restricted = 0
    for _ in range(60):
        a = engine.observe(BENIGN, now=float(t))
        t += 1
        if a.recommended_action.value in RESTRICTIVE:
            restricted += 1

    assert restricted < 30, (
        f"a 4-step burst kept {restricted}/60 benign steps restricted"
    )


# ---------------------------------------------------------------------------
# A4. AMX flooding (POISONING)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN WEAKNESS: AMX evicts weakest-first at capacity, so an "
           "adversary who can mint signatures at higher severity than a real "
           "profile can evict URE's memory of themselves. Fix requires "
           "eviction to weigh campaign status and age, not severity alone.",
)
def test_attack_memory_survives_a_flood_of_decoy_signatures() -> None:
    """A real attack profile must not be evictable by volume."""
    engine = UREEngine(UREConfig(amx_capacity=64))
    amx = engine.attack_memory
    amx.remember("REAL-ATTACK", severity=0.95, now=0.0)
    for index in range(500):
        amx.remember(f"decoy-{index}", severity=0.99, now=1.0 + index)

    assert amx.lookup("REAL-ATTACK", now=600.0) is not None, (
        "500 decoys evicted the real attack profile"
    )


# ---------------------------------------------------------------------------
# A5. BVE poisoning (DEFENCE_DoS)
# ---------------------------------------------------------------------------


def test_benign_evidence_blocks_false_vaccine_synthesis() -> None:
    """The discrimination threshold holds in the straightforward case.

    An adversary causing adverse outcomes while using markers that also appear
    in legitimate traffic should not be able to get those markers vaccinated
    against, because the benign occurrences drive P(adverse | pattern) below
    the synthesis threshold. As few as four benign observations suffice.
    """
    innocuous = ("login", "fetch_profile", "list_items")
    engine = UREEngine(UREConfig(synthesis_threshold=3))
    bve = engine.vaccines

    t = 0.0
    for _ in range(4):
        bve.observe(Observation(markers=innocuous, adverse=False, timestamp=t))
        t += 1
    for _ in range(8):
        bve.observe(Observation(markers=innocuous, adverse=True, timestamp=t))
        t += 1

    assert len(bve) == 0
    assert bve.immunize(innocuous, now=999.0).pressure == 0.0


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN WEAKNESS: BVE's observation window is bounded, so flooding "
           "adverse episodes evicts the benign evidence that would otherwise "
           "block synthesis. The discrimination check then sees an unopposed "
           "pattern. Fix requires benign evidence to be durable rather than "
           "window-scoped, or a minimum benign-observation count.",
)
def test_benign_evidence_cannot_be_flushed_out_of_the_window() -> None:
    """Window eviction must not defeat the discrimination threshold."""
    innocuous = ("login", "fetch_profile", "list_items")
    window = 64
    engine = UREEngine(UREConfig(synthesis_threshold=3))
    bve = engine.vaccines
    bve._observations = type(bve._observations)(maxlen=window)

    t = 0.0
    for _ in range(32):
        bve.observe(Observation(markers=innocuous, adverse=False, timestamp=t))
        t += 1
    for _ in range(window + 16):
        bve.observe(Observation(markers=innocuous, adverse=True, timestamp=t))
        t += 1

    assert bve.immunize(innocuous, now=1e6).pressure == 0.0, (
        "flooding the observation window produced false vaccines that fire "
        "on legitimate traffic"
    )


# ---------------------------------------------------------------------------
# A6. Saturation blinding (BLINDING)
# ---------------------------------------------------------------------------
#
# CORRECTED DIAGNOSIS. The original finding blamed energy saturating at
# ENERGY_MAX, and proposed a renormalising term as the fix. That was wrong, and
# the test below proves it: the two telemetry frames produce byte-identical
# pressure vectors. A 10x latency increase and a 10x adversarial-event
# increase are destroyed at NORMALIZATION, before the Lyapunov function is ever
# evaluated. No change to the energy function, and no duration floor, can
# recover information that was discarded one stage earlier.
#
# Fixing it properly would mean unbounded pressures, and bounded pressures are
# what make V(x) a Lyapunov function at all. So this is an architectural limit,
# not a defect, and the honest thing is to pin it as one and say what an
# operator should watch instead.
# ---------------------------------------------------------------------------


def test_escalation_beyond_full_scale_is_invisible_by_construction() -> None:
    """Pins the real mechanism, so nobody re-diagnoses it as an energy bug."""
    from ure_engine.telemetry import TelemetryAdapter, TelemetryFrame

    adapter = TelemetryAdapter()
    worse = dict(MAXED, latency_ms=50_000.0, adversarial_events=500.0)
    at_max = adapter.to_state_vector(TelemetryFrame(**MAXED), 0.0)
    escalated = adapter.to_state_vector(TelemetryFrame(**worse), 0.0)

    assert at_max.to_dict() == escalated.to_dict(), (
        "pressures no longer clamp identically; the saturation limit has "
        "moved and the reasoning in this module needs revisiting"
    )


def test_duration_under_saturation_is_still_observable() -> None:
    """What URE can still tell an operator once magnitude has run out.

    Every quantity that describes motion goes flat at saturation: energy pins,
    the derivative decays to zero, volatility collapses. Dwell does not, because
    it integrates rather than differentiates. It is the last signal standing,
    which is exactly why it is on the assessment.
    """
    engine = UREEngine(UREConfig(**TUNED))
    for step in range(40):
        a = engine.observe(MAXED, now=float(step))

    assert abs(a.lyapunov_derivative) < 0.01, "precondition: the derivative is flat"
    assert a.hostile_dwell > 0.75, (
        f"40 observations pinned at maximum reported dwell "
        f"{a.hostile_dwell:.2f}; the one surviving signal is not surviving"
    )


def test_saturation_still_recommends_restrictive_action() -> None:
    """The mitigating fact, pinned so it is not lost.

    Saturation costs URE the regime *label*, but hysteresis keeps the recovery
    *action* restrictive. The engine stops explaining itself correctly before
    it stops behaving correctly, which is the better of the two failure
    orders.
    """
    engine = UREEngine(UREConfig(**TUNED))
    for step in range(40):
        a = engine.observe(MAXED, now=float(step))

    assert a.recommended_action in (
        RecoveryAction.QUARANTINE, RecoveryAction.ISOLATE,
    ), f"saturated engine recommended {a.recommended_action.value}"
    assert a.regime is not SystemRegime.NOMINAL
