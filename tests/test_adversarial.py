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
* energy saturates at ENERGY_MAX, destroying trajectory information

Each is a deliberate design choice. Each is also a lever.

HOW THE KNOWN WEAKNESSES ARE ENCODED
------------------------------------
Four of these six attacks currently succeed. They are marked
``xfail(strict=True)`` rather than deleted or softened, which means:

* the suite stays green while the weakness is known and accepted;
* the moment someone fixes one, the test XPASSes, ``strict=True`` turns that
  into a failure, and the fixer is forced to come here and flip the marker.

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


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN WEAKNESS: a constant hostile rate has no derivative, so "
           "URE's trajectory machinery cannot see it. An adversary holding "
           "below the affinity boundary is invisible for any duration. Fix "
           "requires a duration-based floor, not a slope.",
)
def test_sustained_hostile_rate_is_eventually_detected() -> None:
    """A patient adversary at a constant rate should not evade forever."""
    engine = UREEngine(UREConfig(**TUNED))
    telemetry = hostile(0.30)
    for step in range(80):
        assessment = engine.observe(telemetry, now=float(step))
    assert assessment.regime.value not in BENIGN_REGIMES, (
        f"30% sustained hostile rate read as {assessment.regime.value} "
        f"after 80 observations"
    )


def test_evasion_boundary_is_sharp_and_time_independent() -> None:
    """Characterises the weakness above: the boundary is a level, not a duration.

    Not an aspiration. This documents measured behaviour so a change to the
    affinity functions that moves the boundary shows up as a failure here.
    """
    def detected(level: float, steps: int) -> bool:
        engine = UREEngine(UREConfig(**TUNED))
        for step in range(steps):
            a = engine.observe(hostile(level), now=float(step))
        return a.regime.value not in BENIGN_REGIMES

    # Below the boundary, duration does not help: 20 steps or 200, still blind.
    assert not detected(0.30, 20)
    assert not detected(0.30, 200)
    # Above it, detection is immediate rather than gradual.
    assert detected(0.60, 20)


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


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN WEAKNESS: energy is bounded at ENERGY_MAX, so once pinned "
           "there is no headroom left for a derivative and further escalation "
           "is invisible. Same root cause as the slow boil: a constant "
           "defeats a trajectory. Fix requires an unbounded or renormalising "
           "term, or a duration-based floor.",
)
def test_further_escalation_is_visible_even_at_saturation() -> None:
    """A system already at maximum energy must still notice it is getting worse."""
    engine = UREEngine(UREConfig(**TUNED))
    t = 0
    for _ in range(30):
        engine.observe(MAXED, now=float(t))
        t += 1

    worse = dict(MAXED, latency_ms=50_000.0, adversarial_events=500.0,
                 queue_saturation=1.0)
    for _ in range(10):
        a = engine.observe(worse, now=float(t))
        t += 1

    assert abs(a.lyapunov_derivative) > 0.01, (
        f"a large further escalation moved dV/dt only to "
        f"{a.lyapunov_derivative:+.5f}"
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
