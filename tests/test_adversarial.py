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
All six are now closed, or pinned as limits with their cost measured. A1 is
closed by :mod:`ure_engine.dwell`; A4 and A5b by corroboration-weighted
eviction in AMX and a durable benign ledger in BVE. A6 turned out to be an
architectural limit rather than a defect, and is pinned as one below with a
corrected diagnosis.

Where an attack cannot be closed outright, the tests price it instead: what it
costs an adversary is asserted, so a change that makes the attack cheap fails
here rather than in production. Any weakness that is accepted rather than fixed
is marked
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


def run(engine: UREEngine, telemetry: dict, steps: int, start: int = 0) -> tuple:
    """Feed ``steps`` identical observations; return (last assessment, next t)."""
    assessment = None
    for step in range(start, start + steps):
        assessment = engine.observe(telemetry, now=float(step))
    assert assessment is not None
    return assessment, start + steps


def test_sustained_hostile_rate_is_eventually_detected() -> None:
    """A patient adversary at a constant rate should not evade forever.

    CLOSED by :mod:`ure_engine.dwell`. Before it, this failed at every
    duration, because a constant has no derivative and URE had nothing but
    derivatives to reason with.

    The quiet prelude is not scaffolding to make the test pass. Dwell measures
    departure from what this deployment does normally, so it needs to have seen
    normal, and a monitor that has been running since before the attack is the
    deployment being modelled. The case where it has not is
    ``test_an_engine_started_during_an_attack_calibrates_to_it`` below, which
    states that limit rather than hiding it.
    """
    engine = UREEngine(UREConfig(**TUNED))
    _, t = run(engine, BENIGN, 100)
    assessment, _ = run(engine, hostile(0.30), 80, start=t)

    assert assessment.regime.value not in BENIGN_REGIMES, (
        f"30% sustained hostile rate read as {assessment.regime.value} "
        f"after 80 observations"
    )


def test_evasion_boundary_now_costs_the_adversary_time() -> None:
    """Characterises what the fix bought, in the adversary's own currency.

    The old boundary was a knife edge at 36.2% hostile pressure that duration
    could not cross: 20 observations or 200, a rate below it was invisible
    forever. Dwell replaces that with a learned baseline plus a clock.
    """
    def steps_to_detect(level: float, cap: int = 400) -> int | None:
        engine = UREEngine(UREConfig(**TUNED))
        _, t = run(engine, BENIGN, 100)
        for step in range(cap):
            a = engine.observe(hostile(level), now=float(t + step))
            if a.regime.value not in BENIGN_REGIMES:
                return step + 1
        return None

    # What used to evade indefinitely is now detected, and detection time falls
    # as the rate rises: patience has a price.
    assert steps_to_detect(0.30) is not None
    assert steps_to_detect(0.20) is not None
    slow, fast = steps_to_detect(0.20), steps_to_detect(0.40)
    assert slow is not None and fast is not None and slow > fast

    # The old knife edge is gone: 0.30 was permanently invisible before.
    assert steps_to_detect(0.30, cap=80) is not None


def test_no_steady_rate_alarms_that_the_original_engine_did_not() -> None:
    """The regression guard on the whole mechanism.

    The first version of dwell charged above a FIXED floor of 0.12, which made
    a deployment whose normal rejection rate was 25% read ATTACKED after 25
    observations, recommending ISOLATE, with no adversary present. That is
    worse than the attack it fixed: a missed slow boil costs the attacker's
    throughput, a false ISOLATE is an outage you inflicted on yourself.

    The learned baseline must not reintroduce it at any level. 0.362 is where
    the pre-dwell engine detected on affinity alone, so anything below that
    alarming now would be a false positive this module created.
    """
    for level in (0.15, 0.20, 0.25, 0.30, 0.35):
        engine = UREEngine(UREConfig(**TUNED))
        for step in range(600):
            a = engine.observe(hostile(level), now=float(step))
            assert a.regime.value in BENIGN_REGIMES, (
                f"a deployment whose normal hostile rate is {level:.0%} was "
                f"called {a.regime.value} at observation {step}; the pre-dwell "
                f"engine did not alarm below 36.2% and neither may this"
            )


def test_walking_the_baseline_upward_is_slow_and_capped() -> None:
    """The attack surface a learned baseline creates, priced.

    Any adaptive baseline can in principle be walked: ramp slowly enough and it
    absorbs you. The defence is an exchange rate plus a hard cap, so this
    measures both. A retune of the dwell constants that makes walking cheap
    will fail here rather than in production.
    """
    def walk(target: float, rate: float) -> bool:
        """True if the adversary reached ``target`` and held it undetected."""
        engine = UREEngine(UREConfig(**TUNED))
        level, t = 0.0, 0
        while level < target:
            level = min(target, level + rate)
            engine.observe(hostile(level), now=float(t))
            t += 1
        for step in range(200):
            a = engine.observe(hostile(target), now=float(t + step))
            if a.regime.value not in BENIGN_REGIMES:
                return False
        return True

    # Walking at a rate a real campaign might use does not work.
    assert not walk(0.45, rate=1e-2)
    assert not walk(0.45, rate=1e-3)

    # The cap is the real defence: no amount of patience gets above it. Tens of
    # thousands of observations of restraint buy entry to a band where the
    # adversary is, by construction, indistinguishable from a legitimate
    # deployment that simply rejects a lot of traffic. That band cannot be
    # closed without alarming on the legitimate deployment, which is the whole
    # point of the mechanism.
    assert not walk(0.50, rate=1e-5)


def test_an_engine_started_during_an_attack_calibrates_to_it() -> None:
    """The cost of learning a baseline, stated rather than hidden.

    An engine whose first observations are the attack learns the attack as
    normal, up to the ceiling. This is inherent to baseline learning, not a
    tuning error, and the honest response is to pin it, document the operator
    remedy, and show that it self-heals.
    """
    attack = hostile(0.30)
    engine = UREEngine(UREConfig(**TUNED))
    blind, t = run(engine, attack, 200)
    assert blind.regime.value in BENIGN_REGIMES, (
        "cold-start calibration no longer happens; if that is deliberate, this "
        "test and the dwell docstring both need updating"
    )

    # It self-heals. The baseline follows a quieter floor down quickly, so the
    # first genuine lull re-calibrates and a resumption reads as excess. This
    # is why tau_down is much shorter than tau_up.
    _, t = run(engine, BENIGN, 400, start=t)
    resumed, _ = run(engine, attack, 120, start=t)
    assert resumed.regime.value not in BENIGN_REGIMES, (
        "after a lull re-calibrated the baseline, the same attack stayed "
        "invisible; the self-healing property is gone"
    )


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


def test_attack_memory_survives_a_flood_of_decoy_signatures() -> None:
    """A real attack profile must not be evictable by volume.

    CLOSED. Eviction used to rank on ``effective_severity`` alone, and severity
    is whatever the caller reported, so an adversary minting signatures at 0.99
    evicted a real profile sitting at 0.95. The memory forgot precisely the
    thing it most needed to remember, and it cost nothing to make it happen.
    """
    engine = UREEngine(UREConfig(amx_capacity=64))
    amx = engine.attack_memory
    amx.remember("REAL-ATTACK", severity=0.95, now=0.0)
    for index in range(500):
        amx.remember(f"decoy-{index}", severity=0.99, now=1.0 + index)

    assert amx.lookup("REAL-ATTACK", now=600.0) is not None, (
        "500 decoys evicted the real attack profile"
    )


def test_minting_signatures_cannot_evict_established_memory_at_any_volume() -> None:
    """The structural guarantee, not merely a favourable ranking.

    Ordering alone would be a soft defence: an adversary who out-claims the
    incumbent by a hair still wins, and severity is free to claim. So capacity
    is partitioned instead. Signatures seen exactly once compete only for
    their reserved share, which means a flood of minted signatures can evict
    nothing but each other however many of them there are.
    """
    engine = UREEngine(UREConfig(amx_capacity=64))
    amx = engine.attack_memory
    amx.remember("REAL-ATTACK", severity=0.95, now=0.0)

    now = 1.0
    for index in range(20_000):
        amx.remember(f"decoy-{index}", severity=1.0, now=now)
        now += 0.05

    assert amx.lookup("REAL-ATTACK", now=now) is not None, (
        "20,000 minted signatures at maximum severity displaced a real profile; "
        "the probation partition is not holding"
    )
    assert len(amx) <= 64


def test_severity_alone_cannot_outrank_corroboration() -> None:
    """The ordering that makes the partition work, pinned directly.

    Severity is supplied by the caller. Recurrence across elapsed time is
    spent, not supplied. Retention must reflect that asymmetry, or an adversary
    buys rank with a number.
    """
    engine = UREEngine(UREConfig(amx_capacity=1024))
    amx = engine.attack_memory

    amx.remember("established", severity=0.60, now=0.0)
    amx.remember("established", severity=0.60, now=120.0)
    amx.remember("established", severity=0.60, now=300.0)
    amx.remember("newcomer", severity=1.0, now=300.0)

    established = amx.lookup("established", now=301.0)
    newcomer = amx.lookup("newcomer", now=301.0)
    assert established is not None and newcomer is not None
    assert established.retention(301.0) > newcomer.retention(301.0), (
        "a maximum-severity first sighting outranks a signature that has "
        "actually been recurring; severity is deciding eviction again"
    )


def test_a_flood_that_does_succeed_announces_itself() -> None:
    """The half that cannot be fixed by ranking, so it is detected instead.

    An adversary willing to pay for corroborated decoys will match whatever
    evidence shape the incumbent has. What they cannot do is stay quiet about
    it: displacing a full memory takes thousands of adverse events in minutes.
    A memory that is full AND mostly new has just been replaced, and that is
    reported as adversarial pressure so the attack converts into a detection.
    """
    engine = UREEngine(UREConfig(amx_capacity=64))
    amx = engine.attack_memory
    _, t = run(engine, BENIGN, 100)

    for index in range(400):
        for _ in range(3):
            amx.remember(f"decoy-{index}", severity=0.99, now=float(t))
            t += 1

    assert amx.flood_pressure(now=float(t)) > 0.9
    assessment = engine.observe(BENIGN, now=float(t))
    assert assessment.regime.value not in BENIGN_REGIMES, (
        f"the memory was replaced wholesale and URE reported "
        f"{assessment.regime.value} on the next observation"
    )


def test_ordinary_signature_discovery_is_not_mistaken_for_a_flood() -> None:
    """The false positive the flood detector would cause if it were a rate.

    It is deliberately a fraction of this deployment's own memory rather than a
    signatures-per-minute threshold, because there is no universal answer to
    how many new signatures a healthy gateway sees. Both conditions have to
    hold: the memory must be full, and it must be mostly new.
    """
    engine = UREEngine(UREConfig(amx_capacity=64))
    amx = engine.attack_memory

    # Filling up. Trivially all-new, and must not read as flooded.
    for index in range(40):
        amx.remember(f"sig-{index}", severity=0.9, now=float(index))
    assert amx.flood_pressure(now=40.0) == 0.0

    # Full, and still discovering, but slowly. Steady state, not replacement.
    now = 100.0
    for index in range(200):
        amx.remember(f"steady-{index}", severity=0.5, now=now)
        now += 60.0
    assert amx.flood_pressure(now=now) == 0.0


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


def test_benign_evidence_cannot_be_flushed_out_of_the_window() -> None:
    """Window eviction must not defeat the discrimination threshold.

    CLOSED. Benign evidence used to be counted from the same bounded ring
    buffer as adverse evidence, so flooding adverse episodes past its length
    evicted the benign observations that would have blocked synthesis. The
    ratio then saw an unopposed pattern and vaccinated against ordinary
    traffic, which is a defence attacking its own users.

    Evidence that traffic is innocent now lives in a durable ledger. Evidence
    has to outlive the ring buffer or it is not evidence, it is a recency
    effect.
    """
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


def test_a_pattern_seen_behaving_benignly_is_never_vaccinated_at_any_ratio() -> None:
    """Durable evidence alone is not enough; the veto is what finishes it.

    With a durable ledger but only a ratio test, the adversary still wins by
    volume: they control the adverse side of P(adverse | pattern) and need only
    out-produce the benign evidence two or three to one.

    So a pattern known to occur in legitimate traffic is never vaccinated
    against, at any ratio. The vaccine would fire on that traffic, and no
    adverse count makes blocking real users acceptable. An attack that really
    does reuse a benign pattern has to be stopped by signature or policy, not
    by a rule that also hits customers.
    """
    innocuous = ("login", "fetch_profile", "list_items")
    engine = UREEngine(UREConfig(synthesis_threshold=3))
    bve = engine.vaccines

    t = 0.0
    for _ in range(4):
        bve.observe(Observation(markers=innocuous, adverse=False, timestamp=t))
        t += 1
    for _ in range(4000):
        bve.observe(Observation(markers=innocuous, adverse=True, timestamp=t))
        t += 1

    assert len(bve) == 0, "a 1000:1 adverse ratio bought a vaccine anyway"
    assert bve.immunize(innocuous, now=1e6).pressure == 0.0


def test_the_benign_ledger_still_learns_when_it_is_full() -> None:
    """The false positive a frequency-ranked ledger causes if left unpartitioned.

    Ranking eviction on sighting count is what protects well-corroborated
    evidence from being flushed. It also means a full ledger evicts every
    newcomer at one sighting, before it can ever reach two, so legitimate new
    traffic is never recorded as benign and becomes eligible for false
    vaccination. That is the failure this ledger exists to prevent,
    reintroduced one level down, and it was real until a probation share was
    reserved.
    """
    innocuous = ("login", "fetch_profile", "list_items")
    engine = UREEngine(UREConfig(synthesis_threshold=3))
    bve = engine.vaccines

    t = 0.0
    for index in range(5000):
        for _ in range(5):
            bve.observe(
                Observation(markers=(f"o{index}a", f"o{index}b"), adverse=False, timestamp=t)
            )
            t += 1

    for _ in range(4):
        bve.observe(Observation(markers=innocuous, adverse=False, timestamp=t))
        t += 1
    for _ in range(300):
        bve.observe(Observation(markers=innocuous, adverse=True, timestamp=t))
        t += 1

    assert bve.immunize(innocuous, now=1e6).pressure == 0.0, (
        "a benign pattern first seen after the ledger filled up was never "
        "recorded, and got vaccinated against"
    )


def test_burying_benign_evidence_costs_the_adversary_successful_traffic() -> None:
    """The residual, priced rather than papered over.

    Evidence cannot be both durable and unboundedly stored. An adversary who
    generates enough distinct, more-frequently-seen benign marker sets does
    eventually push a protected pattern out of the ledger.

    What it costs them is the point. Burying evidence seen N times requires
    roughly ``BENIGN_LEDGER_CAPACITY * (N + 1)`` requests that **succeed**,
    because only non-adverse outcomes write to this ledger. An adversary who
    can push twenty thousand successful requests through does not need the
    vaccine attack. This pins that moderate effort buys nothing, so a change
    that makes the attack cheap fails here.
    """
    innocuous = ("login", "fetch_profile", "list_items")
    engine = UREEngine(UREConfig(synthesis_threshold=3))
    bve = engine.vaccines

    t = 0.0
    for _ in range(4):
        bve.observe(Observation(markers=innocuous, adverse=False, timestamp=t))
        t += 1
    for index in range(2000):
        for _ in range(5):
            bve.observe(
                Observation(
                    markers=(f"j{index}a", f"j{index}b", f"j{index}c"),
                    adverse=False,
                    timestamp=t,
                )
            )
            t += 1
    for _ in range(200):
        bve.observe(Observation(markers=innocuous, adverse=True, timestamp=t))
        t += 1

    assert bve.immunize(innocuous, now=1e6).pressure == 0.0, (
        "10,000 successful attacker requests were enough to bury the benign "
        "evidence; the ledger's exchange rate has collapsed"
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
    _, t = run(engine, BENIGN, 60)
    a, _ = run(engine, MAXED, 60, start=t)

    assert abs(a.lyapunov_derivative) < 0.01, "precondition: the derivative is flat"
    assert a.hostile_dwell > 0.60, (
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
