"""
Tests for the sentinel_os adapter.

These concentrate on the three things that were actually wrong when the
integration was first built and run, rather than on restating the mapping.
Each failure produced a precisely computed, confidently wrong assessment out of
a correct engine, which is the characteristic way an adapter breaks.
"""

from __future__ import annotations

from ure_engine import (
    RecoveryAction,
    SentinelAdapter,
    SentinelConfig,
    SystemRegime,
    UREConfig,
    UREEngine,
)

CONFIG = SentinelConfig(
    queue_capacity=128, latency_ceiling_s=60.0, dlq_rate_ceiling=20.0, overdue_ceiling=20
)


def snapshot(calls: int, failures: int, rejections: int, state: str = "closed") -> dict:
    return {"state": state, "total_calls": calls, "total_failures": failures,
            "total_rejections": rejections, "total_opens": 0}


def stats(depth: int = 4, overdue: int = 0, age_s: float = 0.3) -> dict:
    return {"depth_ready": depth, "processing_overdue": overdue,
            "oldest_pending_age_s": age_s, "dead": 0, "dead_last_hour": 0}


def test_counters_are_differentiated_not_averaged() -> None:
    """The defect that makes a correct engine confidently wrong.

    sentinel_os counters are lifetime cumulative. A lifetime average converges
    and then stops responding: after a million healthy calls a burst of
    failures barely moves it, and the assessment stays green through an
    outage. This pins that a fresh burst reads as a fresh burst.
    """
    adapter = SentinelAdapter(CONFIG)
    adapter.to_frame(snapshot(1_000_000, 10, 5), stats())
    frame = adapter.to_frame(snapshot(1_000_100, 60, 5), stats())

    assert frame.error_rate == 0.5, (
        f"50 failures in 100 calls after a million healthy ones reported "
        f"{frame.error_rate:.4f}; the counters are being averaged, not "
        f"differentiated"
    )


def test_first_sample_reports_nothing_rather_than_a_lifetime_average() -> None:
    """No interval yet means no rate. Reporting the lifetime figure as current
    would make every cold start begin with a fabricated reading."""
    frame = SentinelAdapter(CONFIG).to_frame(snapshot(500_000, 250_000, 100_000), stats())
    assert frame.error_rate == 0.0
    assert frame.blocked_rate == 0.0


def test_rejections_are_added_back_into_attempts() -> None:
    """Rejections never reach the callable, so they are absent from
    total_calls. Dividing by total_calls alone overstates the block rate."""
    adapter = SentinelAdapter(CONFIG)
    adapter.to_frame(snapshot(0, 0, 0), stats())
    frame = adapter.to_frame(snapshot(75, 0, 25), stats())
    assert frame.blocked_rate == 0.25


def test_queue_age_is_rescaled_to_ure_units() -> None:
    """URE treats 1000ms as full latency pressure; sentinel_os reports queue
    staleness in seconds. Raw pass-through saturates at one second of age,
    which is what made a 7%-ramped surge read STRESSED."""
    adapter = SentinelAdapter(CONFIG)
    adapter.to_frame(snapshot(0, 0, 0), stats())
    mild = adapter.to_frame(snapshot(100, 0, 0), stats(age_s=6.0))
    assert mild.latency_ms == 100.0

    stale = adapter.to_frame(snapshot(200, 0, 0), stats(age_s=600.0))
    assert stale.latency_ms == 1000.0


def test_open_breaker_is_not_counted_into_failure_twice() -> None:
    """URE's own TelemetryAdapter applies the circuit-state step from
    `circuit_state`. Adding it here as well double-counts an open breaker."""
    adapter = SentinelAdapter(CONFIG)
    adapter.to_frame(snapshot(0, 0, 0, state="OPEN"), stats())
    frame = adapter.to_frame(snapshot(100, 0, 0, state="OPEN"), stats())
    assert frame.retry_rate == 0.0
    assert frame.circuit_state == "OPEN"


def test_a_sustained_rejection_rate_reaches_the_engine_as_threat() -> None:
    """The integration's central wiring decision, end to end.

    Routing rejections only into blocked_rate gives them a 0.3 weight inside
    URE, and a sustained rejection rate then reads NOMINAL. From a gateway's
    point of view a sustained rejection rate IS the threat signal, so it drives
    risk_score directly.

    Asserted on the exact arithmetic rather than on "not benign", because a
    regime assertion alone is satisfiable by a dozen unrelated paths and says
    nothing about which pressure carried the signal.
    """
    adapter = SentinelAdapter(CONFIG)
    engine = UREEngine(UREConfig(smoothing_alpha=0.6, slope_window=12,
                                 history_window=64, restore_patience=5))
    calls = rejections = step = 0

    # A quiet stretch first, so the engine knows what this deployment looks
    # like when nobody is attacking it. URE judges hostile pressure against
    # that baseline rather than against a fixed number, precisely so that a
    # deployment which has always rejected 40% is not called ATTACKED for it.
    for _ in range(120):
        calls += 100
        engine.observe(adapter.to_frame(snapshot(calls, 1, rejections), stats()),
                       now=float(step))
        step += 1

    for _ in range(60):
        calls += 60
        rejections += 40
        frame = adapter.to_frame(snapshot(calls, 1, rejections), stats())
        assessment = engine.observe(frame, now=float(step))
        step += 1

    # 40 rejections in 100 attempts, routed at half weight into risk_score.
    assert frame.blocked_rate == 0.4
    assert frame.risk_score == 0.2
    # URE's own weighting: 0.2 * 0.7 + 0.4 * 0.3 = 0.26, so the signal really
    # did arrive through threat and not through some operational pressure.
    assert assessment.state.threat_pressure == 0.26

    # And the conclusion that follows: malice, not load, so isolate rather
    # than throttle.
    assert assessment.regime is SystemRegime.ATTACKED
    assert assessment.recommended_action is RecoveryAction.ISOLATE


def test_reset_restores_cold_start_behaviour() -> None:
    adapter = SentinelAdapter(CONFIG)
    adapter.to_frame(snapshot(0, 0, 0), stats())
    adapter.to_frame(snapshot(100, 50, 0), stats())
    adapter.reset()
    frame = adapter.to_frame(snapshot(999_999, 999, 0), stats())
    assert frame.error_rate == 0.0
