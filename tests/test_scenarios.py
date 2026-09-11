"""Scenario tests: the claims URE makes about whole incidents.

Unit tests check that each part computes what it says. These check the thing
the system is actually for -- that a realistic incident produces the right
sequence of decisions over time. They mirror ``examples/incident_walkthrough.py``
so the demo cannot silently stop demonstrating what it claims.
"""

from __future__ import annotations

import pytest

from ure_engine import RecoveryAction, SystemRegime, UREAssessment, UREConfig, UREEngine

ATTACK_MARKERS = ["ignore_previous", "developer_override", "show_prompt"]


def run(engine: UREEngine, frames: list[dict], start: float = 0.0) -> list[UREAssessment]:
    return [
        engine.observe(frame, now=start + index) for index, frame in enumerate(frames)
    ]


def load_surge(steps: int = 8) -> list[dict]:
    """A legitimate traffic surge: latency, retries and queue climb, risk does not."""
    return [
        {
            "risk_score": 0.05,
            "retry_rate": 0.6 * (step + 1) / steps,
            "latency_ms": 900.0 * (step + 1) / steps,
            "queue_saturation": 0.95 * (step + 1) / steps,
            "substrate_health": 1.0 - 0.5 * (step + 1) / steps,
        }
        for step in range(steps)
    ]


def probing_campaign(steps: int = 8) -> list[dict]:
    """A hostile campaign: risk and block rate climb, the substrate is fine."""
    return [
        {
            "risk_score": 0.50 + 0.20 * (step + 1) / steps,
            "blocked_rate": 0.45 + 0.15 * (step + 1) / steps,
            "retry_rate": 0.05,
            "latency_ms": 40.0,
            "queue_saturation": 0.1,
            "signatures": ["probe-7f3a"],
            "markers": ATTACK_MARKERS,
        }
        for step in range(steps)
    ]


class TestTheCentralAsymmetry:
    """Load and malice are different conditions calling for opposite responses.

    This is the claim that justifies the entire regime model over a scalar risk
    score. If these two scenarios ever produce the same recommendation, the
    model has stopped earning its complexity.
    """

    def test_load_is_stressed_and_throttled(self) -> None:
        final = run(UREEngine(), load_surge())[-1]
        assert final.regime is SystemRegime.STRESSED
        assert final.recommended_action is RecoveryAction.THROTTLE
        assert final.recovery_vector is not None
        assert final.recovery_vector.controls["reduce_rate_limit"] is True

    def test_malice_is_attacked_and_isolated_never_throttled(self) -> None:
        final = run(UREEngine(), probing_campaign())[-1]
        assert final.regime is SystemRegime.ATTACKED
        assert final.recommended_action is RecoveryAction.ISOLATE
        assert final.recovery_vector is not None
        # Throttling here would deliver the denial of service the attacker wanted.
        assert final.recovery_vector.controls["reduce_rate_limit"] is False
        assert final.recovery_vector.controls["isolate_scope"] is True

    def test_the_two_scenarios_diverge_despite_comparable_energy(self) -> None:
        load = run(UREEngine(), load_surge())[-1]
        attack = run(UREEngine(), probing_campaign())[-1]
        # Both are serious -- a scalar risk score would struggle to separate
        # them -- yet the recommended actions are different.
        assert load.lyapunov_energy > 0.5
        assert attack.lyapunov_energy > 0.5
        assert load.recommended_action is not attack.recommended_action
        assert load.regime is not attack.regime


class TestIncidentLifecycle:
    def test_a_surge_that_passes_ends_in_recovery(self) -> None:
        engine = UREEngine()
        frames = load_surge()
        # ... and then the same ramp in reverse.
        frames += list(reversed(load_surge()))
        frames += [{"risk_score": 0.02, "retry_rate": 0.01, "latency_ms": 15.0}] * 6
        assessments = run(engine, frames)

        regimes = [a.regime for a in assessments]
        assert SystemRegime.STRESSED in regimes
        assert SystemRegime.RECOVERING in regimes
        # And the controls are eventually released.
        assert assessments[-1].recommended_action in (
            RecoveryAction.NONE,
            RecoveryAction.RESTORE,
        )

    def test_controls_are_not_released_the_instant_energy_dips(self) -> None:
        # Hysteresis: an oscillating control loop is worse than either state.
        engine = UREEngine(UREConfig(restore_patience=3))
        run(engine, load_surge())
        recovery = run(engine, [{"risk_score": 0.0, "latency_ms": 1.0}], start=100.0)
        assert recovery[0].recommended_action is RecoveryAction.THROTTLE
        assert recovery[0].recovery_vector is not None
        assert "hysteresis" in recovery[0].recovery_vector.rationale

    def test_held_controls_match_the_held_action(self) -> None:
        # A vector whose action says THROTTLE while its controls say resume
        # would be obeyed by its controls, undoing the hold.
        engine = UREEngine(UREConfig(restore_patience=3))
        run(engine, load_surge())
        held = run(engine, [{"risk_score": 0.0, "latency_ms": 1.0}], start=100.0)[0]
        assert held.recommended_action is RecoveryAction.THROTTLE
        assert held.recovery_vector is not None
        assert held.recovery_vector.controls.get("reduce_rate_limit") is True


class TestLearningAcrossAnIncident:
    def test_the_engine_ends_the_incident_knowing_more_than_it_started(self) -> None:
        engine = UREEngine(UREConfig(synthesis_threshold=3))
        assert len(engine.attack_memory) == 0
        assert len(engine.vaccines) == 0

        run(engine, probing_campaign())

        assert len(engine.attack_memory) >= 1
        assert len(engine.vaccines) >= 1
        assert engine.vaccines.adaptation_score() > 0.0

    def test_a_repeat_campaign_is_recognized_faster_than_the_first(self) -> None:
        # The point of AMX: the second encounter should not start from zero.
        engine = UREEngine(UREConfig(synthesis_threshold=3))
        first = run(engine, probing_campaign())

        # Quiet spell, then the same campaign returns.
        run(engine, [{"risk_score": 0.0, "latency_ms": 5.0}] * 10, start=100.0)
        second = run(engine, probing_campaign(), start=200.0)

        first_hit = next(
            (i for i, a in enumerate(first) if a.regime is SystemRegime.ATTACKED), None
        )
        second_hit = next(
            (i for i, a in enumerate(second) if a.regime is SystemRegime.ATTACKED), None
        )
        assert first_hit is not None and second_hit is not None
        assert second_hit <= first_hit
        assert second[second_hit].active_attack_profiles

    def test_vaccines_fire_on_a_partial_sequence_after_learning(self) -> None:
        engine = UREEngine(UREConfig(synthesis_threshold=3))
        run(engine, probing_campaign())

        # Only the first two markers of a known escalation. The whole value of
        # BVE is acting here rather than after the third.
        assessment = engine.observe(
            {"risk_score": 0.1, "latency_ms": 20.0, "markers": ATTACK_MARKERS[:2]},
            now=500.0,
        )
        assert assessment.active_vaccines


class TestAdaptiveThresholdInPractice:
    def test_the_threshold_falls_as_an_incident_develops(self) -> None:
        engine = UREEngine()
        assessments = run(engine, probing_campaign())
        thresholds = [a.recommended_threshold for a in assessments]
        assert thresholds[-1] < thresholds[0]

    def test_a_quiet_system_is_more_generous_than_a_busy_one(self) -> None:
        quiet = run(UREEngine(), [{"risk_score": 0.0, "latency_ms": 5.0}] * 8)[-1]
        busy = run(UREEngine(), load_surge())[-1]
        assert quiet.recommended_threshold > busy.recommended_threshold


class TestExplanationsStayUseful:
    @pytest.mark.parametrize(
        "frames,expected",
        [
            (load_surge(), "STRESSED"),
            (probing_campaign(), "ATTACKED"),
        ],
    )
    def test_the_explanation_names_the_regime(self, frames: list[dict], expected: str) -> None:
        final = run(UREEngine(), frames)[-1]
        assert expected in final.explanation

    def test_the_explanation_is_a_paragraph_not_a_token(self) -> None:
        final = run(UREEngine(), probing_campaign())[-1]
        # An operator reading this at 3am needs the driver and the recommendation.
        assert len(final.explanation) > 120
        assert "Resilience" in final.explanation

    def test_memory_activity_is_reported_when_it_fires(self) -> None:
        engine = UREEngine(UREConfig(synthesis_threshold=3))
        run(engine, probing_campaign())
        final = run(engine, probing_campaign(), start=200.0)[-1]
        assert "AMX recalls" in final.explanation
