"""Tests for trajectory motion, recovery planning, resilience, and thresholds."""

from __future__ import annotations

import pytest

from ure_engine.lyapunov import ENERGY_MAX
from ure_engine.recovery import RecoveryAction, RecoveryPlanner
from ure_engine.regimes import RegimeProfile, SystemRegime
from ure_engine.resilience import ResilienceIndex
from ure_engine.state_vector import StateVector
from ure_engine.thresholds import (
    REGIME_ANCHORS,
    THRESHOLD_CEILING,
    THRESHOLD_FLOOR,
    AdaptiveThresholdController,
)
from ure_engine.trajectory import TrajectoryEngine


def profile_for(regime: SystemRegime, confidence: float = 0.9) -> RegimeProfile:
    return RegimeProfile(
        dominant=regime,
        distribution={regime: confidence},
        confidence=confidence,
        entropy=1.0 - confidence,
        runner_up=SystemRegime.NOMINAL,
        margin=confidence - (1.0 - confidence),
    )


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class TestTrajectoryEngine:
    def test_first_sample_is_at_rest(self) -> None:
        engine = TrajectoryEngine()
        snapshot = engine.update(0.0, 0.5)
        assert snapshot.velocity == 0.0
        assert snapshot.acceleration == 0.0
        assert snapshot.volatility == 0.0

    def test_rejects_a_degenerate_window(self) -> None:
        with pytest.raises(ValueError):
            TrajectoryEngine(window=1)

    def test_velocity_is_the_first_difference(self) -> None:
        engine = TrajectoryEngine()
        engine.update(0.0, 0.0)
        snapshot = engine.update(1.0, 0.5)
        assert snapshot.velocity == pytest.approx(0.5)

    def test_acceleration_detects_a_ramp(self) -> None:
        engine = TrajectoryEngine()
        engine.update(0.0, 0.0)
        engine.update(1.0, 0.1)
        snapshot = engine.update(2.0, 0.4)  # steps of 0.1 then 0.3
        assert snapshot.acceleration > 0.0
        assert snapshot.escalating

    def test_simultaneous_samples_do_not_divide_by_zero(self) -> None:
        engine = TrajectoryEngine()
        engine.update(5.0, 0.1)
        snapshot = engine.update(5.0, 0.9)
        assert snapshot.velocity == pytest.approx(0.8 / 1e-3)  # clamped dt, finite

    def test_steady_signal_has_no_volatility(self) -> None:
        engine = TrajectoryEngine()
        for index in range(10):
            snapshot = engine.update(float(index), 0.5)
        assert snapshot.volatility == pytest.approx(0.0)

    def test_oscillation_is_visible_where_the_mean_slope_is_not(self) -> None:
        # A system flapping between two levels has a mean slope near zero and
        # would be called stable by any derivative-only check.
        engine = TrajectoryEngine()
        for index in range(12):
            snapshot = engine.update(float(index), 0.55 if index % 2 else 0.75)
        assert abs(snapshot.velocity) > 0.0
        assert snapshot.oscillations >= 3
        assert snapshot.unstable

    def test_flat_steps_are_not_counted_as_reversals(self) -> None:
        # An idle system producing identical energies must not register as
        # violently unstable.
        engine = TrajectoryEngine()
        for index in range(20):
            snapshot = engine.update(float(index), 0.4)
        assert snapshot.oscillations == 0
        assert not snapshot.unstable

    def test_settling_is_falling_and_decelerating(self) -> None:
        engine = TrajectoryEngine()
        for index, energy in enumerate((1.0, 0.6, 0.4, 0.35)):
            snapshot = engine.update(float(index), energy)
        assert snapshot.settling

    def test_stability_score_is_high_when_still_and_low_when_erratic(self) -> None:
        calm = TrajectoryEngine()
        for index in range(10):
            calm.update(float(index), 0.5)
        assert calm.stability_score() > 0.9

        erratic = TrajectoryEngine()
        for index in range(20):
            erratic.update(float(index), 0.1 if index % 2 else 1.2)
        assert erratic.stability_score() < 0.4

    def test_drift_measures_travel_through_pressure_space(self) -> None:
        engine = TrajectoryEngine()
        start = StateVector.create(threat_pressure=0.0, timestamp=0.0)
        end = StateVector.create(threat_pressure=1.0, timestamp=1.0)
        engine.update(0.0, 0.1, start)
        engine.update(1.0, 0.2, end)
        assert engine.drift() == pytest.approx(1.0)

    def test_window_is_bounded(self) -> None:
        engine = TrajectoryEngine(window=5)
        for index in range(50):
            engine.update(float(index), 0.5)
        assert engine.depth == 5


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


class TestRecoveryPlanner:
    def test_nominal_recommends_nothing(self) -> None:
        planner = RecoveryPlanner()
        vector = planner.plan(
            StateVector.zero(), profile_for(SystemRegime.NOMINAL), 0.05, 0.0, 0.95
        )
        assert vector.action is RecoveryAction.NONE

    def test_stressed_throttles(self) -> None:
        planner = RecoveryPlanner()
        state = StateVector.create(
            latency_pressure=0.9, failure_pressure=0.8, resource_pressure=0.9
        )
        vector = planner.plan(state, profile_for(SystemRegime.STRESSED), 1.2, 0.01, 0.4)
        assert vector.action is RecoveryAction.THROTTLE
        assert vector.controls["reduce_rate_limit"] is True

    def test_attacked_isolates_and_deliberately_does_not_throttle(self) -> None:
        # The central asymmetry: throttling an attacked system completes the
        # denial of service the adversary was attempting.
        planner = RecoveryPlanner()
        state = StateVector.create(threat_pressure=0.9, adversarial_pressure=0.8)
        vector = planner.plan(state, profile_for(SystemRegime.ATTACKED), 0.9, 0.02, 0.4)
        assert vector.action is RecoveryAction.ISOLATE
        assert vector.controls["reduce_rate_limit"] is False
        assert vector.controls["isolate_scope"] is True

    def test_cascading_quarantines_and_preserves_evidence(self) -> None:
        planner = RecoveryPlanner()
        state = StateVector.create(failure_pressure=1.0, resource_pressure=1.0)
        vector = planner.plan(state, profile_for(SystemRegime.CASCADING), 1.4, 0.1, 0.1)
        assert vector.action is RecoveryAction.QUARANTINE
        assert vector.controls["preserve_state"] is True
        assert vector.controls["disable_learning"] is True

    def test_cascading_freezes_learning(self) -> None:
        # An engine that keeps learning through its own collapse learns the
        # collapse, not the attack.
        planner = RecoveryPlanner()
        vector = planner.plan(
            StateVector.create(failure_pressure=1.0),
            profile_for(SystemRegime.CASCADING),
            1.4,
            0.1,
            0.1,
        )
        assert vector.controls["freeze_vaccines"] is True

    def test_recovering_restores(self) -> None:
        planner = RecoveryPlanner()
        vector = planner.plan(
            StateVector.create(drift_pressure=0.5),
            profile_for(SystemRegime.RECOVERING),
            0.5,
            -0.08,
            0.6,
        )
        assert vector.action is RecoveryAction.RESTORE

    def test_unknown_degrades_rather_than_assuming_health(self) -> None:
        planner = RecoveryPlanner()
        vector = planner.plan(
            StateVector.zero(), profile_for(SystemRegime.UNKNOWN), 0.0, 0.0, 0.5
        )
        assert vector.action is RecoveryAction.DEGRADE
        assert vector.controls["page_operator"] is True

    def test_ambiguous_and_rising_takes_a_cheap_precaution(self) -> None:
        planner = RecoveryPlanner()
        ambiguous = RegimeProfile(
            dominant=SystemRegime.ADAPTING,
            distribution={SystemRegime.ADAPTING: 0.34},
            confidence=0.34,
            entropy=0.9,
            runner_up=SystemRegime.STRESSED,
            margin=0.01,
        )
        vector = planner.plan(StateVector.zero(), ambiguous, 0.5, 0.06, 0.5)
        assert vector.action is RecoveryAction.DEGRADE

    def test_escalation_is_immediate(self) -> None:
        planner = RecoveryPlanner(restore_patience=3)
        planner.plan(StateVector.zero(), profile_for(SystemRegime.NOMINAL), 0.0, 0.0, 1.0)
        vector = planner.plan(
            StateVector.create(failure_pressure=1.0, resource_pressure=1.0),
            profile_for(SystemRegime.CASCADING),
            1.4,
            0.1,
            0.1,
        )
        assert vector.action is RecoveryAction.QUARANTINE

    def test_de_escalation_requires_sustained_improvement(self) -> None:
        # Releasing controls on one good sample is how a control loop starts
        # oscillating, which is worse than either fixed state.
        planner = RecoveryPlanner(restore_patience=3)
        planner.plan(
            StateVector.create(failure_pressure=1.0, resource_pressure=1.0),
            profile_for(SystemRegime.CASCADING),
            1.4,
            0.1,
            0.1,
        )
        held = planner.plan(
            StateVector.zero(), profile_for(SystemRegime.NOMINAL), 0.05, 0.0, 0.95
        )
        assert held.action is RecoveryAction.QUARANTINE
        assert "hysteresis" in held.rationale

    def test_controls_are_released_once_calm_persists(self) -> None:
        planner = RecoveryPlanner(restore_patience=2)
        planner.plan(
            StateVector.create(failure_pressure=1.0, resource_pressure=1.0),
            profile_for(SystemRegime.CASCADING),
            1.4,
            0.1,
            0.1,
        )
        for _ in range(6):
            vector = planner.plan(
                StateVector.zero(), profile_for(SystemRegime.NOMINAL), 0.02, 0.0, 0.98
            )
        assert vector.action is RecoveryAction.NONE

    def test_recovery_score_falls_as_capability_is_spent(self) -> None:
        planner = RecoveryPlanner()
        assert planner.recovery_score() == 1.0
        planner.plan(
            StateVector.create(failure_pressure=1.0, resource_pressure=1.0),
            profile_for(SystemRegime.CASCADING),
            1.4,
            0.1,
            0.1,
        )
        assert planner.recovery_score() < 0.5

    def test_action_disruption_is_ordered(self) -> None:
        assert RecoveryAction.NONE.disruption < RecoveryAction.DEGRADE.disruption
        assert RecoveryAction.THROTTLE.disruption < RecoveryAction.ISOLATE.disruption
        assert RecoveryAction.ISOLATE.disruption < RecoveryAction.QUARANTINE.disruption

    def test_restore_removes_no_capability(self) -> None:
        assert RecoveryAction.RESTORE.disruption < RecoveryAction.DEGRADE.disruption


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


class TestResilienceIndex:
    @staticmethod
    def compute(index: ResilienceIndex, **overrides: object):
        kwargs: dict[str, object] = dict(
            energy=0.0,
            derivative=0.0,
            profile=profile_for(SystemRegime.NOMINAL),
            trajectory_stability=1.0,
            adaptation_score=1.0,
            recovery_score=1.0,
            memory_depth=64,
            memory_recall=0.0,
        )
        kwargs.update(overrides)
        return index.compute(**kwargs)  # type: ignore[arg-type]

    def test_a_perfect_system_scores_one(self) -> None:
        result = self.compute(ResilienceIndex())
        assert result.index == pytest.approx(1.0)
        assert result.band == "RESILIENT"

    def test_saturated_energy_collapses_the_index(self) -> None:
        result = self.compute(
            ResilienceIndex(),
            energy=ENERGY_MAX,
            profile=profile_for(SystemRegime.CASCADING),
            trajectory_stability=0.0,
            adaptation_score=0.0,
            recovery_score=0.0,
            memory_depth=0,
            memory_recall=1.0,
        )
        assert result.index < 0.1
        assert result.band == "CRITICAL"

    def test_regime_ceiling_prevents_a_reassuring_number_during_an_outage(self) -> None:
        # Every term strong, but the system is cascading. The index must not
        # report health during an outage.
        result = self.compute(ResilienceIndex(), profile=profile_for(SystemRegime.CASCADING))
        assert result.index <= 0.20
        assert result.ceiling_binding

    def test_rising_energy_erodes_more_than_falling_restores(self) -> None:
        index = ResilienceIndex()
        rising = self.compute(index, derivative=0.05)
        falling = self.compute(index, derivative=-0.05)
        assert falling.index > rising.index

    def test_learning_raises_resilience(self) -> None:
        index = ResilienceIndex()
        naive = self.compute(index, adaptation_score=0.0)
        seasoned = self.compute(index, adaptation_score=1.0)
        assert seasoned.index > naive.index

    def test_memory_breadth_helps_and_active_recall_hurts(self) -> None:
        index = ResilienceIndex()
        idle_with_memory = self.compute(index, memory_depth=64, memory_recall=0.0)
        under_recall = self.compute(index, memory_depth=64, memory_recall=1.0)
        assert idle_with_memory.index > under_recall.index

    def test_spent_capability_lowers_the_index(self) -> None:
        # A system only upright because half its capability is switched off is
        # not resilient, and the index must say so.
        index = ResilienceIndex()
        intact = self.compute(index, recovery_score=1.0)
        degraded = self.compute(index, recovery_score=0.0)
        assert intact.index > degraded.index

    def test_weakest_term_points_at_the_leverage(self) -> None:
        result = self.compute(ResilienceIndex(), adaptation_score=0.0)
        assert result.weakest_term == "adaptation"

    def test_bands_are_ordered(self) -> None:
        index = ResilienceIndex()
        assert self.compute(index).band == "RESILIENT"
        assert self.compute(index, energy=0.75).band in {"ADEQUATE", "RESILIENT"}

    def test_custom_weights_are_normalized(self) -> None:
        index = ResilienceIndex(
            {
                "stability": 3.0,
                "trajectory": 1.0,
                "adaptation": 1.0,
                "recovery": 1.0,
                "memory": 1.0,
            }
        )
        assert self.compute(index).index == pytest.approx(1.0)

    def test_missing_weights_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            ResilienceIndex({"stability": 1.0})

    def test_breakdown_is_json_safe(self) -> None:
        import json

        payload = json.loads(json.dumps(self.compute(ResilienceIndex()).to_dict()))
        assert set(payload["terms"]) == {
            "stability", "trajectory", "adaptation", "recovery", "memory"
        }


# ---------------------------------------------------------------------------
# Adaptive thresholds
# ---------------------------------------------------------------------------


class TestAdaptiveThresholdController:
    def test_architecture_anchors_are_honoured(self) -> None:
        # The three anchors the architecture specifies by name.
        assert REGIME_ANCHORS[SystemRegime.NOMINAL] == 0.85
        assert REGIME_ANCHORS[SystemRegime.ATTACKED] == 0.45
        assert REGIME_ANCHORS[SystemRegime.CASCADING] == 0.20

    def test_anchors_are_monotone_in_severity(self) -> None:
        ladder = [
            SystemRegime.NOMINAL,
            SystemRegime.ADAPTING,
            SystemRegime.RECOVERING,
            SystemRegime.STRESSED,
            SystemRegime.ATTACKED,
            SystemRegime.CASCADING,
        ]
        values = [REGIME_ANCHORS[regime] for regime in ladder]
        assert values == sorted(values, reverse=True)

    def test_threshold_tightens_as_the_regime_worsens(self) -> None:
        controller = AdaptiveThresholdController()
        calm = controller.compute(
            regime=SystemRegime.NOMINAL, resilience=1.0, attack_pressure=0.0, recovering=False
        )
        attacked = controller.compute(
            regime=SystemRegime.ATTACKED, resilience=0.3, attack_pressure=0.9, recovering=False
        )
        assert calm.threshold > attacked.threshold

    def test_low_resilience_tightens_within_a_regime(self) -> None:
        controller = AdaptiveThresholdController()
        healthy = controller.compute(
            regime=SystemRegime.STRESSED, resilience=1.0, attack_pressure=0.0, recovering=False
        )
        fragile = controller.compute(
            regime=SystemRegime.STRESSED, resilience=0.1, attack_pressure=0.0, recovering=False
        )
        assert healthy.threshold > fragile.threshold

    def test_recovery_relaxes_slightly_but_never_above_the_anchor(self) -> None:
        controller = AdaptiveThresholdController()
        decision = controller.compute(
            regime=SystemRegime.RECOVERING,
            resilience=1.0,
            attack_pressure=0.0,
            recovering=True,
        )
        assert decision.threshold <= REGIME_ANCHORS[SystemRegime.RECOVERING] + 0.05

    def test_threshold_stays_within_hard_bounds(self) -> None:
        controller = AdaptiveThresholdController()
        for regime in SystemRegime:
            for resilience in (0.0, 0.5, 1.0):
                for pressure in (0.0, 0.5, 1.0):
                    decision = controller.compute(
                        regime=regime,
                        resilience=resilience,
                        attack_pressure=pressure,
                        recovering=False,
                    )
                    assert THRESHOLD_FLOOR <= decision.threshold <= THRESHOLD_CEILING

    def test_unknown_regime_does_not_extend_nominal_trust(self) -> None:
        assert REGIME_ANCHORS[SystemRegime.UNKNOWN] < REGIME_ANCHORS[SystemRegime.NOMINAL]

    def test_fdr_feedback_needs_a_sample_before_it_acts(self) -> None:
        controller = AdaptiveThresholdController()
        for _ in range(5):
            controller.record_outcome(True)
        assert controller.trim == 0.0

    def test_excess_false_positives_raise_the_threshold(self) -> None:
        controller = AdaptiveThresholdController(fdr_target=0.05)
        for _ in range(64):
            controller.record_outcome(True)
        assert controller.trim > 0.0
        assert controller.observed_fdr == pytest.approx(1.0)

    def test_fdr_trim_is_bounded_so_it_cannot_override_the_regime(self) -> None:
        controller = AdaptiveThresholdController(fdr_target=0.05)
        for _ in range(5000):
            controller.record_outcome(True)
        assert controller.trim <= 0.08

    def test_rejects_a_degenerate_target(self) -> None:
        with pytest.raises(ValueError):
            AdaptiveThresholdController(fdr_target=0.0)

    def test_reset_clears_feedback(self) -> None:
        controller = AdaptiveThresholdController()
        for _ in range(64):
            controller.record_outcome(True)
        controller.reset()
        assert controller.trim == 0.0
        assert controller.observed_fdr == 0.0
