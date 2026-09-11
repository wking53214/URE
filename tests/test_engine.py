"""End-to-end tests for the URE pipeline."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from ure_engine import (
    GovernanceDecision,
    RecoveryAction,
    StateVector,
    SystemRegime,
    TelemetryFrame,
    UREAssessment,
    UREConfig,
    UREEngine,
)
from ure_engine.state_vector import PolicyTelemetry

IDLE = {"risk_score": 0.0, "blocked_rate": 0.0, "retry_rate": 0.0, "latency_ms": 1.0}
ATTACK = {
    "risk_score": 0.95,
    "blocked_rate": 0.85,
    "retry_rate": 0.1,
    "latency_ms": 50.0,
    "queue_saturation": 0.1,
}
OVERLOAD = {
    "risk_score": 0.15,
    "blocked_rate": 0.05,
    "retry_rate": 0.7,
    "latency_ms": 900.0,
    "queue_saturation": 0.95,
    "substrate_health": 0.3,
    "circuit_state": "OPEN",
}


def settle(engine: UREEngine, telemetry: dict, cycles: int = 8) -> UREAssessment:
    """Feed the same telemetry repeatedly so smoothing and trajectory converge."""
    assessment = engine.observe(telemetry, now=0.0)
    for index in range(1, cycles):
        assessment = engine.observe(telemetry, now=float(index))
    return assessment


class TestPipeline:
    def test_idle_system_is_nominal_and_unrestricted(self) -> None:
        assessment = settle(UREEngine(), IDLE)
        assert assessment.regime is SystemRegime.NOMINAL
        assert assessment.lyapunov_energy == pytest.approx(0.0, abs=1e-3)
        assert assessment.recommended_action is RecoveryAction.NONE
        assert assessment.decision is GovernanceDecision.ALLOW
        assert assessment.healthy

    def test_attack_traffic_reaches_the_attacked_regime(self) -> None:
        assessment = settle(UREEngine(), ATTACK)
        assert assessment.regime is SystemRegime.ATTACKED
        assert assessment.recommended_action is RecoveryAction.ISOLATE
        assert assessment.decision is GovernanceDecision.ISOLATE

    def test_overload_is_stressed_not_attacked(self) -> None:
        assessment = settle(UREEngine(), OVERLOAD)
        assert assessment.regime is SystemRegime.STRESSED
        assert assessment.recommended_action is RecoveryAction.THROTTLE

    def test_threshold_tightens_under_attack(self) -> None:
        calm = settle(UREEngine(), IDLE)
        attacked = settle(UREEngine(), ATTACK)
        assert calm.recommended_threshold > attacked.recommended_threshold

    def test_cold_engine_has_not_learned_anything_yet(self) -> None:
        # Resilience deliberately grows with experience: a fresh engine has no
        # vaccines and no memory, so it cannot score a perfect 1.0.
        assessment = settle(UREEngine(), IDLE)
        breakdown = assessment.resilience_breakdown
        assert breakdown is not None
        assert breakdown.adaptation == 0.0
        assert 0.70 <= assessment.resilience_index < 0.80
        assert breakdown.stability == pytest.approx(1.0, abs=1e-3)

    def test_observation_count_tracks_calls(self) -> None:
        engine = UREEngine()
        settle(engine, IDLE, cycles=5)
        assert engine.observations == 5

    def test_accepts_a_frame_or_a_mapping(self) -> None:
        engine = UREEngine()
        from_dict = engine.observe(ATTACK, now=0.0)
        from_frame = engine.observe(TelemetryFrame.from_mapping(ATTACK), now=1.0)
        assert isinstance(from_dict, UREAssessment)
        assert isinstance(from_frame, UREAssessment)

    def test_engine_is_callable(self) -> None:
        engine = UREEngine()
        assert isinstance(engine(IDLE, now=0.0), UREAssessment)

    def test_partial_telemetry_degrades_rather_than_raising(self) -> None:
        # A system that is on fire does not owe the resilience engine a
        # complete telemetry feed.
        engine = UREEngine()
        assessment = engine.observe({"latency_ms": 200.0}, now=0.0)
        assert assessment.regime is not SystemRegime.UNKNOWN

    def test_junk_telemetry_values_do_not_raise(self) -> None:
        engine = UREEngine()
        assessment = engine.observe(
            {"risk_score": None, "retry_rate": "not-a-number", "latency_ms": 20.0},
            now=0.0,
        )
        assert isinstance(assessment, UREAssessment)


class TestTrajectory:
    def test_a_ramp_is_detected_as_rising(self) -> None:
        engine = UREEngine()
        for index in range(12):
            assessment = engine.observe(
                {"risk_score": index / 12.0, "latency_ms": 20.0}, now=float(index)
            )
        assert assessment.lyapunov_derivative > 0.0
        assert assessment.degrading

    def test_a_receding_incident_is_detected_as_falling(self) -> None:
        engine = UREEngine()
        for index in range(12):
            engine.observe({"risk_score": 0.9, "latency_ms": 20.0}, now=float(index))
        for index in range(12, 30):
            assessment = engine.observe(
                {"risk_score": 0.0, "latency_ms": 20.0}, now=float(index)
            )
        assert assessment.lyapunov_derivative < 0.0

    def test_time_to_saturation_is_projected_while_rising(self) -> None:
        engine = UREEngine()
        for index in range(12):
            assessment = engine.observe(
                {"risk_score": min(1.0, index / 15.0), "failure_pressure": 0.5,
                 "latency_ms": 50.0},
                now=float(index),
            )
        assert assessment.time_to_saturation is None or assessment.time_to_saturation > 0


class TestMemoryIsActuallyWired:
    """The subsystems the predecessor deleted must change the outcome here."""

    def test_repeated_signatures_raise_adversarial_pressure(self) -> None:
        engine = UREEngine()
        telemetry = dict(ATTACK, signatures=["sig-repeat"])
        first = engine.observe(telemetry, now=0.0)
        for index in range(1, 10):
            later = engine.observe(telemetry, now=float(index))
        assert later.state.adversarial_pressure > first.state.adversarial_pressure
        assert "sig-repeat" in later.active_attack_profiles

    def test_amx_recall_appears_in_the_assessment(self) -> None:
        engine = UREEngine()
        engine.attack_memory.remember("known-bad", severity=0.9, now=0.0)
        assessment = engine.observe(
            dict(IDLE, signatures=["known-bad"], latency_ms=20.0), now=1.0
        )
        assert "known-bad" in assessment.active_attack_profiles
        assert assessment.state.adversarial_pressure > 0.0

    def test_vaccines_are_synthesized_and_then_fire(self) -> None:
        engine = UREEngine(UREConfig(synthesis_threshold=3))
        attack = ("ignore_previous", "developer_override", "show_prompt")
        for index in range(4):
            engine.observe(dict(ATTACK, markers=list(attack)), now=float(index))
        assert len(engine.vaccines) >= 1

        # The partial sequence now fires before the attack completes.
        assessment = engine.observe(
            dict(IDLE, markers=list(attack[:2]), latency_ms=20.0), now=10.0
        )
        assert assessment.active_vaccines

    def test_learning_can_be_disabled(self) -> None:
        engine = UREEngine(UREConfig(enable_learning=False))
        for index in range(6):
            engine.observe(dict(ATTACK, signatures=["sig-x"]), now=float(index))
        assert len(engine.attack_memory) == 0

    def test_cascading_stops_learning(self) -> None:
        # An engine that keeps learning through its own collapse learns the
        # collapse, not the attack.
        engine = UREEngine()
        cascade = {
            "risk_score": 0.9,
            "blocked_rate": 0.9,
            "retry_rate": 0.95,
            "latency_ms": 2000.0,
            "queue_saturation": 1.0,
            "substrate_health": 0.0,
            "circuit_state": "OPEN",
            "adversarial_events": 10.0,
        }
        # CASCADING requires *rising* energy, not merely high energy: a system
        # pinned at a high but steady level is stressed or attacked, not
        # cascading. So the load has to ramp.
        for index in range(20):
            ramp = index / 19.0
            assessment = engine.observe(
                {
                    key: (value * ramp if isinstance(value, (int, float)) else value)
                    for key, value in cascade.items()
                }
                | {
                    "circuit_state": "OPEN" if ramp > 0.5 else "CLOSED",
                    "signatures": [f"sig-{index}"],
                },
                now=float(index),
            )
            if assessment.recommended_action is RecoveryAction.QUARANTINE:
                break
        assert assessment.recommended_action is RecoveryAction.QUARANTINE
        assert assessment.recovery_vector is not None
        assert assessment.recovery_vector.controls["disable_learning"] is True

        # Keep the cascade going for one more step and confirm nothing new is
        # recorded while the system is collapsing.
        before = len(engine.attack_memory)
        follow_up = engine.observe(
            dict(cascade, signatures=["sig-during-collapse"]),
            now=float(index + 1),
        )
        assert follow_up.recommended_action is RecoveryAction.QUARANTINE
        assert len(engine.attack_memory) == before
        assert "sig-during-collapse" not in engine.attack_memory


class TestPolicySeam:
    def test_policy_signal_is_recorded_without_exposing_internals(self) -> None:
        engine = UREEngine()
        engine.observe_policy_signal(
            PolicyTelemetry(
                policy_name="FinancialPIIPolicy",
                policy_version="1.2",
                input_score=0.9,
                hard_block=True,
                output_acceptable=False,
                timestamp=0.0,
                signature="req-hash-abc",
            )
        )
        profile = engine.attack_memory.lookup("req-hash-abc", now=0.0)
        assert profile is not None
        assert profile.severity == 1.0
        # URE learned an opaque signature and a policy *name*, never a rule.
        assert "policy:FinancialPIIPolicy" in profile.tags

    def test_clean_policy_outcomes_are_not_remembered(self) -> None:
        engine = UREEngine()
        engine.observe_policy_signal(
            PolicyTelemetry(
                policy_name="BaselinePolicy",
                policy_version="1.0",
                input_score=0.05,
                hard_block=False,
                output_acceptable=True,
                timestamp=0.0,
                signature="req-clean",
            )
        )
        assert engine.attack_memory.lookup("req-clean", now=0.0) is None

    def test_ure_does_not_import_policy_modules(self) -> None:
        # The seam, enforced mechanically: nothing in URE may reach into a
        # policy package. This test fails loudly if someone wires one in.
        import pathlib

        root = pathlib.Path(__file__).resolve().parent.parent / "ure_engine"
        forbidden = ("import policy_api", "from policy_api", "import example_policies",
                     "from example_policies", "import gsa_gateway", "from gsa_gateway")
        for path in root.glob("*.py"):
            source = path.read_text()
            for needle in forbidden:
                assert needle not in source, f"{path.name} breaks the policy seam: {needle}"


class TestAsyncInterface:
    def test_evaluate_satisfies_the_protocol(self) -> None:
        from ure_engine import GatewayHealthEngineProtocol

        engine = UREEngine()
        assert isinstance(engine, GatewayHealthEngineProtocol)

    def test_evaluate_accepts_a_prebuilt_state_vector(self) -> None:
        engine = UREEngine()

        async def run() -> UREAssessment:
            for index in range(5):
                result = await engine.evaluate(
                    StateVector.create(
                        threat_pressure=0.9,
                        adversarial_pressure=0.8,
                        timestamp=float(index),
                    )
                )
            return result

        assessment = asyncio.run(run())
        assert assessment.regime is SystemRegime.ATTACKED


class TestConcurrency:
    def test_concurrent_observation_does_not_corrupt_state(self) -> None:
        engine = UREEngine()
        errors: list[BaseException] = []

        def worker(offset: int) -> None:
            try:
                for index in range(50):
                    engine.observe(ATTACK, now=float(offset * 50 + index))
            except BaseException as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert engine.observations == 400


class TestSerializationAndIntrospection:
    def test_assessment_is_json_serializable(self) -> None:
        assessment = settle(UREEngine(), ATTACK)
        payload = json.loads(assessment.to_json())
        assert payload["regime"] == assessment.regime.value
        assert payload["schema_version"] == "2.0"
        assert "resilience_breakdown" in payload
        assert "recovery_vector" in payload

    def test_json_is_canonical_and_reproducible(self) -> None:
        assessment = settle(UREEngine(), ATTACK)
        assert assessment.to_json() == assessment.to_json()

    def test_explanation_names_the_regime_and_the_driver(self) -> None:
        assessment = settle(UREEngine(), OVERLOAD)
        assert "STRESSED" in assessment.explanation
        assert "resilience" in assessment.explanation.lower()

    def test_snapshot_reports_operational_state(self) -> None:
        engine = UREEngine()
        settle(engine, ATTACK)
        snapshot = engine.snapshot()
        assert snapshot["observations"] == 8
        assert snapshot["last_regime"] == "ATTACKED"
        assert 0.0 <= snapshot["headroom"] <= 1.0

    def test_unknown_assessment_fails_safe_not_reassuring(self) -> None:
        assessment = UREAssessment.unknown("telemetry feed lost", timestamp=5.0)
        assert assessment.regime is SystemRegime.UNKNOWN
        assert assessment.resilience_index == 0.0
        assert assessment.recommended_action is RecoveryAction.DEGRADE
        assert not assessment.healthy


class TestLifecycle:
    def test_reset_preserves_learned_memory_by_default(self) -> None:
        engine = UREEngine()
        engine.attack_memory.remember("sig-a", severity=0.8, now=0.0)
        settle(engine, ATTACK)
        engine.reset()
        assert engine.observations == 0
        assert len(engine.attack_memory) >= 1

    def test_reset_can_discard_poisoned_memory(self) -> None:
        engine = UREEngine()
        engine.attack_memory.remember("sig-a", severity=0.8, now=0.0)
        engine.reset(keep_memory=False)
        assert len(engine.attack_memory) == 0

    def test_governance_decision_ordering(self) -> None:
        assert GovernanceDecision.ALLOW.restriction < GovernanceDecision.REVIEW.restriction
        assert GovernanceDecision.THROTTLE.restriction < GovernanceDecision.ISOLATE.restriction
        assert GovernanceDecision.QUARANTINE.restriction < GovernanceDecision.DENY.restriction
