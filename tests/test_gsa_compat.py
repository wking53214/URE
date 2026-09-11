"""Tests for the legacy GSA interface and the governance adapter.

This file guards the contract that started the project. GSA's gateway contained
``from ure_engine import ClassificationResult, GatewayHealthEngine`` against a
module that did not exist. These tests assert that the import resolves, the
call signature matches, and every field GSA reads is present and sane.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from ure_engine import (
    ClassificationResult,
    GatewayHealthEngine,
    GovernanceDecision,
    GSATelemetryAdapter,
    StateVector,
    SystemRegime,
    UREEngine,
)
from ure_engine.compat import CompatGatewayHealthEngine, binary_entropy
from ure_engine.state_vector import PolicyTelemetry


class TestTheLegacyImport:
    def test_the_import_gsa_was_written_against_resolves(self) -> None:
        # The literal line from gsa_gateway.py:
        #     from ure_engine import ClassificationResult, GatewayHealthEngine
        assert ClassificationResult is not None
        assert GatewayHealthEngine is CompatGatewayHealthEngine

    def test_construction_signature_matches_the_original(self) -> None:
        engine = GatewayHealthEngine(backlog_capacity=256)
        assert engine.backlog_capacity == 256

    def test_assess_has_the_original_keyword_signature(self) -> None:
        signature = inspect.signature(GatewayHealthEngine.assess)
        assert list(signature.parameters)[1:] == [
            "reject_rate",
            "retry_rate",
            "latency_ms",
            "backlog",
        ]

    def test_every_field_gsa_reads_is_present(self) -> None:
        engine = GatewayHealthEngine()
        summary = engine.assess(
            reject_rate=0.2, retry_rate=0.1, latency_ms=50.0, backlog=10
        )
        assert isinstance(summary.result_status.value, str)
        assert isinstance(summary.dominant_regime.value, str)
        assert isinstance(summary.composite_risk_score, float)
        assert isinstance(summary.calculated_energy, float)
        assert isinstance(summary.regime_entropy, float)
        assert isinstance(summary.regime_confidence, float)
        assert isinstance(summary.rationale_statement, str)

    def test_status_values_are_from_the_legacy_vocabulary(self) -> None:
        engine = GatewayHealthEngine()
        for reject_rate in (0.0, 0.25, 0.5, 0.75, 1.0):
            summary = engine.assess(
                reject_rate=reject_rate, retry_rate=0.1, latency_ms=50.0, backlog=10
            )
            assert summary.result_status.value in {
                "NEUTRAL",
                "RISK_INCREASING",
                "REGRESSIVE",
            }

    def test_regime_values_are_from_the_modern_vocabulary(self) -> None:
        engine = GatewayHealthEngine()
        summary = engine.assess(
            reject_rate=0.9, retry_rate=0.1, latency_ms=50.0, backlog=10
        )
        assert summary.dominant_regime.value in {r.value for r in SystemRegime}


class TestTheDefectTheShimExistedToWorkAround:
    """The archive's shim documented a specific failure and its cause.

    "the regime never leaves NOMINAL even at a 60% block rate" -- because
    reject_rate was fed only into ``blocked_rate``, which carries 0.3 of the
    threat term. The adapter now drives threat pressure from it directly.
    """

    @staticmethod
    def sustained(reject_rate: float, cycles: int = 10) -> ClassificationResult:
        engine = GatewayHealthEngine(backlog_capacity=256)
        for _ in range(cycles):
            summary = engine.assess(
                reject_rate=reject_rate, retry_rate=0.1, latency_ms=50.0, backlog=20
            )
        return summary

    def test_sustained_sixty_percent_rejection_is_not_nominal(self) -> None:
        summary = self.sustained(0.6)
        assert summary.dominant_regime.value != "NOMINAL"
        assert summary.result_status.value != "NEUTRAL"

    def test_a_quiet_gateway_is_still_neutral(self) -> None:
        summary = self.sustained(0.0)
        assert summary.dominant_regime.value == "NOMINAL"
        assert summary.result_status.value == "NEUTRAL"

    def test_status_escalates_monotonically_with_rejection_rate(self) -> None:
        order = {"NEUTRAL": 0, "RISK_INCREASING": 1, "REGRESSIVE": 2}
        previous = -1
        for reject_rate in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95):
            level = order[self.sustained(reject_rate).result_status.value]
            assert level >= previous, f"status regressed at {reject_rate}"
            previous = level

    def test_composite_risk_rises_with_rejection_rate(self) -> None:
        low = self.sustained(0.1).composite_risk_score
        high = self.sustained(0.9).composite_risk_score
        assert high > low


class TestDerivedFields:
    def test_composite_risk_is_normalized(self) -> None:
        engine = GatewayHealthEngine()
        for reject_rate in (0.0, 0.5, 1.0):
            summary = engine.assess(
                reject_rate=reject_rate, retry_rate=0.9, latency_ms=2000.0, backlog=500
            )
            assert 0.0 <= summary.composite_risk_score <= 1.0

    def test_entropy_and_confidence_are_bounded(self) -> None:
        engine = GatewayHealthEngine()
        for reject_rate in (0.0, 0.3, 0.7, 1.0):
            summary = engine.assess(
                reject_rate=reject_rate, retry_rate=0.2, latency_ms=100.0, backlog=30
            )
            assert 0.0 <= summary.regime_entropy <= 1.0
            assert 0.0 <= summary.regime_confidence <= 1.0

    def test_entropy_is_a_real_measurement_not_a_placeholder(self) -> None:
        # The predecessor had this field but nothing to fill it with. The soft
        # classifier computes it for real, so an ambiguous case must show
        # visibly higher entropy than a clear one.
        engine = GatewayHealthEngine()
        for _ in range(10):
            clear = engine.assess(
                reject_rate=0.0, retry_rate=0.0, latency_ms=1.0, backlog=0
            )
        ambiguous = GatewayHealthEngine()
        for _ in range(10):
            mid = ambiguous.assess(
                reject_rate=0.45, retry_rate=0.3, latency_ms=400.0, backlog=120
            )
        assert mid.regime_entropy > clear.regime_entropy

    def test_the_shim_never_reports_calmer_than_the_engine(self) -> None:
        # The compatibility layer must not be a place where warnings get lost.
        engine = GatewayHealthEngine()
        order = {"NEUTRAL": 0, "RISK_INCREASING": 1, "REGRESSIVE": 2, "UNKNOWN": 1}
        for reject_rate in (0.0, 0.3, 0.6, 0.9):
            for _ in range(8):
                summary = engine.assess(
                    reject_rate=reject_rate, retry_rate=0.4, latency_ms=300.0, backlog=80
                )
            engine_view = SystemRegime(summary.dominant_regime.value).health_status
            assert order[summary.result_status.value] >= order[engine_view]

    def test_rationale_is_human_readable(self) -> None:
        engine = GatewayHealthEngine()
        summary = engine.assess(
            reject_rate=0.8, retry_rate=0.2, latency_ms=100.0, backlog=40
        )
        assert len(summary.rationale_statement) > 40
        assert summary.dominant_regime.value in summary.rationale_statement

    def test_result_is_json_safe(self) -> None:
        import json

        engine = GatewayHealthEngine()
        summary = engine.assess(
            reject_rate=0.5, retry_rate=0.1, latency_ms=50.0, backlog=10
        )
        payload = json.loads(json.dumps(summary.to_dict()))
        assert payload["result_status"] == summary.result_status.value


class TestMigrationPath:
    def test_the_modern_assessment_is_reachable_from_the_legacy_result(self) -> None:
        engine = GatewayHealthEngine()
        summary = engine.assess(
            reject_rate=0.9, retry_rate=0.1, latency_ms=50.0, backlog=10
        )
        assert summary.assessment is not None
        assert summary.assessment.resilience_index >= 0.0
        assert summary.assessment.recovery_vector is not None

    def test_the_underlying_engine_is_exposed(self) -> None:
        engine = GatewayHealthEngine()
        assert isinstance(engine.engine, UREEngine)

    def test_the_modern_classify_entry_point_works(self) -> None:
        engine = GatewayHealthEngine()
        assessment = engine.classify({"risk_score": 0.9, "latency_ms": 20.0})
        assert assessment.regime in set(SystemRegime)

    def test_the_async_entry_point_works(self) -> None:
        engine = GatewayHealthEngine()
        state = StateVector.create(threat_pressure=0.9, timestamp=1.0)
        assessment = asyncio.run(engine.evaluate(state))
        assert assessment.lyapunov_energy > 0.0

    def test_binary_entropy_is_retained_for_old_callers(self) -> None:
        assert binary_entropy(0.5) == pytest.approx(1.0)
        assert binary_entropy(1.0) == 0.0
        assert binary_entropy(0.0) == 0.0


class TestGSATelemetryAdapter:
    def test_the_archives_original_signature_is_preserved(self) -> None:
        adapter = GSATelemetryAdapter()
        state = adapter.build_state_vector(
            reject_rate=0.5, retry_rate=0.2, latency_ms=100.0, backlog=64, now=1.0
        )
        assert isinstance(state, StateVector)
        assert state.threat_pressure == pytest.approx(0.5)
        assert state.failure_pressure == pytest.approx(0.2)
        assert state.latency_pressure == pytest.approx(0.1)
        assert state.resource_pressure == pytest.approx(0.25)

    def test_backlog_is_normalized_against_capacity(self) -> None:
        adapter = GSATelemetryAdapter(backlog_capacity=100)
        assert adapter.build_state_vector(0, 0, 0, 50).resource_pressure == pytest.approx(0.5)
        # And saturates rather than exceeding the range.
        assert adapter.build_state_vector(0, 0, 0, 500).resource_pressure == 1.0

    def test_rejects_a_zero_capacity(self) -> None:
        with pytest.raises(ValueError):
            GSATelemetryAdapter(backlog_capacity=0)

    def test_build_frame_carries_signatures_markers_and_policy(self) -> None:
        adapter = GSATelemetryAdapter()
        policy = PolicyTelemetry(
            policy_name="BaselinePolicy",
            policy_version="1.0",
            input_score=0.4,
            hard_block=False,
            output_acceptable=True,
            timestamp=0.0,
        )
        frame = adapter.build_frame(
            reject_rate=0.3,
            signatures=["sig-a"],
            markers=["m1", "m2"],
            policy=policy,
        )
        assert frame.signatures == ("sig-a",)
        assert frame.markers == ("m1", "m2")
        assert frame.policy is policy


class TestGovernanceDecisions:
    @staticmethod
    def assess(telemetry: dict, cycles: int = 8):
        engine = UREEngine()
        for index in range(cycles):
            assessment = engine.observe(telemetry, now=float(index))
        return assessment

    def test_hard_block_overrides_everything(self) -> None:
        # Absolute rules stay absolute. No system state makes a raw SSN
        # acceptable, and no resilience reading relaxes it.
        calm = self.assess({"risk_score": 0.0, "latency_ms": 1.0})
        outcome = GSATelemetryAdapter.decide(calm, score=0.0, hard_block=True)
        assert outcome.decision is GovernanceDecision.DENY
        assert outcome.hard_blocked

    def test_a_low_score_on_a_healthy_system_is_allowed(self) -> None:
        calm = self.assess({"risk_score": 0.0, "latency_ms": 1.0})
        outcome = GSATelemetryAdapter.decide(calm, score=0.1)
        assert outcome.decision is GovernanceDecision.ALLOW
        assert outcome.allowed

    def test_a_near_threshold_score_is_flagged_for_review(self) -> None:
        calm = self.assess({"risk_score": 0.0, "latency_ms": 1.0})
        outcome = GSATelemetryAdapter.decide(
            calm, score=calm.recommended_threshold * 0.85
        )
        assert outcome.decision is GovernanceDecision.REVIEW

    def test_a_high_score_is_denied(self) -> None:
        calm = self.assess({"risk_score": 0.0, "latency_ms": 1.0})
        outcome = GSATelemetryAdapter.decide(calm, score=0.99)
        assert outcome.decision is GovernanceDecision.DENY

    def test_quarantine_governs_regardless_of_score(self) -> None:
        engine = UREEngine()
        for index in range(20):
            ramp = index / 19.0
            assessment = engine.observe(
                {
                    "risk_score": 0.9 * ramp,
                    "blocked_rate": 0.9 * ramp,
                    "retry_rate": 0.95 * ramp,
                    "latency_ms": 2000.0 * ramp,
                    "queue_saturation": ramp,
                    "substrate_health": 1.0 - ramp,
                    "circuit_state": "OPEN" if ramp > 0.5 else "CLOSED",
                    "adversarial_events": 10.0 * ramp,
                },
                now=float(index),
            )
        outcome = GSATelemetryAdapter.decide(assessment, score=0.0)
        assert outcome.decision is GovernanceDecision.QUARANTINE

    def test_the_same_score_gets_different_answers_in_different_regimes(self) -> None:
        # The central claim of the adaptive threshold: identical requests are
        # not identical decisions once the system's state is accounted for.
        calm = self.assess({"risk_score": 0.0, "latency_ms": 1.0})
        attacked = self.assess(
            {"risk_score": 0.95, "blocked_rate": 0.9, "latency_ms": 50.0}
        )
        score = 0.6
        calm_outcome = GSATelemetryAdapter.decide(calm, score=score)
        attacked_outcome = GSATelemetryAdapter.decide(attacked, score=score)
        assert calm_outcome.decision.restriction < attacked_outcome.decision.restriction

    def test_outcome_is_json_safe(self) -> None:
        import json

        calm = self.assess({"risk_score": 0.0, "latency_ms": 1.0})
        outcome = GSATelemetryAdapter.decide(calm, score=0.2)
        payload = json.loads(json.dumps(outcome.to_dict()))
        assert payload["decision"] == outcome.decision.value
