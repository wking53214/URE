"""Tests for the telemetry projection layer."""

from __future__ import annotations

import pytest

from ure_engine.state_vector import StateVector
from ure_engine.telemetry import (
    SmoothingAdapter,
    TelemetryAdapter,
    TelemetryFrame,
)


class TestTelemetryFrame:
    def test_defaults_are_a_healthy_system(self) -> None:
        frame = TelemetryFrame()
        assert frame.substrate_health == 1.0
        assert frame.circuit_state == "CLOSED"
        assert frame.risk_score == 0.0

    def test_from_mapping_reads_known_keys(self) -> None:
        frame = TelemetryFrame.from_mapping(
            {"risk_score": 0.4, "latency_ms": 120.0, "circuit_state": "OPEN"}
        )
        assert frame.risk_score == 0.4
        assert frame.latency_ms == 120.0
        assert frame.circuit_state == "OPEN"

    def test_unknown_keys_are_preserved_for_audit_not_dropped(self) -> None:
        frame = TelemetryFrame.from_mapping({"risk_score": 0.1, "tenant": "acme"})
        assert frame.extra["tenant"] == "acme"

    def test_none_and_junk_coerce_to_safe_defaults(self) -> None:
        # Telemetry arrives from systems under stress; a None where a float was
        # expected is a symptom, not a reason to raise inside URE.
        frame = TelemetryFrame.from_mapping(
            {"risk_score": None, "retry_rate": "abc", "substrate_health": None}
        )
        assert frame.risk_score == 0.0
        assert frame.retry_rate == 0.0
        assert frame.substrate_health == 1.0

    def test_nan_coerces_to_the_default(self) -> None:
        frame = TelemetryFrame.from_mapping({"risk_score": float("nan")})
        assert frame.risk_score == 0.0

    def test_signatures_and_markers_normalize_to_tuples_of_str(self) -> None:
        frame = TelemetryFrame.from_mapping({"signatures": ["a", 1], "markers": ("m",)})
        assert frame.signatures == ("a", "1")
        assert frame.markers == ("m",)

    def test_missing_sequences_default_to_empty(self) -> None:
        frame = TelemetryFrame.from_mapping({"signatures": None})
        assert frame.signatures == ()


class TestTelemetryAdapter:
    @pytest.fixture()
    def adapter(self) -> TelemetryAdapter:
        return TelemetryAdapter()

    def test_an_empty_frame_maps_to_the_origin(self, adapter: TelemetryAdapter) -> None:
        state = adapter.to_state_vector(TelemetryFrame(), 0.0)
        assert state.pressures == (0.0,) * 6

    def test_risk_dominates_the_threat_term(self, adapter: TelemetryAdapter) -> None:
        state = adapter.to_state_vector(
            TelemetryFrame(risk_score=1.0, blocked_rate=0.0), 0.0
        )
        assert state.threat_pressure == pytest.approx(0.7)

    def test_block_rate_contributes_to_threat(self, adapter: TelemetryAdapter) -> None:
        state = adapter.to_state_vector(
            TelemetryFrame(risk_score=0.0, blocked_rate=1.0), 0.0
        )
        assert state.threat_pressure == pytest.approx(0.3)

    def test_latency_is_measured_against_the_configured_ceiling(self) -> None:
        adapter = TelemetryAdapter(latency_ceiling_ms=500.0)
        state = adapter.to_state_vector(TelemetryFrame(latency_ms=250.0), 0.0)
        assert state.latency_pressure == pytest.approx(0.5)

    def test_latency_volatility_adds_to_the_mean(self) -> None:
        # A system with acceptable mean latency and wild variance is worse off
        # than its mean suggests, and the mean is what hides that.
        adapter = TelemetryAdapter()
        steady = adapter.to_state_vector(TelemetryFrame(latency_ms=200.0), 0.0)
        jittery = adapter.to_state_vector(
            TelemetryFrame(latency_ms=200.0, latency_volatility=0.3), 0.0
        )
        assert jittery.latency_pressure > steady.latency_pressure

    def test_open_circuit_is_a_discrete_step_not_a_rate(self, adapter: TelemetryAdapter) -> None:
        closed = adapter.to_state_vector(TelemetryFrame(circuit_state="CLOSED"), 0.0)
        half = adapter.to_state_vector(TelemetryFrame(circuit_state="HALF_OPEN"), 0.0)
        open_ = adapter.to_state_vector(TelemetryFrame(circuit_state="OPEN"), 0.0)
        assert closed.failure_pressure == 0.0
        assert half.failure_pressure == pytest.approx(0.2)
        assert open_.failure_pressure == pytest.approx(0.5)

    def test_circuit_state_is_case_insensitive(self, adapter: TelemetryAdapter) -> None:
        state = adapter.to_state_vector(TelemetryFrame(circuit_state="open"), 0.0)
        assert state.failure_pressure == pytest.approx(0.5)

    def test_substrate_degradation_raises_resource_pressure(
        self, adapter: TelemetryAdapter
    ) -> None:
        state = adapter.to_state_vector(TelemetryFrame(substrate_health=0.4), 0.0)
        assert state.resource_pressure == pytest.approx(0.6)

    def test_adversarial_events_normalize_against_the_scale(self) -> None:
        adapter = TelemetryAdapter(adversarial_scale=20.0)
        state = adapter.to_state_vector(TelemetryFrame(adversarial_events=10.0), 0.0)
        assert state.adversarial_pressure == pytest.approx(0.5)

    def test_every_dimension_saturates_rather_than_overflowing(
        self, adapter: TelemetryAdapter
    ) -> None:
        # The units problem the predecessor had: unbounded counts summed with
        # bounded rates. Nothing here may exceed 1.0.
        state = adapter.to_state_vector(
            TelemetryFrame(
                risk_score=9.0,
                blocked_rate=9.0,
                retry_rate=9.0,
                latency_ms=1_000_000.0,
                queue_saturation=9.0,
                substrate_health=-5.0,
                adversarial_events=9_999.0,
                drift_metric=9.0,
                circuit_state="OPEN",
            ),
            0.0,
        )
        assert all(0.0 <= p <= 1.0 for p in state.pressures)

    def test_rejects_degenerate_configuration(self) -> None:
        with pytest.raises(ValueError):
            TelemetryAdapter(latency_ceiling_ms=0.0)
        with pytest.raises(ValueError):
            TelemetryAdapter(adversarial_scale=0.0)

    def test_timestamp_is_carried_through(self, adapter: TelemetryAdapter) -> None:
        assert adapter.to_state_vector(TelemetryFrame(), 42.0).timestamp == 42.0


class TestSmoothingAdapter:
    def test_first_observation_passes_through_unsmoothed(self) -> None:
        adapter = SmoothingAdapter(TelemetryAdapter(), alpha=0.4)
        state = adapter.to_state_vector(TelemetryFrame(risk_score=1.0), 0.0)
        assert state.threat_pressure == pytest.approx(0.7)

    def test_a_single_spike_is_damped(self) -> None:
        # One pathological request should move latency pressure, not define it.
        adapter = SmoothingAdapter(TelemetryAdapter(), alpha=0.4)
        for index in range(10):
            adapter.to_state_vector(TelemetryFrame(latency_ms=10.0), float(index))
        spiked = adapter.to_state_vector(TelemetryFrame(latency_ms=900.0), 10.0)
        assert spiked.latency_pressure < 0.5

    def test_sustained_change_is_eventually_tracked(self) -> None:
        adapter = SmoothingAdapter(TelemetryAdapter(), alpha=0.4)
        for index in range(20):
            state = adapter.to_state_vector(TelemetryFrame(risk_score=1.0), float(index))
        assert state.threat_pressure == pytest.approx(0.7, abs=0.01)

    def test_alpha_of_one_disables_smoothing(self) -> None:
        adapter = SmoothingAdapter(TelemetryAdapter(), alpha=1.0)
        adapter.to_state_vector(TelemetryFrame(risk_score=0.0), 0.0)
        state = adapter.to_state_vector(TelemetryFrame(risk_score=1.0), 1.0)
        assert state.threat_pressure == pytest.approx(0.7)

    def test_rejects_an_out_of_range_alpha(self) -> None:
        with pytest.raises(ValueError):
            SmoothingAdapter(TelemetryAdapter(), alpha=0.0)
        with pytest.raises(ValueError):
            SmoothingAdapter(TelemetryAdapter(), alpha=1.5)

    def test_reset_clears_the_running_average(self) -> None:
        adapter = SmoothingAdapter(TelemetryAdapter(), alpha=0.2)
        for index in range(10):
            adapter.to_state_vector(TelemetryFrame(risk_score=1.0), float(index))
        adapter.reset()
        state = adapter.to_state_vector(TelemetryFrame(risk_score=0.0), 20.0)
        assert state.threat_pressure == 0.0

    def test_wraps_any_source(self) -> None:
        class Constant:
            def to_state_vector(self, frame: TelemetryFrame, timestamp: float) -> StateVector:
                return StateVector.create(threat_pressure=0.5, timestamp=timestamp)

        adapter = SmoothingAdapter(Constant(), alpha=0.5)
        assert adapter.to_state_vector(TelemetryFrame(), 0.0).threat_pressure == 0.5
