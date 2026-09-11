"""Tests for the pressure state space."""

from __future__ import annotations

import dataclasses
import math

import pytest

from ure_engine.state_vector import (
    PRESSURE_NAMES,
    PolicyTelemetry,
    StateVector,
    clamp,
    mean_state,
)


class TestClamp:
    def test_passes_through_in_range(self) -> None:
        assert clamp(0.5) == 0.5

    def test_clamps_both_ends(self) -> None:
        assert clamp(-3.0) == 0.0
        assert clamp(7.0) == 1.0

    def test_nan_collapses_to_floor(self) -> None:
        # NaN must not propagate: it silently defeats every comparison in the
        # regime classifier and produces untraceable UNKNOWN results.
        assert clamp(float("nan")) == 0.0

    def test_custom_bounds(self) -> None:
        assert clamp(0.5, 0.6, 0.9) == 0.6
        assert clamp(2.0, 0.6, 0.9) == 0.9


class TestStateVector:
    def test_create_clamps_every_dimension(self) -> None:
        state = StateVector.create(
            threat_pressure=5.0,
            latency_pressure=-2.0,
            failure_pressure=0.4,
            timestamp=10.0,
        )
        assert state.threat_pressure == 1.0
        assert state.latency_pressure == 0.0
        assert state.failure_pressure == 0.4
        assert state.timestamp == 10.0

    def test_zero_is_the_origin(self) -> None:
        state = StateVector.zero(timestamp=3.0)
        assert state.pressures == (0.0,) * 6
        assert state.magnitude == 0.0
        assert state.timestamp == 3.0

    def test_pressures_follow_canonical_order(self) -> None:
        state = StateVector(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, timestamp=0.0)
        assert state.pressures == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        for name, value in zip(PRESSURE_NAMES, state.pressures, strict=True):
            assert getattr(state, name) == value

    def test_is_iterable(self) -> None:
        state = StateVector(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, timestamp=0.0)
        assert list(state) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    def test_magnitude_is_euclidean(self) -> None:
        state = StateVector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, timestamp=0.0)
        assert state.magnitude == pytest.approx(1.0)
        full = StateVector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, timestamp=0.0)
        assert full.magnitude == pytest.approx(math.sqrt(6))

    def test_dominant_identifies_the_heaviest_dimension(self) -> None:
        state = StateVector(0.1, 0.2, 0.9, 0.0, 0.3, 0.1, timestamp=0.0)
        assert state.dominant == "failure_pressure"

    def test_dominant_breaks_ties_toward_security(self) -> None:
        # threat comes first in PRESSURE_NAMES, so an exact tie resolves there.
        state = StateVector(0.5, 0.5, 0.0, 0.0, 0.0, 0.0, timestamp=0.0)
        assert state.dominant == "threat_pressure"

    def test_distance_ignores_timestamp(self) -> None:
        a = StateVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, timestamp=0.0)
        b = StateVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, timestamp=9999.0)
        assert a.distance_to(b) == 0.0

    def test_blend_interpolates(self) -> None:
        a = StateVector.zero()
        b = StateVector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, timestamp=5.0)
        mid = a.blend(b, 0.5)
        assert all(p == pytest.approx(0.5) for p in mid.pressures)
        assert mid.timestamp == 5.0

    def test_blend_with_zero_weight_keeps_self(self) -> None:
        a = StateVector(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, timestamp=0.0)
        b = StateVector(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, timestamp=1.0)
        assert a.blend(b, 0.0).pressures == a.pressures

    def test_with_pressure_replaces_and_clamps(self) -> None:
        state = StateVector.zero().with_pressure("threat_pressure", 3.0)
        assert state.threat_pressure == 1.0

    def test_with_pressure_rejects_unknown_dimension(self) -> None:
        with pytest.raises(KeyError):
            StateVector.zero().with_pressure("vibes", 0.5)

    def test_raised_accumulates_and_saturates(self) -> None:
        state = StateVector.zero().raised("adversarial_pressure", 0.4)
        assert state.adversarial_pressure == pytest.approx(0.4)
        state = state.raised("adversarial_pressure", 0.9)
        assert state.adversarial_pressure == 1.0

    def test_round_trips_through_dict(self) -> None:
        original = StateVector(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, timestamp=12.5)
        restored = StateVector.from_mapping(original.to_dict())
        assert restored.pressures == pytest.approx(original.pressures)
        assert restored.timestamp == original.timestamp

    def test_is_immutable(self) -> None:
        state = StateVector.zero()
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            state.threat_pressure = 0.5  # type: ignore[misc]


class TestMeanState:
    def test_empty_returns_zero(self) -> None:
        assert mean_state([]).pressures == (0.0,) * 6

    def test_averages_component_wise(self) -> None:
        a = StateVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, timestamp=1.0)
        b = StateVector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, timestamp=2.0)
        result = mean_state([a, b])
        assert all(p == pytest.approx(0.5) for p in result.pressures)
        assert result.timestamp == 2.0


class TestPolicyTelemetry:
    def test_hard_block_is_adverse(self) -> None:
        telemetry = PolicyTelemetry(
            policy_name="p", policy_version="1", input_score=0.0,
            hard_block=True, output_acceptable=True, timestamp=0.0,
        )
        assert telemetry.adverse

    def test_unacceptable_output_is_adverse(self) -> None:
        telemetry = PolicyTelemetry(
            policy_name="p", policy_version="1", input_score=0.0,
            hard_block=False, output_acceptable=False, timestamp=0.0,
        )
        assert telemetry.adverse

    def test_high_score_is_adverse(self) -> None:
        telemetry = PolicyTelemetry(
            policy_name="p", policy_version="1", input_score=0.7,
            hard_block=False, output_acceptable=True, timestamp=0.0,
        )
        assert telemetry.adverse

    def test_clean_low_score_is_not_adverse(self) -> None:
        telemetry = PolicyTelemetry(
            policy_name="p", policy_version="1", input_score=0.1,
            hard_block=False, output_acceptable=True, timestamp=0.0,
        )
        assert not telemetry.adverse
