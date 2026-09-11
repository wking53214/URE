"""Tests for Lyapunov energy and its derivative.

The recentering test in this file is the most important assertion in the
repository. See the module docstring of ``ure_engine.lyapunov``: an energy
function with a non-zero floor makes NOMINAL unreachable and reports every
healthy idle system as stressed.
"""

from __future__ import annotations

import math

import pytest

from ure_engine.lyapunov import (
    ENERGY_MAX,
    ENERGY_WEIGHTS,
    LyapunovTrajectoryEngine,
    compute_energy,
    energy_contributions,
)
from ure_engine.state_vector import PRESSURE_NAMES, StateVector


class TestComputeEnergy:
    def test_zero_state_has_exactly_zero_energy(self) -> None:
        # THE regression test. The predecessor used 1.5*sigmoid(x), whose floor
        # is 0.75 at zero pressure. That single constant made NOMINAL and
        # RECOVERING mathematically unreachable.
        assert compute_energy(StateVector.zero()) == 0.0

    def test_energy_is_bounded_by_the_ceiling(self) -> None:
        saturated = StateVector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, timestamp=0.0)
        energy = compute_energy(saturated)
        assert 0.0 < energy < ENERGY_MAX

    def test_energy_is_monotone_in_each_dimension(self) -> None:
        for name in PRESSURE_NAMES:
            previous = -1.0
            for level in (0.0, 0.25, 0.5, 0.75, 1.0):
                state = StateVector.zero().with_pressure(name, level)
                energy = compute_energy(state)
                assert energy > previous, f"{name} not monotone at {level}"
                previous = energy

    def test_security_dimensions_weigh_more_than_operational_ones(self) -> None:
        # The weighting is a deliberate policy choice: a compromised system is
        # worse than a slow one. Assert it so a weight edit is a visible change.
        adversarial = compute_energy(
            StateVector.zero().with_pressure("adversarial_pressure", 0.8)
        )
        latency = compute_energy(
            StateVector.zero().with_pressure("latency_pressure", 0.8)
        )
        assert adversarial > latency

    def test_convexity_favours_concentrated_pressure(self) -> None:
        # Squaring makes one severe pressure outweigh several moderate ones of
        # the same total. That bias is what makes the engine notice attacks
        # rather than averaging them into background load.
        concentrated = StateVector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, timestamp=0.0)
        diffuse = StateVector(0.25, 0.25, 0.25, 0.25, 0.0, 0.0, timestamp=0.0)
        assert compute_energy(concentrated) > compute_energy(diffuse)

    def test_rejects_wrong_weight_count(self) -> None:
        with pytest.raises(ValueError):
            compute_energy(StateVector.zero(), weights=(1.0, 1.0))

    def test_matches_the_closed_form(self) -> None:
        state = StateVector(0.5, 0.4, 0.3, 0.2, 0.1, 0.6, timestamp=0.0)
        expected_quadratic = sum(
            w * p * p for w, p in zip(ENERGY_WEIGHTS, state.pressures, strict=True)
        )
        assert compute_energy(state) == pytest.approx(
            ENERGY_MAX * math.tanh(expected_quadratic / 2.0)
        )


class TestEnergyContributions:
    def test_zero_state_contributes_nothing_without_dividing_by_zero(self) -> None:
        shares = energy_contributions(StateVector.zero())
        assert set(shares) == set(PRESSURE_NAMES)
        assert all(value == 0.0 for value in shares.values())

    def test_shares_sum_to_one(self) -> None:
        state = StateVector(0.5, 0.2, 0.7, 0.1, 0.3, 0.4, timestamp=0.0)
        assert sum(energy_contributions(state).values()) == pytest.approx(1.0)

    def test_identifies_the_driving_dimension(self) -> None:
        state = StateVector.zero().with_pressure("failure_pressure", 0.9)
        shares = energy_contributions(state)
        assert shares["failure_pressure"] == pytest.approx(1.0)


class TestLyapunovTrajectoryEngine:
    def test_starts_empty(self) -> None:
        engine = LyapunovTrajectoryEngine()
        assert len(engine) == 0
        assert engine.energy == 0.0
        assert engine.derivative == 0.0

    def test_rejects_degenerate_windows(self) -> None:
        with pytest.raises(ValueError):
            LyapunovTrajectoryEngine(window=1)
        with pytest.raises(ValueError):
            LyapunovTrajectoryEngine(slope_window=1)

    def test_observe_records_and_returns_energy(self) -> None:
        engine = LyapunovTrajectoryEngine()
        state = StateVector.zero().with_pressure("threat_pressure", 0.5)
        energy = engine.observe(state)
        assert energy == compute_energy(state)
        assert engine.energy == energy
        assert len(engine) == 1

    def test_single_sample_has_no_derivative(self) -> None:
        engine = LyapunovTrajectoryEngine()
        engine.record(0.0, 0.5)
        assert engine.derivative == 0.0

    def test_detects_a_rising_trajectory(self) -> None:
        engine = LyapunovTrajectoryEngine()
        for index in range(8):
            engine.record(float(index), 0.1 * index)
        assert engine.derivative == pytest.approx(0.1, abs=1e-6)
        assert engine.is_diverging()
        assert not engine.is_dissipating()

    def test_detects_a_falling_trajectory(self) -> None:
        engine = LyapunovTrajectoryEngine()
        for index in range(8):
            engine.record(float(index), 1.0 - 0.1 * index)
        assert engine.derivative == pytest.approx(-0.1, abs=1e-6)
        assert engine.is_dissipating()

    def test_flat_trajectory_is_stable(self) -> None:
        engine = LyapunovTrajectoryEngine()
        for index in range(8):
            engine.record(float(index), 0.4)
        assert engine.derivative == pytest.approx(0.0)
        assert engine.is_stable()

    def test_identical_timestamps_do_not_divide_by_zero(self) -> None:
        engine = LyapunovTrajectoryEngine()
        for _ in range(5):
            engine.record(100.0, 0.5)
        assert engine.derivative == 0.0

    def test_slope_is_robust_to_a_single_outlier(self) -> None:
        # Least-squares over the window rather than a two-point difference:
        # one spike must not flip the reported direction of travel.
        engine = LyapunovTrajectoryEngine(slope_window=8)
        for index in range(7):
            engine.record(float(index), 0.1 * index)
        engine.record(7.0, 0.0)  # outlier dip
        assert engine.derivative > 0.0

    def test_uses_epoch_scale_timestamps_without_precision_loss(self) -> None:
        # Timestamps are ~1.7e9; squaring them directly destroys the covariance
        # term in float64. The engine re-bases time at the window start.
        engine = LyapunovTrajectoryEngine()
        base = 1_700_000_000.0
        for index in range(8):
            engine.record(base + index, 0.1 * index)
        assert engine.derivative == pytest.approx(0.1, abs=1e-4)

    def test_window_is_bounded(self) -> None:
        engine = LyapunovTrajectoryEngine(window=4)
        for index in range(20):
            engine.record(float(index), 0.5)
        assert len(engine) == 4

    def test_time_to_threshold_projects_forward(self) -> None:
        engine = LyapunovTrajectoryEngine()
        for index in range(8):
            engine.record(float(index), 0.1 * index)  # ends at 0.7, +0.1/s
        eta = engine.time_to_threshold(1.0)
        assert eta is not None
        assert eta == pytest.approx(3.0, abs=0.2)

    def test_time_to_threshold_is_none_when_already_crossed(self) -> None:
        engine = LyapunovTrajectoryEngine()
        for index in range(4):
            engine.record(float(index), 1.2)
        assert engine.time_to_threshold(1.0) is None

    def test_time_to_threshold_is_none_when_not_rising(self) -> None:
        engine = LyapunovTrajectoryEngine()
        for index in range(8):
            engine.record(float(index), 0.5 - 0.01 * index)
        assert engine.time_to_threshold(1.0) is None

    def test_headroom_is_full_at_rest_and_empty_at_saturation(self) -> None:
        engine = LyapunovTrajectoryEngine()
        engine.record(0.0, 0.0)
        assert engine.headroom() == pytest.approx(1.0)
        engine.record(1.0, ENERGY_MAX)
        assert engine.headroom() == pytest.approx(0.0)

    def test_reset_clears_history(self) -> None:
        engine = LyapunovTrajectoryEngine()
        for index in range(5):
            engine.record(float(index), 0.5)
        engine.reset()
        assert len(engine) == 0
        assert engine.energy == 0.0
