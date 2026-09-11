"""
lyapunov.py, system energy V(x) and its derivative dV/dt.
==========================================================

URE borrows the Lyapunov framing from control theory: define a scalar "energy"
over the state space that is zero at the desired equilibrium and positive
everywhere else, then watch its *derivative*. A system whose energy is high but
falling is recovering; a system whose energy is moderate but climbing is the
one about to fail. The trajectory signal is more informative than the
instantaneous level, which is the central design claim of the whole engine.

The recentering fix
-------------------
This is the single most important correction carried forward from the archive.

The original redesign computed::

    V(x) = 1.5 * sigmoid(sum of squared pressures)

``sigmoid(0) == 0.5``, so an idle system with zero pressure on all six
dimensions scored ``1.5 * 0.5 == 0.75`` energy. The regime bands placed NOMINAL
below 0.35 and RECOVERING below 0.90 on a falling trend, which meant **NOMINAL
was mathematically unreachable** and a healthy idle gateway reported as
STRESSED forever.

The fix is to remove the baseline rather than re-tune the bands::

    V(x) = 1.5 * tanh(sum of squared pressures / 2)

which is algebraically ``2 * (sigmoid(e) - 0.5)`` scaled the same way: exactly
0 at zero pressure, monotonically increasing, saturating toward 1.5. Every
regime becomes reachable, and ``tests/test_regimes.py`` asserts that property
over a full sweep of the state space so the bug cannot silently return.

Weighting
---------
Pressures are not equally dangerous. The weights below are security-biased:
adversarial and threat pressure count for more than latency or drift, because
a slow system is a nuisance and a compromised one is an incident.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from .state_vector import PRESSURE_NAMES, StateVector, clamp

__all__ = [
    "ENERGY_MAX",
    "ENERGY_WEIGHTS",
    "EnergySample",
    "LyapunovTrajectoryEngine",
    "compute_energy",
    "energy_contributions",
]

#: Per-pressure weights, in :data:`~ure_engine.state_vector.PRESSURE_NAMES` order:
#: threat, latency, failure, drift, adversarial, resource.
ENERGY_WEIGHTS: Final[tuple[float, ...]] = (1.4, 0.8, 1.3, 0.7, 1.5, 0.9)

#: Saturation ceiling of V(x). Energy approaches but never reaches this value.
ENERGY_MAX: Final[float] = 1.5


def compute_energy(state: StateVector, weights: Iterable[float] = ENERGY_WEIGHTS) -> float:
    """Lyapunov energy V(x): 0 at zero pressure, saturating toward :data:`ENERGY_MAX`.

    Squaring each pressure makes the function convex, so simultaneous moderate
    pressure on several dimensions scores lower than severe pressure on one --
    which is the correct bias for an engine whose job is to notice concentrated
    attacks rather than diffuse background load.
    """
    weight_tuple = tuple(weights)
    if len(weight_tuple) != len(PRESSURE_NAMES):
        raise ValueError(
            f"expected {len(PRESSURE_NAMES)} weights, got {len(weight_tuple)}"
        )
    quadratic = sum(w * p * p for w, p in zip(weight_tuple, state.pressures, strict=True))
    return ENERGY_MAX * math.tanh(quadratic / 2.0)


def energy_contributions(
    state: StateVector, weights: Iterable[float] = ENERGY_WEIGHTS
) -> dict[str, float]:
    """Fraction of the pre-saturation energy term owed to each pressure.

    The tanh saturation is monotone, so these shares explain *why* the energy
    is what it is even though they are taken before saturation. Returns all
    zeros for a zero state rather than dividing by zero.
    """
    weight_tuple = tuple(weights)
    terms = {
        name: w * p * p
        for name, w, p in zip(PRESSURE_NAMES, weight_tuple, state.pressures, strict=True)
    }
    total = sum(terms.values())
    if total <= 0.0:
        return {name: 0.0 for name in PRESSURE_NAMES}
    return {name: value / total for name, value in terms.items()}


@dataclass(frozen=True, slots=True)
class EnergySample:
    """One point on the energy trajectory."""

    timestamp: float
    energy: float


class LyapunovTrajectoryEngine:
    """Tracks V(x) over time and estimates dV/dt.

    Holds a bounded history of energy samples and reports the derivative as a
    least-squares slope over the recent window rather than a single backward
    difference. A two-point difference is dominated by sampling jitter at the
    millisecond timescales a gateway operates on; a regression over the window
    is what makes "rising" and "falling" mean something stable enough to gate a
    regime transition on.
    """

    __slots__ = ("_history", "_slope_window", "_window")

    def __init__(self, window: int = 64, slope_window: int = 8) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        if slope_window < 2:
            raise ValueError("slope_window must be at least 2")
        self._window = window
        self._slope_window = min(slope_window, window)
        self._history: deque[EnergySample] = deque(maxlen=window)

    # -- ingest -----------------------------------------------------------

    def observe(self, state: StateVector, weights: Iterable[float] = ENERGY_WEIGHTS) -> float:
        """Compute and record the energy of ``state``; returns V(x)."""
        energy = compute_energy(state, weights)
        self.record(state.timestamp, energy)
        return energy

    def record(self, timestamp: float, energy: float) -> None:
        """Record a pre-computed energy sample."""
        self._history.append(EnergySample(timestamp=timestamp, energy=energy))

    # -- read -------------------------------------------------------------

    @property
    def energy(self) -> float:
        """Most recent V(x), or 0.0 before any observation."""
        return self._history[-1].energy if self._history else 0.0

    @property
    def samples(self) -> tuple[EnergySample, ...]:
        """The retained energy history, oldest first."""
        return tuple(self._history)

    def __len__(self) -> int:
        return len(self._history)

    @property
    def derivative(self) -> float:
        """dV/dt estimated by least-squares slope over the recent window.

        Returns 0.0 with fewer than two samples, and 0.0 when every sample
        shares a timestamp (zero variance in t, so the slope is undefined
        rather than infinite).
        """
        recent = list(self._history)[-self._slope_window :]
        if len(recent) < 2:
            return 0.0

        n = float(len(recent))
        t0 = recent[0].timestamp
        # Re-base time at the window start: absolute epoch seconds are ~1e9 and
        # squaring them loses float precision in the covariance term.
        times = [s.timestamp - t0 for s in recent]
        energies = [s.energy for s in recent]

        mean_t = sum(times) / n
        mean_e = sum(energies) / n
        variance = sum((t - mean_t) ** 2 for t in times)
        if variance <= 1e-12:
            return 0.0
        covariance = sum((t - mean_t) * (e - mean_e) for t, e in zip(times, energies, strict=True))
        return covariance / variance

    def is_stable(self, tolerance: float = 0.02) -> bool:
        """True when energy is neither climbing nor falling beyond ``tolerance``."""
        return abs(self.derivative) <= tolerance

    def is_dissipating(self, tolerance: float = 0.02) -> bool:
        """True when energy is falling: the Lyapunov condition for recovery."""
        return self.derivative < -tolerance

    def is_diverging(self, tolerance: float = 0.02) -> bool:
        """True when energy is climbing: the precursor to regime escalation."""
        return self.derivative > tolerance

    def time_to_threshold(self, threshold: float) -> float | None:
        """Seconds until V(x) reaches ``threshold`` at the current slope.

        Returns ``None`` when the threshold is already crossed or the current
        trajectory never reaches it. This is URE's early-warning primitive: it
        turns "energy is rising" into "you have about 40 seconds", which is the
        form a recovery planner can actually act on.
        """
        slope = self.derivative
        current = self.energy
        if current >= threshold:
            return None
        if slope <= 1e-9:
            return None
        return (threshold - current) / slope

    def headroom(self) -> float:
        """Normalized distance from current energy to saturation, in [0, 1]."""
        return clamp(1.0 - self.energy / ENERGY_MAX)

    def reset(self) -> None:
        """Drop all history. Used when a deployment is intentionally restarted."""
        self._history.clear()
