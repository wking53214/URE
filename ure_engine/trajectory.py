"""
trajectory.py — velocity, acceleration, and volatility of the state.
====================================================================

The Lyapunov engine answers "is energy rising?". The trajectory engine answers
"how fast, is it speeding up, and how erratic has it been?". Those three
questions separate cases the energy derivative alone conflates:

* steady climb under real load           -> rising, low acceleration, low volatility
* an attack ramping                      -> rising, *positive* acceleration
* a system oscillating near a threshold  -> near-zero slope, high volatility

The third case is the one static thresholds miss entirely. A system flapping
between 0.55 and 0.75 energy has a mean slope of roughly zero and will be
reported as stable by any derivative-only check, while an operator watching the
graph would immediately call it unhealthy. Volatility is what makes URE's
ADAPTING regime distinguishable from genuine NOMINAL.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from itertools import pairwise

from .state_vector import StateVector, clamp

__all__ = ["TrajectoryEngine", "TrajectorySnapshot"]

#: Minimum dt used in finite differences. Two samples inside the same
#: millisecond would otherwise divide by ~0 and report an infinite velocity.
_MIN_DT: float = 1e-3


@dataclass(frozen=True, slots=True)
class TrajectorySnapshot:
    """Derived motion of the system through its state space."""

    energy: float
    #: First derivative of energy with respect to time.
    velocity: float
    #: Second derivative: is the rate of change itself increasing?
    acceleration: float
    #: Standard deviation of energy across the retained window.
    volatility: float
    #: Mean absolute step size; a scale-free measure of jitter.
    jitter: float
    #: Number of sign changes in velocity across the window.
    oscillations: int

    @property
    def escalating(self) -> bool:
        """Rising *and* speeding up: the signature of a developing cascade."""
        return self.velocity > 0.0 and self.acceleration > 0.0

    @property
    def settling(self) -> bool:
        """Falling and decelerating toward rest: a clean recovery."""
        return self.velocity < 0.0 and self.acceleration >= 0.0

    @property
    def unstable(self) -> bool:
        """Flapping: enough direction reversals that the mean slope is a lie."""
        return self.oscillations >= 3 and self.volatility > 0.05

    def to_dict(self) -> dict[str, float | int]:
        return {
            "energy": round(self.energy, 6),
            "velocity": round(self.velocity, 6),
            "acceleration": round(self.acceleration, 6),
            "volatility": round(self.volatility, 6),
            "jitter": round(self.jitter, 6),
            "oscillations": self.oscillations,
        }

    @classmethod
    def at_rest(cls, energy: float = 0.0) -> TrajectorySnapshot:
        """A snapshot with no motion, for cold start."""
        return cls(
            energy=energy,
            velocity=0.0,
            acceleration=0.0,
            volatility=0.0,
            jitter=0.0,
            oscillations=0,
        )


class TrajectoryEngine:
    """Rolling-window motion tracker over (timestamp, energy) samples.

    Kept deliberately separate from :class:`~ure_engine.lyapunov.LyapunovTrajectoryEngine`
    even though both consume the same samples: the Lyapunov engine produces the
    smoothed regression slope used for regime *decisions*, while this engine
    produces the raw finite-difference motion used for *explanation* and for
    detecting oscillation. Conflating them was a defect in the predecessor,
    where trend was computed twice by two code paths that could disagree.
    """

    __slots__ = ("_history", "_state_history", "_window")

    def __init__(self, window: int = 64) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        self._window = window
        self._history: deque[tuple[float, float]] = deque(maxlen=window)
        self._state_history: deque[StateVector] = deque(maxlen=window)

    def update(
        self, timestamp: float, energy: float, state: StateVector | None = None
    ) -> TrajectorySnapshot:
        """Record a sample and return the motion it implies."""
        self._history.append((timestamp, energy))
        if state is not None:
            self._state_history.append(state)

        if len(self._history) < 2:
            return TrajectorySnapshot.at_rest(energy)

        t_prev, e_prev = self._history[-2]
        dt = max(_MIN_DT, timestamp - t_prev)
        velocity = (energy - e_prev) / dt

        acceleration = 0.0
        if len(self._history) >= 3:
            t_prev2, e_prev2 = self._history[-3]
            prev_velocity = (e_prev - e_prev2) / max(_MIN_DT, t_prev - t_prev2)
            acceleration = (velocity - prev_velocity) / dt

        energies = [e for _, e in self._history]
        mean = sum(energies) / len(energies)
        volatility = math.sqrt(sum((e - mean) ** 2 for e in energies) / len(energies))

        steps = [abs(b - a) for a, b in pairwise(energies)]
        jitter = sum(steps) / len(steps) if steps else 0.0

        return TrajectorySnapshot(
            energy=energy,
            velocity=velocity,
            acceleration=acceleration,
            volatility=volatility,
            jitter=jitter,
            oscillations=self._count_oscillations(energies),
        )

    @staticmethod
    def _count_oscillations(energies: list[float]) -> int:
        """Count sign changes in the first difference, ignoring flat steps.

        Flat steps are skipped rather than counted as a direction: an idle
        system producing identical energies would otherwise register a
        reversal on every sample and be reported as violently unstable.
        """
        direction = 0
        reversals = 0
        for previous, current in pairwise(energies):
            delta = current - previous
            if abs(delta) < 1e-9:
                continue
            sign = 1 if delta > 0 else -1
            if direction != 0 and sign != direction:
                reversals += 1
            direction = sign
        return reversals

    # -- state-space views ------------------------------------------------

    @property
    def depth(self) -> int:
        """How many samples are currently retained."""
        return len(self._history)

    def drift(self) -> float:
        """Distance travelled in pressure space between the window's ends.

        Complements energy drift: two states can share an energy value while
        sitting in entirely different corners of the pressure space, and a
        system that has migrated across the space has changed character even
        if its scalar energy has not.
        """
        if len(self._state_history) < 2:
            return 0.0
        return self._state_history[0].distance_to(self._state_history[-1])

    def stability_score(self) -> float:
        """A [0, 1] score where 1 means motionless and 0 means violently unstable.

        Used as the "trajectory" term of the resilience index.
        """
        if len(self._history) < 2:
            return 1.0
        energies = [e for _, e in self._history]
        mean = sum(energies) / len(energies)
        volatility = math.sqrt(sum((e - mean) ** 2 for e in energies) / len(energies))
        reversals = self._count_oscillations(energies)
        reversal_penalty = clamp(reversals / max(1, len(energies) - 1))
        return clamp(1.0 - clamp(volatility * 4.0) * 0.7 - reversal_penalty * 0.3)

    def reset(self) -> None:
        self._history.clear()
        self._state_history.clear()
