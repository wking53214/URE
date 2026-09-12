"""
dwell.py, how long has this been going on.
==========================================

URE reasons about *change*. Energy, its derivative, velocity, acceleration,
volatility: every one of those is a statement about motion, and every one of
them reads zero for a quantity that is simply held constant.

That is a real hole, and an adversary can drive a truck through it. Hold a
hostile rate steady just under the boundary where affinity alone would name the
regime ATTACKED, and there is nothing left to detect: the level is too low, and
the derivative of a constant is zero. Red-teaming measured the edge at roughly
36% sustained hostile pressure, below which evasion was indefinite. Not
"detected late". Never detected, at any duration.

The missing quantity is exposure. A 30% hostile rate held for two observations
is noise; the same rate held for eighty is a campaign, and no amount of
differentiating will tell those apart. So this module integrates instead: it
measures how long hostile pressure has been meaningfully elevated, and reports
it as a scalar the classifier can treat as evidence.

Why it tracks hostile pressure and nothing else
-----------------------------------------------
Duration is dangerous evidence. Every real system spends long stretches under
sustained load, and a dwell term over *operational* pressure would convert
every busy afternoon into an incident. Legitimate load is supposed to persist.

Hostile pressure is not. A sustained threat or adversarial reading means
somebody is still pushing after the system has been rejecting them for a
hundred observations, which is exactly the signal that separates an adversary
from a customer. So dwell is computed over ``max(threat, adversarial)`` alone,
and a system that is merely overloaded accumulates none of it. That restriction
is the whole reason this can be added without turning URE into a false-positive
generator, and it is the first thing to check if it ever becomes one.

Duration, not magnitude
-----------------------
Above the floor the accumulator charges toward 1.0 at a fixed rate regardless
of how far above the floor the pressure sits. That is deliberate. Magnitude is
already fully represented: it drives the pressure vector, the energy, and the
affinity functions. Weighting dwell by magnitude too would double-count what
URE already sees well and would leave the low-and-patient adversary, the only
case this exists for, accumulating the most slowly.

Asymmetric time constants
-------------------------
Charging is slower than discharging is *not* the choice here; the reverse is.
``rise`` is shorter than ``fall`` so that evidence accumulates faster than it
is forgiven. An adversary who alternates pressure with quiet in order to keep
the accumulator drained has to spend more time quiet than hostile, which costs
them most of their throughput. Symmetric constants would make that free.
"""

from __future__ import annotations

import math

from .state_vector import StateVector, clamp

__all__ = ["HostileDwell"]

#: Hostile pressure below which nothing accumulates. Ordinary traffic carries a
#: nonzero block rate, and a floor of zero would have every healthy deployment
#: slowly charging toward ATTACKED. Set above realistic background rejection.
_DEFAULT_FLOOR: float = 0.12

#: Width of the soft edge above the floor. A hard cutoff would make behaviour
#: at the boundary a coin flip on the last decimal place of the smoother.
_DEFAULT_BAND: float = 0.12

#: Observations of continuous elevated pressure to reach ~63% charge.
_DEFAULT_RISE: float = 25.0

#: Observations of quiet to decay by the same factor. Longer than the rise on
#: purpose; see the module docstring.
_DEFAULT_FALL: float = 60.0

#: Guard against a pathological or replayed timestamp producing a single step
#: large enough to saturate or flush the accumulator in one observation.
_MAX_STEP: float = 300.0


class HostileDwell:
    """Leaky integrator over time spent under elevated hostile pressure.

    Reports a scalar in [0, 1]: 0.0 means no sustained hostile exposure, 1.0
    means the system has been under it for several multiples of ``rise``.
    """

    __slots__ = ("_band", "_fall", "_floor", "_last_timestamp", "_rise", "_value")

    def __init__(
        self,
        floor: float = _DEFAULT_FLOOR,
        band: float = _DEFAULT_BAND,
        rise: float = _DEFAULT_RISE,
        fall: float = _DEFAULT_FALL,
    ) -> None:
        if not 0.0 <= floor < 1.0:
            raise ValueError("floor must be in [0, 1)")
        if band <= 0.0:
            raise ValueError("band must be positive")
        if rise <= 0.0 or fall <= 0.0:
            raise ValueError("rise and fall must be positive")
        self._floor = floor
        self._band = band
        self._rise = rise
        self._fall = fall
        self._value = 0.0
        self._last_timestamp: float | None = None

    @property
    def value(self) -> float:
        """Current dwell, without advancing the accumulator."""
        return self._value

    def elevated(self, hostile: float) -> float:
        """Soft gate: 0 below the floor, 1 above floor + band, smooth between.

        Smoothstep rather than a linear ramp so the derivative is continuous at
        both ends, which keeps an adversary from finding a sharp corner to sit
        on.
        """
        x = clamp((hostile - self._floor) / self._band)
        return x * x * (3.0 - 2.0 * x)

    def update(self, state: StateVector, timestamp: float) -> float:
        """Advance the accumulator by one observation and return the new dwell."""
        hostile = max(state.threat_pressure, state.adversarial_pressure)
        if not math.isfinite(hostile) or not math.isfinite(timestamp):
            # A broken reading must not be able to charge or flush the
            # accumulator. Hold and let the caller's own NaN handling speak.
            return self._value

        previous = self._last_timestamp
        self._last_timestamp = timestamp
        # First observation establishes the clock only. Charging from a single
        # sample would let a cold-started engine inherit a dwell it never saw.
        dt = 1.0 if previous is None else min(_MAX_STEP, max(0.0, timestamp - previous))

        target = self.elevated(hostile)
        tau = self._rise if target > self._value else self._fall
        self._value = clamp(self._value + (target - self._value) * (1.0 - math.exp(-dt / tau)))
        return self._value

    def reset(self) -> None:
        self._value = 0.0
        self._last_timestamp = None
