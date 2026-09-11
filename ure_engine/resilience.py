"""
resilience.py — the single executive metric.
============================================

One number, 0.0 to 1.0, that a dashboard can show and an executive can read:

    0.00  collapsed
    0.50  stressed
    1.00  highly resilient

The temptation with a metric like this is to make it a rebranded risk score.
That would be a mistake and the architecture says so explicitly: the target
replaces ``composite_risk_score`` with ``resilience_index``, and the two are not
inverses of each other. Risk asks "how bad is it right now". Resilience asks
"how much can this system absorb before it stops working" -- a question about
*capacity*, which depends on trajectory, learning, and how much capability has
already been spent on staying upright.

The five terms
--------------
``stability``   (weight 0.30) Energy headroom. How far from saturation.
``trajectory``  (weight 0.25) Whether the situation is improving or degrading,
                and how erratically.
``adaptation``  (weight 0.20) What BVE has learned. A system that recognizes
                what is hitting it is more resilient than one meeting it fresh.
``recovery``    (weight 0.15) How much capability remains unspent. A system
                holding steady only because it is quarantined scores low here.
``memory``      (weight 0.10) AMX depth, which cuts *both* ways -- see below.

Why memory cuts both ways
-------------------------
Having a rich attack memory is an asset: it is how the system recognizes
recurrence. Being under active recall right now is a liability: something the
system remembers is happening again. The memory term rewards the first and
penalizes the second, so an idle deployment with deep memory scores well, and
the same deployment mid-campaign does not.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from .lyapunov import ENERGY_MAX
from .regimes import RegimeProfile, SystemRegime
from .state_vector import clamp

__all__ = ["ResilienceBreakdown", "ResilienceIndex"]

#: Term weights. Must sum to 1.0; asserted at import.
WEIGHTS: Final[Mapping[str, float]] = {
    "stability": 0.30,
    "trajectory": 0.25,
    "adaptation": 0.20,
    "recovery": 0.15,
    "memory": 0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "resilience weights must sum to 1.0"

#: Ceiling imposed on the index by the dominant regime. A CASCADING system
#: cannot report as resilient no matter how the terms average out -- this is
#: the guard against a metric that looks reassuring during an outage.
_REGIME_CEILING: Final[dict[SystemRegime, float]] = {
    SystemRegime.NOMINAL: 1.00,
    SystemRegime.ADAPTING: 0.90,
    SystemRegime.RECOVERING: 0.75,
    SystemRegime.STRESSED: 0.60,
    SystemRegime.ATTACKED: 0.45,
    SystemRegime.CASCADING: 0.20,
    SystemRegime.UNKNOWN: 0.50,
}


@dataclass(frozen=True, slots=True)
class ResilienceBreakdown:
    """The index alongside every term that produced it.

    Reported in full because a single number nobody can decompose is a number
    nobody trusts. When the index drops, the breakdown says which term moved.
    """

    index: float
    stability: float
    trajectory: float
    adaptation: float
    recovery: float
    memory: float
    #: The regime ceiling that was applied, if any (else 1.0).
    ceiling: float
    #: True when the ceiling, not the weighted terms, determined the result.
    ceiling_binding: bool

    @property
    def band(self) -> str:
        """Coarse label for dashboards and alert routing."""
        if self.index >= 0.75:
            return "RESILIENT"
        if self.index >= 0.50:
            return "ADEQUATE"
        if self.index >= 0.25:
            return "FRAGILE"
        return "CRITICAL"

    @property
    def weakest_term(self) -> str:
        """Name of the term dragging the index down hardest.

        Weighted contribution, not raw value: a strong weight on a mediocre
        term can cost more than a weak weight on a terrible one, and the
        operator wants to know where the leverage is.
        """
        deficits = {
            name: (1.0 - getattr(self, name)) * weight for name, weight in WEIGHTS.items()
        }
        return max(deficits, key=lambda name: deficits[name])

    def to_dict(self) -> dict[str, object]:
        return {
            "index": round(self.index, 4),
            "band": self.band,
            "weakest_term": self.weakest_term,
            "terms": {
                "stability": round(self.stability, 4),
                "trajectory": round(self.trajectory, 4),
                "adaptation": round(self.adaptation, 4),
                "recovery": round(self.recovery, 4),
                "memory": round(self.memory, 4),
            },
            "ceiling": round(self.ceiling, 4),
            "ceiling_binding": self.ceiling_binding,
        }


class ResilienceIndex:
    """Computes the executive resilience metric from URE's subsystem outputs."""

    __slots__ = ("_weights",)

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        resolved = dict(WEIGHTS if weights is None else weights)
        missing = set(WEIGHTS) - set(resolved)
        if missing:
            raise ValueError(f"missing resilience weights: {sorted(missing)}")
        total = sum(resolved.values())
        if total <= 0:
            raise ValueError("resilience weights must sum to a positive value")
        # Normalize so callers can pass relative weights without doing the maths.
        self._weights = {name: value / total for name, value in resolved.items()}

    def compute(
        self,
        *,
        energy: float,
        derivative: float,
        profile: RegimeProfile,
        trajectory_stability: float,
        adaptation_score: float,
        recovery_score: float,
        memory_depth: int,
        memory_recall: float,
    ) -> ResilienceBreakdown:
        """Combine subsystem signals into the index.

        Every input is expected in its natural range; the method normalizes.
        """
        stability = clamp(1.0 - energy / ENERGY_MAX)

        # Trajectory: the smoothness score from the trajectory engine, adjusted
        # by direction. Rising energy erodes more than falling energy restores,
        # because degradation compounds and recovery does not.
        trajectory = clamp(trajectory_stability)
        if derivative > 0:
            trajectory = clamp(trajectory - clamp(derivative * 5.0, 0.0, 0.40))
        else:
            trajectory = clamp(trajectory + clamp(-derivative * 2.0, 0.0, 0.20))

        adaptation = clamp(adaptation_score)
        recovery = clamp(recovery_score)

        # Memory: breadth is an asset (saturating at 64 profiles), active
        # recall is a liability. Floor at zero rather than going negative so a
        # term can never flip the sign of the weighted sum.
        breadth = clamp(memory_depth / 64.0)
        memory = clamp(0.35 + 0.65 * breadth - clamp(memory_recall))

        weighted = (
            stability * self._weights["stability"]
            + trajectory * self._weights["trajectory"]
            + adaptation * self._weights["adaptation"]
            + recovery * self._weights["recovery"]
            + memory * self._weights["memory"]
        )

        ceiling = _REGIME_CEILING.get(profile.dominant, 1.0)
        index = min(clamp(weighted), ceiling)

        return ResilienceBreakdown(
            index=index,
            stability=stability,
            trajectory=trajectory,
            adaptation=adaptation,
            recovery=recovery,
            memory=memory,
            ceiling=ceiling,
            ceiling_binding=clamp(weighted) > ceiling,
        )
