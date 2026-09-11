"""
state_vector.py — the six-dimensional pressure state URE reasons over.
======================================================================

URE does not reason about requests, policies, or domain rules. It reasons
about *pressure*: six normalized [0, 1] scalars that any system can be
projected onto. That projection is the only thing URE consumes, which is
what keeps the policy seam intact (see ``governance_adapter``).

Why pressures rather than raw counters
--------------------------------------
The predecessor engine (UGPIS-Omega's ``SystemMetricsTelemetry``) mixed rates
(``dropout_rate``, 0..1) with unbounded counts (``backlog_depth``, 0..N) in a
single weighted sum. The count dimension then dominated the energy term at
arbitrary scale, and the weights had to be re-tuned per deployment. Normalizing
every dimension to [0, 1] up front removes the scale problem and makes the
energy weights portable.

The six dimensions
------------------
``threat_pressure``
    Hostile *intent* observed right now: policy risk scores, block rates.
``latency_pressure``
    Time-domain strain: absolute latency plus its volatility.
``failure_pressure``
    Things breaking: retries, open circuits, error rates.
``drift_pressure``
    Divergence from the expected operating distribution.
``adversarial_pressure``
    Hostile *capability* accumulated over time: AMX/BVE memory, not this
    request's score. Distinct from ``threat_pressure`` on purpose -- a single
    nasty request is threat; a recognized recurring campaign is adversarial.
``resource_pressure``
    Substrate exhaustion: queue saturation, degraded health, capacity loss.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

__all__ = [
    "PRESSURE_NAMES",
    "PolicyTelemetry",
    "StateVector",
    "clamp",
]


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp ``value`` into ``[lower, upper]``, mapping NaN to ``lower``.

    NaN handling matters: telemetry arithmetic upstream (a rate computed as
    ``0 / 0``) can produce NaN, and NaN propagates silently through every
    comparison in the regime classifier, making the engine report UNKNOWN for
    reasons no operator can trace. Collapsing it to the floor here fails
    visibly at the boundary instead.
    """
    if math.isnan(value):
        return lower
    return max(lower, min(upper, value))


#: Canonical ordering of the pressure dimensions. Every weight tuple, energy
#: computation, and serialization in URE uses this order.
PRESSURE_NAMES: Final[tuple[str, ...]] = (
    "threat_pressure",
    "latency_pressure",
    "failure_pressure",
    "drift_pressure",
    "adversarial_pressure",
    "resource_pressure",
)


@dataclass(frozen=True, slots=True)
class StateVector:
    """An immutable snapshot of system pressure at a point in time.

    All six pressures are normalized to [0, 1]. Construct through
    :meth:`create` (which clamps) rather than the raw constructor when the
    inputs come from outside URE.
    """

    threat_pressure: float
    latency_pressure: float
    failure_pressure: float
    drift_pressure: float
    adversarial_pressure: float
    resource_pressure: float
    timestamp: float

    # -- construction -----------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        threat_pressure: float = 0.0,
        latency_pressure: float = 0.0,
        failure_pressure: float = 0.0,
        drift_pressure: float = 0.0,
        adversarial_pressure: float = 0.0,
        resource_pressure: float = 0.0,
        timestamp: float = 0.0,
    ) -> StateVector:
        """Build a StateVector, clamping every pressure into [0, 1]."""
        return cls(
            threat_pressure=clamp(threat_pressure),
            latency_pressure=clamp(latency_pressure),
            failure_pressure=clamp(failure_pressure),
            drift_pressure=clamp(drift_pressure),
            adversarial_pressure=clamp(adversarial_pressure),
            resource_pressure=clamp(resource_pressure),
            timestamp=timestamp,
        )

    @classmethod
    def zero(cls, timestamp: float = 0.0) -> StateVector:
        """The origin of the state space: no pressure on any dimension."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, timestamp)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> StateVector:
        """Rebuild a StateVector from :meth:`to_dict` output."""
        return cls.create(
            threat_pressure=float(data.get("threat_pressure", 0.0)),
            latency_pressure=float(data.get("latency_pressure", 0.0)),
            failure_pressure=float(data.get("failure_pressure", 0.0)),
            drift_pressure=float(data.get("drift_pressure", 0.0)),
            adversarial_pressure=float(data.get("adversarial_pressure", 0.0)),
            resource_pressure=float(data.get("resource_pressure", 0.0)),
            timestamp=float(data.get("timestamp", 0.0)),
        )

    # -- vector behaviour -------------------------------------------------

    @property
    def pressures(self) -> tuple[float, ...]:
        """The six pressures in :data:`PRESSURE_NAMES` order."""
        return (
            self.threat_pressure,
            self.latency_pressure,
            self.failure_pressure,
            self.drift_pressure,
            self.adversarial_pressure,
            self.resource_pressure,
        )

    def __iter__(self) -> Iterator[float]:
        return iter(self.pressures)

    @property
    def magnitude(self) -> float:
        """Euclidean norm of the pressure vector (0 .. sqrt(6))."""
        return math.sqrt(sum(p * p for p in self.pressures))

    @property
    def dominant(self) -> str:
        """Name of the pressure dimension carrying the most load.

        Ties resolve toward the earlier dimension in :data:`PRESSURE_NAMES`,
        which orders security-relevant pressures first.
        """
        pressures = self.pressures
        best_index = max(range(len(pressures)), key=lambda i: pressures[i])
        return PRESSURE_NAMES[best_index]

    def distance_to(self, other: StateVector) -> float:
        """Euclidean distance between two states, ignoring timestamps."""
        return math.sqrt(
            sum((a - b) ** 2 for a, b in zip(self.pressures, other.pressures, strict=True))
        )

    def blend(self, other: StateVector, weight: float) -> StateVector:
        """Linear interpolation toward ``other``; ``weight`` of 0 returns self.

        Used by the telemetry layer to smooth jittery inputs without holding a
        mutable running average.
        """
        w = clamp(weight)
        a, b, c, d, e, f = (
            x + (y - x) * w
            for x, y in zip(self.pressures, other.pressures, strict=True)
        )
        return StateVector(a, b, c, d, e, f, timestamp=other.timestamp)

    def with_pressure(self, name: str, value: float) -> StateVector:
        """Return a copy with one named pressure replaced (clamped)."""
        if name not in PRESSURE_NAMES:
            raise KeyError(f"unknown pressure dimension: {name!r}")
        return replace(self, **{name: clamp(value)})

    def raised(self, name: str, amount: float) -> StateVector:
        """Return a copy with one pressure increased by ``amount`` (clamped).

        This is how AMX and BVE feed learned adversarial history back into the
        state without either subsystem needing to know the vector's shape.
        """
        return self.with_pressure(name, getattr(self, name) + amount)

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, float]:
        """JSON-safe mapping, rounded for stable audit-log comparison."""
        out = {
            name: round(value, 6)
            for name, value in zip(PRESSURE_NAMES, self.pressures, strict=True)
        }
        out["timestamp"] = self.timestamp
        return out


@dataclass(frozen=True, slots=True)
class PolicyTelemetry:
    """What GSA reports to URE *about* a policy decision, never the policy itself.

    This is the structure the archive's URE v2 review called for: it lets AMX
    correlate and BVE learn from policy outcomes while keeping URE ignorant of
    policy type, implementation, domain, and rules. URE sees a score and a
    verdict shape -- never a rule.
    """

    policy_name: str
    policy_version: str
    input_score: float
    hard_block: bool
    output_acceptable: bool
    timestamp: float
    #: Opaque, non-reversible request fingerprint used for AMX/BVE correlation.
    #: GSA supplies a hash; URE never sees request content.
    signature: str = ""

    @property
    def adverse(self) -> bool:
        """True when this decision is evidence of hostile or failing behaviour."""
        return self.hard_block or not self.output_acceptable or self.input_score >= 0.5


def mean_state(states: Sequence[StateVector]) -> StateVector:
    """Component-wise mean of a sequence of states (timestamp of the last one).

    Returns the zero vector for an empty sequence so callers never have to
    special-case cold start.
    """
    if not states:
        return StateVector.zero()
    count = len(states)
    sums = [0.0] * len(PRESSURE_NAMES)
    for state in states:
        for index, pressure in enumerate(state.pressures):
            sums[index] += pressure
    a, b, c, d, e, f = (total / count for total in sums)
    return StateVector(a, b, c, d, e, f, timestamp=states[-1].timestamp)
