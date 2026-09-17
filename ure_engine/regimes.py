"""
regimes.py, what kind of trouble the system is in.
===================================================

A regime is a qualitative answer to "what is happening", as distinct from the
quantitative "how bad is it" that energy and the resilience index report. The
predecessor engine had three coarse health states (NEUTRAL / REGRESSIVE /
RISK_INCREASING) which could not distinguish a system under attack from a
system merely overloaded -- and those two conditions call for opposite
responses. Throttling an overloaded system helps; throttling an attacked system
hands the attacker a denial of service.

Soft classification, not thresholds
-----------------------------------
The architecture is explicit that regime classification should use "Lyapunov
energy, Lyapunov derivative, historical memory, and trajectory features -- not
static thresholds". The predecessor's cascade of ``if energy >= 0.60`` branches
was a stopgap: correct at the calibration points, arbitrary between them, and
impossible to explain to an operator standing at a boundary.

This module instead scores a continuous *affinity* for each regime and takes
the distribution seriously. That buys three things the cascade could not give:

1. **Confidence.** The winning regime's share of the probability mass says how
   sure the engine is, so a 0.34/0.33/0.33 split is visibly a coin flip rather
   than a confident answer.
2. **Entropy.** Distribution entropy is a direct read on classifier ambiguity,
   and the legacy GSA interface already had a field for it that the old engine
   could only fill with a hand-wave.
3. **Explanation.** The runner-up regime and the margin between them are what
   an operator actually wants to see at a boundary.

Duration as evidence
--------------------
Energy, derivative, volatility and acceleration are all statements about
*motion*, and all of them read zero for a quantity held constant. That is the
hole a patient adversary uses: sit at a hostile rate below the affinity
boundary and there is no level to trip and no slope to measure. The ``dwell``
input closes it by supplying the one thing none of the others carry, how long
the hostile pressure has been elevated. See :mod:`ure_engine.dwell` for why it
is restricted to hostile pressure and must not be extended to operational load.

The predecessor's calibration bands are preserved as the anchor points the
affinity functions are fitted to, and its reachability sweep is carried forward
as a test: every regime must be reachable somewhere in the state space, and
UNKNOWN must never appear for well-formed input. That sweep is what caught the
uncentered-energy bug originally, so it stays.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final

from .state_vector import StateVector, clamp

__all__ = [
    "RegimeClassifier",
    "RegimeProfile",
    "SystemRegime",
]


class SystemRegime(str, Enum):
    """The operating regimes URE distinguishes.

    Ordered from healthiest to most severe. ``str`` mixin so the values
    serialize directly into audit records and Prometheus labels.
    """

    NOMINAL = "NOMINAL"
    """Low energy, quiet trajectory. Nothing to do."""

    ADAPTING = "ADAPTING"
    """Mid-band energy with motion but no dominant hostile or failing pressure.
    The system is working harder than baseline and absorbing it."""

    RECOVERING = "RECOVERING"
    """Energy is dissipating. The Lyapunov condition for a system returning to
    equilibrium. Do not intervene; intervention here is what turns a recovery
    into an oscillation."""

    STRESSED = "STRESSED"
    """Elevated energy driven by latency, failure, or resource pressure --
    load, not malice. Throttling and shedding help."""

    ATTACKED = "ATTACKED"
    """Threat or adversarial pressure dominates. Tighten thresholds and
    isolate; do NOT simply throttle, which rewards the attacker."""

    CASCADING = "CASCADING"
    """High energy, rising, with broad failure or resource involvement.
    Failures are inducing failures. This is the regime recovery exists for."""

    UNKNOWN = "UNKNOWN"
    """Classification could not be established -- degenerate or absent input.
    Treated as a fault, never as a benign default."""

    @property
    def health_status(self) -> str:
        """Project onto the three-state vocabulary legacy GSA understands.

        This is a deliberate lossy projection kept only for the compatibility
        layer. New integrations should read :class:`SystemRegime` directly.
        """
        if self in (SystemRegime.NOMINAL, SystemRegime.RECOVERING, SystemRegime.ADAPTING):
            return "NEUTRAL"
        if self in (SystemRegime.STRESSED, SystemRegime.ATTACKED):
            return "RISK_INCREASING"
        if self is SystemRegime.CASCADING:
            return "REGRESSIVE"
        return "UNKNOWN"

    @property
    def severity(self) -> int:
        """Ordinal severity, 0 (NOMINAL) .. 5 (CASCADING); UNKNOWN sorts high.

        UNKNOWN ranks above STRESSED on purpose: an engine that cannot classify
        its own state is not in a mild condition.
        """
        return _SEVERITY[self]

    @property
    def is_hostile(self) -> bool:
        """True when the regime implies an adversary rather than mere load."""
        return self in (SystemRegime.ATTACKED, SystemRegime.CASCADING)


_SEVERITY: Final[dict[SystemRegime, int]] = {
    SystemRegime.NOMINAL: 0,
    SystemRegime.ADAPTING: 1,
    SystemRegime.RECOVERING: 2,
    SystemRegime.STRESSED: 3,
    SystemRegime.ATTACKED: 4,
    SystemRegime.CASCADING: 5,
    SystemRegime.UNKNOWN: 4,
}

#: Every regime the classifier will assign. UNKNOWN is excluded: it is a fault
#: signal, not a candidate hypothesis.
CLASSIFIABLE: Final[tuple[SystemRegime, ...]] = (
    SystemRegime.NOMINAL,
    SystemRegime.ADAPTING,
    SystemRegime.RECOVERING,
    SystemRegime.STRESSED,
    SystemRegime.ATTACKED,
    SystemRegime.CASCADING,
)

#: Normalizer for distribution entropy, so entropy reports on [0, 1].
_MAX_ENTROPY: Final[float] = math.log(len(CLASSIFIABLE))

#: Derivative magnitude at which "rising" or "falling" is fully asserted.
_TREND_SCALE: Final[float] = 0.05

#: Volatility at which the trajectory is considered fully erratic.
_VOLATILITY_SCALE: Final[float] = 0.15

#: Centre and width of the ADAPTING energy band, in normalized energy.
_ADAPTING_CENTRE: Final[float] = 0.45
_ADAPTING_WIDTH: Final[float] = 0.25


#: Normalized energy below which CASCADING carries no evidence at all. At the
#: default ceiling of 1.5 this is an absolute energy of ~0.98, matching the
#: predecessor's calibrated `energy >= 1.0` cascade band.
_CASCADE_FLOOR: Final[float] = 0.65


def _bump(x: float, centre: float, width: float) -> float:
    """Gaussian membership function, 1.0 at ``centre``, falling off by ``width``."""
    return math.exp(-(((x - centre) / width) ** 2))


def _gate(x: float, floor: float) -> float:
    """Rescale ``x`` above ``floor`` onto [0, 1]; exactly 0 at or below it.

    A hard floor rather than a soft weight: some hypotheses should carry *no*
    evidence outside their band, not merely a little.
    """
    if x <= floor or floor >= 1.0:
        return 0.0
    return (x - floor) / (1.0 - floor)


@dataclass(frozen=True, slots=True)
class RegimeProfile:
    """The classifier's full output, not just its winner."""

    dominant: SystemRegime
    distribution: Mapping[SystemRegime, float]
    confidence: float
    entropy: float
    runner_up: SystemRegime
    margin: float

    @property
    def ambiguous(self) -> bool:
        """True when the top two hypotheses are close enough to be a toss-up.

        Callers that escalate on ambiguity (the recovery planner does) use this
        rather than re-deriving the margin threshold.
        """
        return self.margin < 0.10 or self.confidence < 0.40

    def probability(self, regime: SystemRegime) -> float:
        """Probability mass assigned to ``regime`` (0.0 if not a candidate)."""
        return self.distribution.get(regime, 0.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "dominant": self.dominant.value,
            "confidence": round(self.confidence, 4),
            "entropy": round(self.entropy, 4),
            "runner_up": self.runner_up.value,
            "margin": round(self.margin, 4),
            "distribution": {
                regime.value: round(p, 4) for regime, p in self.distribution.items()
            },
        }

    @classmethod
    def unknown(cls) -> RegimeProfile:
        """The profile returned when no hypothesis has any support."""
        uniform = 1.0 / len(CLASSIFIABLE)
        return cls(
            dominant=SystemRegime.UNKNOWN,
            distribution={regime: uniform for regime in CLASSIFIABLE},
            confidence=0.0,
            entropy=1.0,
            runner_up=SystemRegime.UNKNOWN,
            margin=0.0,
        )


class RegimeClassifier:
    """Scores continuous affinities for each regime and normalizes them.

    Stateless with respect to individual calls: all history arrives through the
    energy derivative, the volatility, and the ``recall_strength`` memory term,
    which keeps the classifier itself trivially testable.
    """

    __slots__ = ("_energy_max",)

    def __init__(self, energy_max: float = 1.5) -> None:
        if energy_max <= 0.0:
            raise ValueError("energy_max must be positive")
        self._energy_max = energy_max

    def affinities(
        self,
        state: StateVector,
        energy: float,
        derivative: float,
        volatility: float = 0.0,
        recall_strength: float = 0.0,
        dwell: float = 0.0,
    ) -> dict[SystemRegime, float]:
        """Unnormalized evidence for each candidate regime.

        ``recall_strength`` is AMX's confidence that the current signature
        matches a remembered attack. It is the "historical memory" input the
        architecture requires: it sharpens ATTACKED without being able to
        manufacture it from nothing, because it enters multiplicatively
        alongside observed hostile pressure rather than as a standalone term.

        ``dwell`` is the same kind of input from the other direction: how long
        hostile pressure has been elevated, supplied by
        :class:`~ure_engine.dwell.HostileDwell`. It exists because every other
        input here describes a level or a rate of change, and an adversary
        holding a constant defeats both. Like ``recall`` it is multiplicative
        and cannot fabricate a regime: with no hostile pressure the ATTACKED
        term is zero and no amount of dwell lifts it.
        """
        normalized_energy = clamp(energy / self._energy_max)
        rising = clamp(derivative / _TREND_SCALE)
        falling = clamp(-derivative / _TREND_SCALE)
        erratic = clamp(volatility / _VOLATILITY_SCALE)
        recall = clamp(recall_strength)
        exposure = clamp(dwell)

        hostile = max(state.threat_pressure, state.adversarial_pressure)
        operational = max(
            state.latency_pressure, state.failure_pressure, state.resource_pressure
        )
        breaking = max(state.failure_pressure, state.resource_pressure)

        return {
            # Quiet and low. Cubed so it collapses quickly once energy lifts.
            # Quiet and low -- but only if it has also been quiet. The dwell
            # term is what stops a patient adversary from sitting at a low
            # constant rate and being called NOMINAL forever. Damped to 0.9 so
            # NOMINAL can never be driven to exactly zero: a distribution with
            # no support at all is reported as UNKNOWN, and "I cannot classify
            # this" is the wrong answer to "somebody has been probing you".
            SystemRegime.NOMINAL: (
                ((1.0 - normalized_energy) ** 3) * (1.0 - erratic) * (1.0 - 0.9 * exposure)
            ),
            # Mid-band, in motion, nothing specific wrong.
            SystemRegime.ADAPTING: (
                _bump(normalized_energy, _ADAPTING_CENTRE, _ADAPTING_WIDTH)
                * (0.35 + 0.65 * erratic)
                * (1.0 - hostile)
            ),
            # Energy dissipating. Weighted by how much there is left to shed,
            # so falling from high energy reads as recovery more strongly than
            # a twitch near the floor.
            SystemRegime.RECOVERING: (
                falling * (0.30 + 0.70 * normalized_energy) * (1.0 - 0.8 * hostile)
            ),
            # Load, not malice: energy driven by operational pressure.
            SystemRegime.STRESSED: (
                normalized_energy * operational * (1.0 - 0.6 * hostile) * 1.2
            ),
            # Malice. Suppressed while energy is actively dissipating, since a
            # receding attack is a recovery.
            SystemRegime.ATTACKED: (
                (hostile ** 1.5)
                * (0.40 + 0.60 * (1.0 - falling))
                * (1.0 + 0.5 * recall)
                # Faded out by hostile pressure itself. Dwell exists to
                # supply what the level cannot, and at hostile pressure near
                # 1.0 the level already says everything -- amplifying there
                # only inflates ATTACKED past CASCADING on a genuine cascade,
                # which downgrades the recovery action from QUARANTINE to
                # ISOLATE at exactly the wrong moment. Measured, not assumed:
                # test_quarantine_governs_regardless_of_score caught it.
                * (1.0 + 2.0 * exposure * (1.0 - hostile))
                * 1.6
            ),
            # Failures inducing failures: near saturation, rising, and broadly
            # breaking. The energy term is gated at _CASCADE_FLOOR rather than
            # merely weighted, because a steep *legitimate* load ramp looks
            # identical to a cascade in every other respect -- rising energy,
            # high failure and resource pressure -- and calling it CASCADING
            # quarantines a system that is only busy. Requiring the system to
            # be genuinely near collapse is what separates the two, and it
            # matches the predecessor's calibrated `energy >= 1.0` band.
            SystemRegime.CASCADING: (
                (_gate(normalized_energy, _CASCADE_FLOOR) ** 2)
                * rising
                * (0.30 + 0.70 * breaking)
                * 3.6
            ),
        }

    def classify(
        self,
        state: StateVector,
        energy: float,
        derivative: float,
        volatility: float = 0.0,
        recall_strength: float = 0.0,
        dwell: float = 0.0,
    ) -> RegimeProfile:
        """Classify the current regime and report the full distribution."""
        # Guard non-finite inputs before they reach the affinity functions.
        # clamp() maps NaN to its floor, which is the right defensive choice
        # for a *pressure* but catastrophic for energy: a NaN energy would
        # normalize to 0.0 and the engine would confidently report NOMINAL for
        # a computation that had actually broken. Failing visibly beats failing
        # reassuringly.
        if not (
            math.isfinite(energy)
            and math.isfinite(derivative)
            and math.isfinite(volatility)
        ):
            return RegimeProfile.unknown()

        scores = self.affinities(
            state, energy, derivative, volatility, recall_strength, dwell
        )
        total = sum(scores.values())

        if not math.isfinite(total) or total <= 1e-9:
            # No hypothesis has support. This is a fault, not a quiet system:
            # a genuinely quiet system scores high NOMINAL affinity.
            return RegimeProfile.unknown()

        distribution = {regime: score / total for regime, score in scores.items()}
        ranked = sorted(distribution.items(), key=lambda kv: (-kv[1], kv[0].severity))
        dominant, top = ranked[0]
        runner_up, second = ranked[1]

        return RegimeProfile(
            dominant=dominant,
            distribution=distribution,
            confidence=top,
            entropy=self.entropy(distribution),
            runner_up=runner_up,
            margin=top - second,
        )

    @staticmethod
    def entropy(distribution: Mapping[SystemRegime, float]) -> float:
        """Shannon entropy of the distribution, normalized to [0, 1].

        0.0 means the classifier put all mass on one regime; 1.0 means it could
        not distinguish between them at all.
        """
        total = 0.0
        for probability in distribution.values():
            if probability > 0.0:
                total -= probability * math.log(probability)
        return clamp(total / _MAX_ENTROPY) if _MAX_ENTROPY > 0 else 0.0
