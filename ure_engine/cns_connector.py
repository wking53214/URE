"""Optional connector to CNS (``cns.gate``).

URE is independent. It has no runtime dependency, imports nothing from CNS
when it loads, and its whole suite passes with CNS absent. This module is the
one place that knows CNS exists, and it asks for CNS only when one of its
functions is called. Without CNS installed those calls raise
:class:`CnsNotInstalled` with the install command; nothing else in URE is
affected.

What connecting means
---------------------
URE concludes something about the state of a system and recommends a
governance decision (:class:`~ure_engine.assessment.GovernanceDecision`). URE
recommends and the gateway decides, and that stays true here: this module
restates the recommendation in CNS's shared gate vocabulary
(``cns.gate.GateResult``) so a consumer that speaks CNS can resolve it
alongside gates from other repositories. Whether to honour it is still the
consumer's call.

The mapping, and why
--------------------
=====================  =============================================
URE decision           CNS outcome
=====================  =============================================
``ALLOW``              ``PASS``
``REVIEW``             ``RETRY``  (waiting on a human, not refused)
``THROTTLE``           ``RETRY``  (the system is stressed; ask again
                       at a lower rate or when it has recovered)
``DEGRADE``            ``RETRY``  (same: serve less, or later)
``ISOLATE``            ``TERMINAL_BREACH``  (cut off; asking again
                       does not change the state it was cut off for)
``QUARANTINE``         ``TERMINAL_BREACH``
``DENY``               ``TERMINAL_BREACH``
anything else          ``TERMINAL_BREACH``  (an unknown decision is
                       never read as permission)
=====================  =============================================

The decision is read against the assessment's state, never on its own:
``decision`` defaults to ``ALLOW``, so a hand-built assessment can leave it
unset. The regime and the recommended action each imply a least restrictive
decision (the floors the engine's planner never goes below); a stated decision
below either one means state and decision disagree, and that is
``TERMINAL_BREACH``. A decision, regime or recommended action that is not of
URE's own enum type is an unbound ``TERMINAL_BREACH``.

The reason is URE's own explanation, with the regime and the decision in
front of it. ``subject`` is ``"assessment"`` and ``subject_digest`` is
``cns.gate.subject_digest`` over ``assessment.to_dict()``, so a verdict binds
to that exact observation (its timestamp is in the dictionary) and not to a
later one.

Freshness
---------
An assessment describes one moment. The gate takes ``max_age`` (seconds,
default ``DEFAULT_MAX_AGE_SECONDS``); an assessment older than that, dated
more than ``FUTURE_SKEW_SECONDS`` ahead of this clock, or with a timestamp
that is not a finite number is a ``TERMINAL_BREACH``. A ``max_age`` that is
not a positive finite number is refused with ``ValueError``.

Position
--------
``OMEGA`` by default: the assessment describes the state the work left the
system in, and a result is held when that state is not healthy. A consumer
that wants it as a precondition (is the system fit to take this work on?)
passes ``position="alpha"``. URE's own assessment is the same either way; only
the end it is seated at differs.

Install with the extra: ``pip install 'ure-engine[cns]'``.
"""

from __future__ import annotations

import importlib
import math
import time
from types import ModuleType
from typing import Any

from .assessment import GovernanceDecision, UREAssessment
from .recovery import RecoveryAction
from .regimes import SystemRegime

__all__ = [
    "DEFAULT_MAX_AGE_SECONDS",
    "FUTURE_SKEW_SECONDS",
    "CnsNotInstalled",
    "UREGate",
    "assessment_digest",
    "cns_available",
    "to_cns_result",
]

INSTALL_HINT = "pip install 'ure-engine[cns]'"
SUBJECT = "assessment"

#: How old an assessment may be, in seconds, before the gate calls it stale.
#: An assessment describes one moment; a minute later it no longer describes
#: the system. Callers that observe less often pass a larger ``max_age``.
DEFAULT_MAX_AGE_SECONDS = 60.0

#: How far ahead of this clock an assessment's timestamp may be (clock skew)
#: before the gate calls it future-dated.
FUTURE_SKEW_SECONDS = 5.0

_RETRY = frozenset(
    {GovernanceDecision.REVIEW, GovernanceDecision.THROTTLE, GovernanceDecision.DEGRADE}
)

#: The least restrictive decision each regime admits. These are the floors the
#: engine's own recovery planner never goes below (``recovery.py``), so a real
#: assessment always meets them; a hand-built one that does not is a breach.
_REGIME_FLOOR = {
    SystemRegime.NOMINAL: GovernanceDecision.ALLOW,
    SystemRegime.ADAPTING: GovernanceDecision.ALLOW,
    SystemRegime.RECOVERING: GovernanceDecision.ALLOW,
    SystemRegime.STRESSED: GovernanceDecision.DEGRADE,
    SystemRegime.UNKNOWN: GovernanceDecision.DEGRADE,
    SystemRegime.ATTACKED: GovernanceDecision.ISOLATE,
    SystemRegime.CASCADING: GovernanceDecision.QUARANTINE,
}


class CnsNotInstalled(ImportError):
    """Raised by this module's functions when ``cns`` cannot be imported."""


def _cns_gate() -> ModuleType:
    """Import ``cns.gate`` on demand, or say exactly what is missing."""
    try:
        return importlib.import_module("cns.gate")
    except ImportError as exc:
        raise CnsNotInstalled(
            "ure_engine.cns_connector needs the CNS package (cns.gate), which is not "
            f"installed. Install it with: {INSTALL_HINT}. URE itself works "
            "without it."
        ) from exc


def cns_available() -> bool:
    """Whether the CNS gate contract can be imported in this environment."""
    try:
        _cns_gate()
    except CnsNotInstalled:
        return False
    return True


def assessment_digest(assessment: UREAssessment) -> str:
    """The digest a bound verdict on ``assessment`` carries.

    Pass it, with ``SUBJECT``, to ``GateResult.binds`` to check that a
    verdict was issued against exactly this observation.
    """
    return str(_cns_gate().subject_digest(assessment.to_dict()))


def _max_age(max_age: float) -> float:
    if (
        isinstance(max_age, bool)
        or not isinstance(max_age, (int, float))
        or not math.isfinite(max_age)
        or max_age <= 0
    ):
        raise ValueError(f"max_age must be a positive finite number of seconds, not {max_age!r}")
    return float(max_age)


def _staleness(timestamp: object, max_age: float) -> str | None:
    """Why ``timestamp`` cannot be judged as current, or None when it can."""
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(timestamp)
    ):
        return f"the assessment's timestamp {timestamp!r} is not a finite number of seconds"
    now = time.time()
    if timestamp > now + FUTURE_SKEW_SECONDS:
        return f"the assessment is future-dated by {timestamp - now:.1f}s"
    if now - timestamp > max_age:
        return f"the assessment is stale: {now - timestamp:.1f}s old, max_age {max_age:g}s"
    return None


def to_cns_result(
    assessment: UREAssessment,
    *,
    position: str = "omega",
    max_age: float = DEFAULT_MAX_AGE_SECONDS,
) -> Any:
    """Restate one URE assessment as a ``cns.gate.GateResult``.

    An assessment older than ``max_age`` seconds, or dated more than
    ``FUTURE_SKEW_SECONDS`` ahead of this clock, is a ``TERMINAL_BREACH``.
    """
    gate = _cns_gate()
    end = _position(gate, position)
    max_age = _max_age(max_age)
    if not isinstance(assessment, UREAssessment):
        raise TypeError(f"URE's gate judges a UREAssessment; got {type(assessment).__name__}")
    decision = assessment.decision
    regime = assessment.regime
    action = assessment.recommended_action
    if (
        not isinstance(decision, GovernanceDecision)
        or not isinstance(regime, SystemRegime)
        or not isinstance(action, RecoveryAction)
    ):
        return gate.GateResult(
            gate="ure.assessment",
            position=end,
            outcome=gate.GateOutcome.TERMINAL_BREACH,
            reason=(
                "the assessment's decision, regime or recommended action is not of URE's "
                "own type; nothing was judged"
            ),
        )
    # Judge from the state, not from ``decision`` alone: ``decision`` has a
    # default (ALLOW), so a hand-built assessment can carry it unset. The state
    # implies the least restrictive decision it admits; a stated decision below
    # that is a disagreement, and a disagreement is a breach.
    implied = max(
        GovernanceDecision.from_recovery(action),
        _REGIME_FLOOR[regime],
        key=lambda d: d.restriction,
    )
    disagrees = decision.restriction < implied.restriction
    if disagrees:
        outcome = gate.GateOutcome.TERMINAL_BREACH
    elif decision is GovernanceDecision.ALLOW:
        outcome = gate.GateOutcome.PASS
    elif decision in _RETRY:
        outcome = gate.GateOutcome.RETRY
    else:
        outcome = gate.GateOutcome.TERMINAL_BREACH
    reason = f"{regime.value}, {decision.value}: {assessment.explanation}"
    if disagrees:
        reason = (
            f"state and decision disagree: {regime.value} with recommended action "
            f"{action.value} implies at least {implied.value}, but the decision is "
            f"{decision.value}. {reason}"
        )
    try:
        digest = assessment_digest(assessment)
    except (TypeError, ValueError, AttributeError) as exc:
        return gate.GateResult(
            gate="ure.assessment",
            position=end,
            outcome=gate.GateOutcome.TERMINAL_BREACH,
            reason=f"the assessment cannot be described for binding ({exc}); nothing was judged",
        )
    stale = _staleness(assessment.timestamp, max_age)
    if stale is not None:
        outcome = gate.GateOutcome.TERMINAL_BREACH
        reason = f"{stale}. {reason}"
    return gate.GateResult(
        gate="ure.assessment",
        position=end,
        outcome=outcome,
        reason=reason,
        subject=SUBJECT,
        subject_digest=digest,
    )


def _position(gate: ModuleType, position: str) -> Any:
    if position not in ("alpha", "omega"):
        raise ValueError(f"position must be 'alpha' or 'omega', not {position!r}")
    return gate.GatePosition.ALPHA if position == "alpha" else gate.GatePosition.OMEGA


class UREGate:
    """URE's recommendation as a ``cns.gate.Gate``. Judges a :class:`UREAssessment`."""

    name = "ure.assessment"

    def __init__(
        self, position: str = "omega", *, max_age: float = DEFAULT_MAX_AGE_SECONDS
    ) -> None:
        gate = _cns_gate()  # fail here, at construction, not on first use
        self._position = position
        _position(gate, position)
        self._max_age = _max_age(max_age)

    @property
    def position(self) -> Any:
        return _position(_cns_gate(), self._position)

    def check(self, candidate: object) -> Any:
        return to_cns_result(  # type: ignore[arg-type]
            candidate, position=self._position, max_age=self._max_age
        )
