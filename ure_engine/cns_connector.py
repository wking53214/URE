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
from types import ModuleType
from typing import Any

from .assessment import GovernanceDecision, UREAssessment
from .recovery import RecoveryAction
from .regimes import SystemRegime

__all__ = [
    "CnsNotInstalled",
    "UREGate",
    "assessment_digest",
    "cns_available",
    "to_cns_result",
]

INSTALL_HINT = "pip install 'ure-engine[cns]'"
SUBJECT = "assessment"

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


def to_cns_result(assessment: UREAssessment, *, position: str = "omega") -> Any:
    """Restate one URE assessment as a ``cns.gate.GateResult``."""
    gate = _cns_gate()
    end = _position(gate, position)
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

    def __init__(self, position: str = "omega") -> None:
        gate = _cns_gate()  # fail here, at construction, not on first use
        self._position = position
        _position(gate, position)

    @property
    def position(self) -> Any:
        return _position(_cns_gate(), self._position)

    def check(self, candidate: object) -> Any:
        return to_cns_result(candidate, position=self._position)  # type: ignore[arg-type]
