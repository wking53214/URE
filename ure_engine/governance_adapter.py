"""
governance_adapter.py — the GSA <-> URE translation layer.
==========================================================

The architecture identifies this module as the one place with a "direct
integration point" to policy. Every other URE module consumes either a
StateVector or nothing at all. Concentrating the coupling here is what keeps
the policy seam intact while still letting policy outcomes influence
resilience.

The seam, restated
------------------
A policy answers exactly two questions -- ``assess_input`` and
``inspect_output`` -- and returns graded verdicts. It never imports gateway
internals, and nothing downstream of it may import policy internals. URE's side
of that bargain::

    URE must never know: policy type, policy implementation, policy domain,
    policy rules, policy internals. It consumes only gateway telemetry.

So this adapter accepts a *score* and a *verdict shape*. It does not accept a
policy object, and it has no import path to one. What it produces is the
``threat_pressure`` dimension: the policy result becomes one signal among six
rather than the whole decision.

Direction of dependency
-----------------------
GSA imports URE. URE does not import GSA. The
:class:`~ure_engine.engine.GatewayHealthEngine` protocol lets GSA depend on a
shape rather than a class, so neither side needs the other's package installed
to be type-checked or tested.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .assessment import GovernanceDecision, UREAssessment
from .recovery import RecoveryAction
from .state_vector import PolicyTelemetry, StateVector, clamp
from .telemetry import TelemetryFrame

__all__ = ["GSATelemetryAdapter", "GovernanceOutcome"]


@dataclass(frozen=True, slots=True)
class GovernanceOutcome:
    """What the gateway should actually do with this request.

    Combines URE's system-level recommendation with the policy's
    request-level verdict. Neither alone is sufficient: the policy knows
    whether *this* request is acceptable, URE knows whether the system can
    afford to be generous right now.
    """

    decision: GovernanceDecision
    #: Threshold that was applied to the policy score.
    threshold: float
    #: The policy score that was compared against it.
    score: float
    #: True when a policy hard-block bypassed the threshold entirely.
    hard_blocked: bool
    assessment: UREAssessment
    rationale: str

    @property
    def allowed(self) -> bool:
        return self.decision is GovernanceDecision.ALLOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "threshold": round(self.threshold, 4),
            "score": round(self.score, 4),
            "hard_blocked": self.hard_blocked,
            "rationale": self.rationale,
            "assessment": self.assessment.to_dict(),
        }


class GSATelemetryAdapter:
    """Translates GSA's operational signals into URE's state space.

    The archive specifies this class by name and signature. The original
    four-signal form (:meth:`build_state_vector`) is kept verbatim for callers
    that still speak it; :meth:`build_frame` is the richer path that also
    carries policy outcome, signatures, and markers.
    """

    __slots__ = ("_backlog_capacity", "_latency_ceiling_ms")

    def __init__(
        self, backlog_capacity: int = 256, latency_ceiling_ms: float = 1000.0
    ) -> None:
        if backlog_capacity < 1:
            raise ValueError("backlog_capacity must be at least 1")
        self._backlog_capacity = backlog_capacity
        self._latency_ceiling_ms = latency_ceiling_ms

    # -- the archive's original signature ---------------------------------

    def build_state_vector(
        self,
        reject_rate: float,
        retry_rate: float,
        latency_ms: float,
        backlog: int,
        *,
        now: float | None = None,
    ) -> StateVector:
        """Project GSA's four legacy signals onto the six pressures.

        Note on ``reject_rate``: it drives threat pressure directly, not just
        the block-rate term. From the gateway's perspective a surge of policy
        rejections *is* the threat signal -- the gateway is actively refusing
        hostile input. Routing it only into the 0.3-weighted block-rate term
        under-reports it badly: the compatibility work in the archive found the
        regime never left NOMINAL even at a sustained 60% rejection rate.
        """
        timestamp = time.time() if now is None else now
        return StateVector.create(
            threat_pressure=reject_rate,
            latency_pressure=latency_ms / self._latency_ceiling_ms,
            failure_pressure=retry_rate,
            drift_pressure=0.0,
            adversarial_pressure=0.0,
            resource_pressure=backlog / self._backlog_capacity,
            timestamp=timestamp,
        )

    # -- the richer path ---------------------------------------------------

    def build_frame(
        self,
        *,
        reject_rate: float = 0.0,
        retry_rate: float = 0.0,
        latency_ms: float = 0.0,
        backlog: int = 0,
        risk_score: float | None = None,
        substrate_health: float = 1.0,
        circuit_state: str = "CLOSED",
        latency_volatility: float = 0.0,
        drift_metric: float = 0.0,
        adversarial_events: float = 0.0,
        signatures: Sequence[str] = (),
        markers: Sequence[str] = (),
        policy: PolicyTelemetry | None = None,
    ) -> TelemetryFrame:
        """Assemble a full telemetry frame from gateway signals.

        ``risk_score`` defaults to ``reject_rate`` when the caller has no
        separate score, preserving the semantics above.
        """
        return TelemetryFrame(
            risk_score=clamp(reject_rate if risk_score is None else risk_score),
            blocked_rate=clamp(reject_rate),
            retry_rate=clamp(retry_rate),
            latency_ms=max(0.0, latency_ms),
            latency_volatility=clamp(latency_volatility),
            queue_saturation=clamp(backlog / self._backlog_capacity),
            substrate_health=clamp(substrate_health),
            circuit_state=circuit_state,
            adversarial_events=max(0.0, adversarial_events),
            drift_metric=clamp(drift_metric),
            signatures=tuple(signatures),
            markers=tuple(markers),
            policy=policy,
        )

    # -- decision composition ---------------------------------------------

    @staticmethod
    def decide(
        assessment: UREAssessment,
        *,
        score: float,
        hard_block: bool = False,
    ) -> GovernanceOutcome:
        """Combine a policy verdict with URE's assessment into one decision.

        Precedence, most authoritative first:

        1. **Policy hard-block.** Absolute rules stay absolute. No system state
           makes a raw SSN acceptable, and no resilience reading relaxes it.
        2. **System recovery posture.** If URE is recommending QUARANTINE or
           ISOLATE, that governs regardless of this request's score -- the
           system is not in a position to serve it.
        3. **Adaptive threshold.** Otherwise compare the score against the
           threshold URE computed for the current regime.
        """
        if hard_block:
            return GovernanceOutcome(
                decision=GovernanceDecision.DENY,
                threshold=assessment.recommended_threshold,
                score=score,
                hard_blocked=True,
                assessment=assessment,
                rationale=(
                    "Policy asserted an absolute rule (hard_block). Denied "
                    "irrespective of system state; adaptive thresholds do not "
                    "apply to non-negotiable rules."
                ),
            )

        if assessment.recommended_action is RecoveryAction.QUARANTINE:
            return GovernanceOutcome(
                decision=GovernanceDecision.QUARANTINE,
                threshold=assessment.recommended_threshold,
                score=score,
                hard_blocked=False,
                assessment=assessment,
                rationale=(
                    f"System is {assessment.regime.value} with resilience "
                    f"{assessment.resilience_index:.2f}; admission is stopped "
                    "regardless of this request's score."
                ),
            )

        # Under attack the threshold has already tightened sharply. A request
        # that still clears it is served; one that does not is isolated rather
        # than merely denied, so the scope is contained.
        if (
            assessment.recommended_action is RecoveryAction.ISOLATE
            and score >= assessment.recommended_threshold
        ):
            return GovernanceOutcome(
                decision=GovernanceDecision.ISOLATE,
                threshold=assessment.recommended_threshold,
                score=score,
                hard_blocked=False,
                assessment=assessment,
                rationale=(
                    f"Score {score:.2f} exceeds the tightened {assessment.regime.value} "
                    f"threshold {assessment.recommended_threshold:.2f}; isolating scope."
                ),
            )

        if score >= assessment.recommended_threshold:
            return GovernanceOutcome(
                decision=GovernanceDecision.DENY,
                threshold=assessment.recommended_threshold,
                score=score,
                hard_blocked=False,
                assessment=assessment,
                rationale=(
                    f"Score {score:.2f} exceeds the adaptive threshold "
                    f"{assessment.recommended_threshold:.2f} "
                    f"({assessment.regime.value} regime)."
                ),
            )

        # Below threshold, but the system may still be shedding load.
        if assessment.recommended_action in (
            RecoveryAction.THROTTLE,
            RecoveryAction.DEGRADE,
        ):
            decision = GovernanceDecision.from_recovery(assessment.recommended_action)
            return GovernanceOutcome(
                decision=decision,
                threshold=assessment.recommended_threshold,
                score=score,
                hard_blocked=False,
                assessment=assessment,
                rationale=(
                    f"Score {score:.2f} is within the {assessment.recommended_threshold:.2f} "
                    f"threshold, but the system is {assessment.regime.value} and shedding "
                    f"load ({decision.value})."
                ),
            )

        # Near-threshold requests get flagged rather than silently allowed:
        # they are the ones an operator wants in the review queue.
        if score >= assessment.recommended_threshold * 0.8:
            return GovernanceOutcome(
                decision=GovernanceDecision.REVIEW,
                threshold=assessment.recommended_threshold,
                score=score,
                hard_blocked=False,
                assessment=assessment,
                rationale=(
                    f"Score {score:.2f} is within 20% of the "
                    f"{assessment.recommended_threshold:.2f} threshold; allowed and flagged."
                ),
            )

        return GovernanceOutcome(
            decision=GovernanceDecision.ALLOW,
            threshold=assessment.recommended_threshold,
            score=score,
            hard_blocked=False,
            assessment=assessment,
            rationale=(
                f"Score {score:.2f} is within the {assessment.recommended_threshold:.2f} "
                f"threshold and the system is {assessment.regime.value}."
            ),
        )
