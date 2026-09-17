"""
assessment.py, the canonical URE output contract.
==================================================

The archive is explicit about why this type exists: it was added "to prevent
drift across future implementations". Every consumer of URE reads this
structure and nothing else. The engine's internals may be reorganized freely;
this shape is the promise.

``UREAssessment`` is frozen and fully serializable, so it can go straight into
an audit ledger and be replayed later without re-running the engine.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .recovery import RecoveryAction, RecoveryVector
from .regimes import RegimeProfile, SystemRegime
from .resilience import ResilienceBreakdown
from .state_vector import StateVector

__all__ = ["GovernanceDecision", "UREAssessment"]


class GovernanceDecision(str, Enum):
    """The decision vocabulary GSA's decision layer selects from.

    URE recommends; the gateway decides. Listed in ascending restriction so a
    caller can compare decisions ordinally.
    """

    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    THROTTLE = "THROTTLE"
    DEGRADE = "DEGRADE"
    ISOLATE = "ISOLATE"
    QUARANTINE = "QUARANTINE"
    DENY = "DENY"

    @property
    def restriction(self) -> int:
        return _RESTRICTION[self]

    @classmethod
    def from_recovery(cls, action: RecoveryAction) -> GovernanceDecision:
        """Project a recovery action onto the governance vocabulary."""
        return {
            RecoveryAction.NONE: cls.ALLOW,
            RecoveryAction.RESTORE: cls.ALLOW,
            RecoveryAction.DEGRADE: cls.DEGRADE,
            RecoveryAction.THROTTLE: cls.THROTTLE,
            RecoveryAction.ISOLATE: cls.ISOLATE,
            RecoveryAction.QUARANTINE: cls.QUARANTINE,
        }[action]


_RESTRICTION: dict[GovernanceDecision, int] = {
    GovernanceDecision.ALLOW: 0,
    GovernanceDecision.REVIEW: 1,
    GovernanceDecision.THROTTLE: 2,
    GovernanceDecision.DEGRADE: 3,
    GovernanceDecision.ISOLATE: 4,
    GovernanceDecision.QUARANTINE: 5,
    GovernanceDecision.DENY: 6,
}


@dataclass(frozen=True, slots=True)
class UREAssessment:
    """Everything URE concluded about one observation.

    Field names follow the archive's canonical contract exactly; the
    breakdown objects are additions that carry the supporting detail without
    changing the promised surface.
    """

    # -- the headline ----------------------------------------------------
    resilience_index: float
    regime: SystemRegime

    # -- Lyapunov --------------------------------------------------------
    lyapunov_energy: float
    lyapunov_derivative: float

    # -- trajectory ------------------------------------------------------
    trajectory_velocity: float
    trajectory_acceleration: float
    trajectory_volatility: float

    # -- memory and learning ---------------------------------------------
    active_attack_profiles: list[str]
    active_vaccines: list[str]

    # -- recommendation ---------------------------------------------------
    recommended_action: RecoveryAction

    # -- narrative --------------------------------------------------------
    explanation: str
    timestamp: float

    # -- supporting detail (additive; not part of the legacy surface) ------
    state: StateVector = field(default_factory=StateVector.zero)
    regime_confidence: float = 0.0
    regime_entropy: float = 0.0
    regime_profile: RegimeProfile | None = None
    resilience_breakdown: ResilienceBreakdown | None = None
    recovery_vector: RecoveryVector | None = None
    #: Threshold the gateway should apply to policy scores right now.
    recommended_threshold: float = 0.80
    #: Governance decision implied by the recommendation.
    decision: GovernanceDecision = GovernanceDecision.ALLOW
    #: Seconds until energy reaches saturation on the current trajectory.
    time_to_saturation: float | None = None
    #: How long hostile pressure has been sustained, on [0, 1]. The one signal
    #: that still moves once every pressure has clamped at 1.0 and energy,
    #: derivative and volatility have all gone flat, so an integrator watching
    #: for "this is getting worse" at saturation should watch this.
    hostile_dwell: float = 0.0
    #: Engine schema version, so replayed records identify their producer.
    schema_version: str = "2.0"

    # -- derived views -----------------------------------------------------

    @property
    def healthy(self) -> bool:
        """True when nothing needs doing."""
        return (
            self.regime in (SystemRegime.NOMINAL, SystemRegime.RECOVERING)
            and self.recommended_action in (RecoveryAction.NONE, RecoveryAction.RESTORE)
        )

    @property
    def degrading(self) -> bool:
        """True when energy is climbing, whatever the current regime."""
        return self.lyapunov_derivative > 0.02

    @property
    def band(self) -> str:
        """Resilience band label (RESILIENT / ADEQUATE / FRAGILE / CRITICAL)."""
        if self.resilience_breakdown is not None:
            return self.resilience_breakdown.band
        if self.resilience_index >= 0.75:
            return "RESILIENT"
        if self.resilience_index >= 0.50:
            return "ADEQUATE"
        if self.resilience_index >= 0.25:
            return "FRAGILE"
        return "CRITICAL"

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Fully JSON-safe representation, suitable for an audit ledger."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "resilience_index": round(self.resilience_index, 4),
            "band": self.band,
            "regime": self.regime.value,
            "regime_confidence": round(self.regime_confidence, 4),
            "regime_entropy": round(self.regime_entropy, 4),
            "hostile_dwell": round(self.hostile_dwell, 4),
            "lyapunov": {
                "energy": round(self.lyapunov_energy, 4),
                "derivative": round(self.lyapunov_derivative, 6),
                "time_to_saturation": (
                    round(self.time_to_saturation, 2)
                    if self.time_to_saturation is not None
                    else None
                ),
            },
            "trajectory": {
                "velocity": round(self.trajectory_velocity, 6),
                "acceleration": round(self.trajectory_acceleration, 6),
                "volatility": round(self.trajectory_volatility, 6),
            },
            "memory": {
                "active_attack_profiles": list(self.active_attack_profiles),
                "active_vaccines": list(self.active_vaccines),
            },
            "recommended_action": self.recommended_action.value,
            "recommended_threshold": round(self.recommended_threshold, 4),
            "decision": self.decision.value,
            "explanation": self.explanation,
            "state": self.state.to_dict(),
        }
        if self.regime_profile is not None:
            payload["regime_profile"] = self.regime_profile.to_dict()
        if self.resilience_breakdown is not None:
            payload["resilience_breakdown"] = self.resilience_breakdown.to_dict()
        if self.recovery_vector is not None:
            payload["recovery_vector"] = self.recovery_vector.to_dict()
        return payload

    def to_json(self, **kwargs: Any) -> str:
        """Canonical JSON. Keys are sorted so records hash reproducibly."""
        kwargs.setdefault("sort_keys", True)
        kwargs.setdefault("separators", (",", ":"))
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def unknown(cls, reason: str, timestamp: float | None = None) -> UREAssessment:
        """A safe assessment for when the engine cannot conclude anything.

        Reports UNKNOWN with zero resilience rather than a comfortable default.
        An engine that fails toward "everything is fine" is worse than no
        engine, because it manufactures confidence the operator will act on.
        """
        return cls(
            resilience_index=0.0,
            regime=SystemRegime.UNKNOWN,
            lyapunov_energy=0.0,
            lyapunov_derivative=0.0,
            trajectory_velocity=0.0,
            trajectory_acceleration=0.0,
            trajectory_volatility=0.0,
            active_attack_profiles=[],
            active_vaccines=[],
            recommended_action=RecoveryAction.DEGRADE,
            explanation=f"URE could not assess system state: {reason}",
            timestamp=time.time() if timestamp is None else timestamp,
            decision=GovernanceDecision.REVIEW,
            recommended_threshold=0.60,
        )
