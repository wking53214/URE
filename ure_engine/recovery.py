"""
recovery.py, turning a diagnosis into a plan.
==============================================

URE detects instability; FORTRESS decides recovery strategy. This module is the
seam between those two responsibilities: it produces a *recommendation* with a
machine-readable action vector, and it has no authority to execute anything.
That separation is deliberate and load-bearing. An engine that can both declare
an emergency and act on it can put a system into isolation on the strength of
its own misclassification, with nothing in the loop to disagree.

The action ladder
-----------------
Five actions, ordered by how much capability they remove:

``DEGRADE``     Shed optional work; keep serving.
``THROTTLE``    Reduce admitted rate.
``ISOLATE``     Cut the affected tenant or path off from shared resources.
``QUARANTINE``  Stop admitting entirely; preserve state for forensics.
``RESTORE``     Walk back the ladder toward normal operation.

Regime determines which rung
----------------------------
The critical asymmetry, and the reason the regime model exists at all:
**STRESSED throttles, ATTACKED isolates.** Throttling an overloaded system
relieves it. Throttling an attacked system does the attacker's work -- the
attacker wanted the service degraded, and the defence delivered it. Under
attack the correct move is to narrow *who* is served, not *how much* service
exists.

Hysteresis
----------
Recovery recommendations are damped: escalation is prompt, de-escalation is
slow. A system that drops back to NOMINAL for one sample and immediately
un-throttles will re-enter STRESSED on the next sample and oscillate, and an
oscillating control loop is worse than either fixed state. The planner requires
sustained improvement before recommending RESTORE.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final

from .regimes import RegimeProfile, SystemRegime
from .state_vector import StateVector, clamp

__all__ = [
    "RecoveryAction",
    "RecoveryPlanner",
    "RecoveryVector",
]


class RecoveryAction(str, Enum):
    """What URE recommends be done, in ascending order of capability removed."""

    NONE = "NONE"
    RESTORE = "RESTORE"
    DEGRADE = "DEGRADE"
    THROTTLE = "THROTTLE"
    ISOLATE = "ISOLATE"
    QUARANTINE = "QUARANTINE"

    @property
    def disruption(self) -> int:
        """0 (no effect) .. 5 (stop serving). RESTORE ranks low: it removes nothing."""
        return _DISRUPTION[self]


_DISRUPTION: Final[dict[RecoveryAction, int]] = {
    RecoveryAction.NONE: 0,
    RecoveryAction.RESTORE: 1,
    RecoveryAction.DEGRADE: 2,
    RecoveryAction.THROTTLE: 3,
    RecoveryAction.ISOLATE: 4,
    RecoveryAction.QUARANTINE: 5,
}


@dataclass(frozen=True, slots=True)
class RecoveryVector:
    """A concrete, machine-readable recovery instruction.

    The ``controls`` mapping is what an executor (FORTRESS) actually applies.
    Keys are stable across releases so a control plane can switch on them.
    """

    action: RecoveryAction
    #: How aggressively to apply the action, in [0, 1].
    intensity: float
    #: Named control-plane toggles, e.g. {"reduce_rate_limit": True}.
    controls: Mapping[str, object]
    #: Human-readable justification, carried into the audit record.
    rationale: str
    #: Scope hint: which tenant, path, or subsystem the action applies to.
    scope: str = "global"

    @classmethod
    def noop(cls) -> RecoveryVector:
        return cls(
            action=RecoveryAction.NONE,
            intensity=0.0,
            controls={},
            rationale="System within nominal operating envelope; no action required.",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "intensity": round(self.intensity, 4),
            "controls": dict(self.controls),
            "rationale": self.rationale,
            "scope": self.scope,
        }


class RecoveryPlanner:
    """Maps a classified state onto a recovery recommendation.

    Stateful only in the hysteresis history, which records recent
    recommendations so de-escalation can require sustained improvement.
    """

    __slots__ = ("_applied", "_applied_vector", "_proposals", "_restore_patience")

    def __init__(self, restore_patience: int = 3, history: int = 16) -> None:
        if restore_patience < 1:
            raise ValueError("restore_patience must be at least 1")
        self._restore_patience = restore_patience
        # What conditions *suggested*, not what was applied. Recording the
        # held action here instead would make the history self-reinforcing:
        # every hold would re-assert the elevated action as evidence for
        # continuing to hold it, and controls could never be released.
        self._proposals: deque[RecoveryAction] = deque(
            maxlen=max(history, restore_patience)
        )
        self._applied: RecoveryAction = RecoveryAction.NONE
        self._applied_vector: RecoveryVector = RecoveryVector.noop()

    def plan(
        self,
        state: StateVector,
        profile: RegimeProfile,
        energy: float,
        derivative: float,
        resilience: float,
        *,
        campaign: bool = False,
        scope: str = "global",
    ) -> RecoveryVector:
        """Produce a recovery recommendation for the current assessment."""
        proposed = self._propose(
            state, profile, energy, derivative, resilience, campaign, scope
        )
        self._proposals.append(proposed.action)
        return self._damp(proposed)

    def _propose(
        self,
        state: StateVector,
        profile: RegimeProfile,
        energy: float,
        derivative: float,
        resilience: float,
        campaign: bool,
        scope: str,
    ) -> RecoveryVector:
        regime = profile.dominant
        rising = derivative > 0.02
        falling = derivative < -0.02

        if regime is SystemRegime.CASCADING:
            # Failures inducing failures. Preserve the system and the evidence.
            intensity = clamp(0.65 + 0.35 * (1.0 - resilience))
            return RecoveryVector(
                action=RecoveryAction.QUARANTINE,
                intensity=intensity,
                controls={
                    "stop_admission": True,
                    "reduce_rate_limit": True,
                    "disable_learning": True,
                    "freeze_vaccines": True,
                    "preserve_state": True,
                    "page_operator": True,
                },
                rationale=(
                    f"CASCADING: energy {energy:.2f} rising at {derivative:+.4f}/s with "
                    f"resilience {resilience:.2f}. Failures are inducing failures; "
                    f"admission stopped and state preserved for forensics."
                ),
                scope=scope,
            )

        if regime is SystemRegime.ATTACKED:
            # Isolate, do not throttle: throttling delivers the denial of
            # service the adversary is trying to cause.
            intensity = clamp(0.5 + 0.5 * max(state.threat_pressure, state.adversarial_pressure))
            return RecoveryVector(
                action=RecoveryAction.ISOLATE,
                intensity=intensity,
                controls={
                    "isolate_scope": True,
                    "tighten_threshold": True,
                    # Learning stays ON under attack. This is the moment the
                    # highest-value adversarial signal is available, and an
                    # engine that stops recording exactly when it is being
                    # attacked can never become historically aware. Poisoning
                    # is defended against inside BVE instead -- by the
                    # discrimination threshold at synthesis and the
                    # effectiveness feedback loop -- not by going blind.
                    "disable_learning": False,
                    "freeze_vaccines": False,
                    "preserve_state": True,
                    "reduce_rate_limit": False,
                },
                rationale=(
                    f"ATTACKED: threat {state.threat_pressure:.2f} / adversarial "
                    f"{state.adversarial_pressure:.2f}"
                    + (" (recurring campaign)" if campaign else "")
                    + ". Isolating scope and tightening thresholds; rate limiting "
                    "withheld deliberately so the defence does not complete the "
                    "denial of service. Learning remains active to capture the "
                    "attack signature."
                ),
                scope=scope,
            )

        if regime is SystemRegime.STRESSED:
            # Load, not malice. Throttling is exactly right here.
            heavy = energy >= 1.0 or state.resource_pressure > 0.8
            intensity = clamp(0.35 + 0.65 * state.resource_pressure)
            return RecoveryVector(
                action=RecoveryAction.THROTTLE if heavy else RecoveryAction.DEGRADE,
                intensity=intensity,
                controls={
                    "reduce_rate_limit": heavy,
                    "shed_optional_work": True,
                    "increase_timeout_budget": True,
                    "disable_learning": False,
                },
                rationale=(
                    f"STRESSED: operational pressure (latency {state.latency_pressure:.2f}, "
                    f"failure {state.failure_pressure:.2f}, resource "
                    f"{state.resource_pressure:.2f}) at energy {energy:.2f}. "
                    f"{'Throttling admission' if heavy else 'Shedding optional work'}; "
                    "this is load, not an adversary."
                ),
                scope=scope,
            )

        if regime is SystemRegime.RECOVERING and falling:
            return RecoveryVector(
                action=RecoveryAction.RESTORE,
                intensity=clamp(0.3 + 0.4 * resilience),
                controls={
                    "relax_rate_limit": True,
                    "resume_optional_work": True,
                    "resume_learning": True,
                },
                rationale=(
                    f"RECOVERING: energy {energy:.2f} dissipating at {derivative:+.4f}/s. "
                    "Walking controls back toward normal operation."
                ),
                scope=scope,
            )

        if regime is SystemRegime.UNKNOWN:
            # Cannot classify. Degrade rather than assume health: the engine
            # not knowing its own state is itself a reason for caution.
            return RecoveryVector(
                action=RecoveryAction.DEGRADE,
                intensity=0.4,
                controls={"shed_optional_work": True, "page_operator": True},
                rationale=(
                    "UNKNOWN regime: the classifier found no supported hypothesis. "
                    "Degrading conservatively rather than assuming health."
                ),
                scope=scope,
            )

        if profile.ambiguous and rising:
            # Genuinely uncertain and getting worse. A cheap, reversible action
            # is the right response to low-confidence bad news.
            return RecoveryVector(
                action=RecoveryAction.DEGRADE,
                intensity=0.3,
                controls={"shed_optional_work": True},
                rationale=(
                    f"Ambiguous classification ({profile.dominant.value} at "
                    f"{profile.confidence:.2f} confidence, margin {profile.margin:.2f}) "
                    f"with energy rising at {derivative:+.4f}/s. Taking a cheap, "
                    "reversible precaution."
                ),
                scope=scope,
            )

        return RecoveryVector.noop()

    def _damp(self, proposed: RecoveryVector) -> RecoveryVector:
        """Apply hysteresis: escalate immediately, de-escalate only on evidence."""
        # Escalation always passes straight through.
        if proposed.action.disruption >= self._applied.disruption:
            self._applied = proposed.action
            self._applied_vector = proposed
            return proposed

        # De-escalating. Release only once the last `restore_patience`
        # observations have *all* agreed that less intervention is warranted.
        recent = list(self._proposals)[-self._restore_patience :]
        if len(recent) < self._restore_patience or any(
            action.disruption > proposed.action.disruption for action in recent
        ):
            return self._hold(proposed)

        self._applied = proposed.action
        self._applied_vector = proposed
        return proposed

    def _hold(self, proposed: RecoveryVector) -> RecoveryVector:
        """Keep the controls currently in force, and say why.

        The held vector carries the *applied* action's controls, not the
        proposed action's. Mixing them would hand an executor a vector whose
        action says QUARANTINE while its controls say resume admission, and
        the executor would obey the controls.
        """
        held = self._applied_vector
        return RecoveryVector(
            action=held.action,
            intensity=held.intensity,
            controls=held.controls,
            rationale=(
                f"Holding {held.action.value} (hysteresis): conditions suggest "
                f"{proposed.action.value}, but improvement has not been sustained "
                "long enough to release controls without risking oscillation."
            ),
            scope=proposed.scope,
        )

    @property
    def current_action(self) -> RecoveryAction:
        """The action currently in force, after hysteresis."""
        return self._applied

    def recovery_score(self) -> float:
        """How un-degraded the system currently is, in [0, 1].

        Feeds the "recovery" term of the resilience index: a system only
        staying healthy because half its capability is switched off is not
        resilient, and this term is what stops the index from claiming it is.
        """
        return clamp(1.0 - self._applied.disruption / 5.0)

    def reset(self) -> None:
        self._proposals.clear()
        self._applied = RecoveryAction.NONE
        self._applied_vector = RecoveryVector.noop()
