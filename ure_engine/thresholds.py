"""
thresholds.py — Adaptive Threshold Controller v2.
=================================================

The gateway blocks a request when its policy score exceeds a threshold. In the
predecessor that threshold was the constant ``0.80``, which is the wrong shape
for the problem: the same score means different things depending on what the
system is currently absorbing. A 0.75 request during a quiet Tuesday is worth
letting through and logging. The identical request arriving during a cascading
failure is not.

The architecture calls for::

    threshold = f(regime, stability, attack_pressure, recovery_state)

with the calibration anchors::

    NOMINAL    0.85
    ATTACKED   0.45
    CASCADING  0.20

Those three anchors set the shape: the threshold falls faster than linearly as
severity rises, because the cost of a false negative grows much faster than the
cost of a false positive once a system is already in trouble.

Relationship to the policy seam
-------------------------------
This controller sets the threshold; it does not produce the score. Policies
produce scores and remain entirely ignorant of the threshold, exactly as the
seam requires. A policy that asserts ``hard_block`` bypasses the threshold
entirely and this module never sees it -- absolute rules stay absolute no
matter how calm the system is.

False-discovery feedback
------------------------
The predecessor GSA tuned its threshold against a false-discovery-rate signal.
That mechanism is preserved here as an additive trim on top of the regime term,
bounded so feedback can nudge the threshold but never override the regime. A
noisy FDR signal during an incident must not be able to relax a threshold that
the regime tightened.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from .regimes import SystemRegime
from .state_vector import clamp

__all__ = ["AdaptiveThresholdController", "ThresholdDecision"]

#: Regime anchors. NOMINAL / ATTACKED / CASCADING are from the architecture;
#: the rest are interpolated to keep the ladder monotone in severity.
REGIME_ANCHORS: Final[Mapping[SystemRegime, float]] = {
    SystemRegime.NOMINAL: 0.85,
    SystemRegime.ADAPTING: 0.75,
    SystemRegime.RECOVERING: 0.70,
    SystemRegime.STRESSED: 0.60,
    SystemRegime.ATTACKED: 0.45,
    SystemRegime.CASCADING: 0.20,
    # An engine that cannot classify itself gets the STRESSED threshold: not
    # panicked, but not extending nominal trust either.
    SystemRegime.UNKNOWN: 0.60,
}

#: Hard bounds. Even a perfectly calm system keeps some floor, and even a
#: collapsing one does not block literally everything on score alone.
THRESHOLD_FLOOR: Final[float] = 0.15
THRESHOLD_CEILING: Final[float] = 0.95

#: Maximum magnitude of the false-discovery-rate trim.
MAX_FDR_TRIM: Final[float] = 0.08


@dataclass(frozen=True, slots=True)
class ThresholdDecision:
    """The computed threshold and the reasoning behind it."""

    threshold: float
    regime_anchor: float
    stability_adjustment: float
    attack_adjustment: float
    recovery_adjustment: float
    fdr_trim: float
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": round(self.threshold, 4),
            "regime_anchor": round(self.regime_anchor, 4),
            "adjustments": {
                "stability": round(self.stability_adjustment, 4),
                "attack": round(self.attack_adjustment, 4),
                "recovery": round(self.recovery_adjustment, 4),
                "fdr": round(self.fdr_trim, 4),
            },
            "rationale": self.rationale,
        }


class AdaptiveThresholdController:
    """Computes the block threshold from system state rather than a constant."""

    __slots__ = ("_fdr_target", "_outcomes", "_trim")

    def __init__(self, fdr_target: float = 0.05, feedback_window: int = 256) -> None:
        if not 0.0 < fdr_target < 1.0:
            raise ValueError("fdr_target must be in (0, 1)")
        self._fdr_target = fdr_target
        self._outcomes: deque[bool] = deque(maxlen=feedback_window)
        self._trim = 0.0

    def compute(
        self,
        *,
        regime: SystemRegime,
        resilience: float,
        attack_pressure: float,
        recovering: bool,
    ) -> ThresholdDecision:
        """Derive the threshold for the current moment."""
        anchor = REGIME_ANCHORS.get(regime, 0.60)

        # Low resilience tightens further: a fragile system has less margin for
        # a false negative than a healthy one in the same regime.
        stability_adjustment = -0.15 * (1.0 - clamp(resilience))

        # Sustained adversarial pressure tightens beyond the regime anchor,
        # capped so it cannot on its own drive the threshold to the floor.
        attack_adjustment = -0.20 * clamp(attack_pressure)

        # A recovering system relaxes slightly, but never above its anchor:
        # relaxation during recovery is what causes the second spike.
        recovery_adjustment = 0.05 if recovering else 0.0

        base = anchor + stability_adjustment + attack_adjustment + recovery_adjustment
        threshold = clamp(base + self._trim, THRESHOLD_FLOOR, THRESHOLD_CEILING)

        return ThresholdDecision(
            threshold=threshold,
            regime_anchor=anchor,
            stability_adjustment=stability_adjustment,
            attack_adjustment=attack_adjustment,
            recovery_adjustment=recovery_adjustment,
            fdr_trim=self._trim,
            rationale=(
                f"{regime.value} anchor {anchor:.2f}; resilience {resilience:.2f} "
                f"({stability_adjustment:+.3f}), attack pressure {attack_pressure:.2f} "
                f"({attack_adjustment:+.3f}), recovery ({recovery_adjustment:+.3f}), "
                f"FDR trim ({self._trim:+.3f}) -> {threshold:.3f}"
            ),
        )

    # -- false-discovery feedback -----------------------------------------

    def record_outcome(self, was_false_positive: bool) -> None:
        """Report whether a block turned out to be a false positive.

        Only *blocks* should be reported. Feeding allowed requests in would
        make the denominator the whole traffic stream and the measured rate
        would drift toward zero regardless of how the gate is performing.
        """
        self._outcomes.append(was_false_positive)
        self._recompute_trim()

    def _recompute_trim(self) -> None:
        """Nudge the trim toward the false-discovery target.

        Deliberately a small proportional step rather than a solve: this signal
        is noisy, arrives in bursts, and reflects the threshold from several
        seconds ago. A controller that chases it aggressively oscillates.
        """
        if len(self._outcomes) < 16:
            self._trim = 0.0
            return
        observed = sum(self._outcomes) / len(self._outcomes)
        error = observed - self._fdr_target
        # Too many false positives -> raise the threshold (positive trim).
        self._trim = clamp(self._trim + error * 0.10, -MAX_FDR_TRIM, MAX_FDR_TRIM)

    @property
    def observed_fdr(self) -> float:
        """Measured false-discovery rate over the feedback window."""
        if not self._outcomes:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    @property
    def trim(self) -> float:
        return self._trim

    def reset(self) -> None:
        self._outcomes.clear()
        self._trim = 0.0
