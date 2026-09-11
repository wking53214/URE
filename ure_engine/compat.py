"""
compat.py — the legacy GSA interface, satisfied exactly.
========================================================

This module closes the gap the whole project began with. GSA's gateway
contained the line::

    from ure_engine import ClassificationResult, GatewayHealthEngine

and no such module existed. Everything else in this package is the
next-generation engine that line *should* have been importing. This module is
what makes that import work today, against the interface GSA actually wrote.

The legacy surface
------------------
GSA calls::

    summary = health_engine.assess(reject_rate=, retry_rate=, latency_ms=, backlog=)

and reads::

    summary.result_status        (.value in NEUTRAL / REGRESSIVE / RISK_INCREASING)
    summary.dominant_regime      (.value)
    summary.composite_risk_score (float)
    summary.calculated_energy    (float)
    summary.regime_entropy       (float)
    summary.regime_confidence    (float)
    summary.rationale_statement  (str)

Mapped vs. derived
------------------
Mapped directly (URE produces these natively):

* ``dominant_regime``      <- the classified regime
* ``calculated_energy``    <- Lyapunov energy V(x)
* ``regime_entropy``       <- distribution entropy, which the soft classifier
  computes for real. The predecessor had this field but nothing to fill it
  with; here it is a genuine measure of classifier ambiguity.
* ``regime_confidence``    <- the dominant regime's probability mass

Derived:

* ``composite_risk_score`` <- energy normalized to [0, 1]. The architecture
  replaces this metric with ``resilience_index``, and the two are not
  inverses, so this is an honest stand-in for legacy consumers rather than a
  claim of equivalence. New integrations should read ``resilience_index``.
* ``result_status``        <- the regime's three-state projection, with a
  security floor described below.

The security floor
------------------
The regime model's ``health_status`` treats mid-band regimes as NEUTRAL,
which is right for general operational health: a system working harder than
baseline is not yet a problem. For a security *gateway* it is not right. A
sustained elevated composite risk is not neutral even before it reaches the
ATTACKED band, and GSA's health signal exists to surface exactly that.

So status is derived from composite risk, floored by the engine's own
health_status: this layer will report a condition as *worse* than the engine
does, never calmer. Reporting calmer than the underlying engine would make the
compatibility layer a place where warnings go to be lost.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .engine import UREConfig, UREEngine
from .governance_adapter import GSATelemetryAdapter
from .lyapunov import ENERGY_MAX
from .state_vector import clamp

__all__ = [
    "ClassificationResult",
    "CompatGatewayHealthEngine",
    "ResultStatus",
]


@dataclass(frozen=True, slots=True)
class ResultStatus:
    """A minimal enum-alike so legacy ``.value`` access resolves.

    Deliberately not an :class:`enum.Enum`: GSA compares these by ``.value``
    string, and a real enum would add a closed membership constraint that the
    legacy call sites never had.
    """

    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Mirrors the field names GSA's gateway reads off a URE summary.

    Field order and names match the archive's specification exactly. The
    ``assessment`` attachment is additive: legacy consumers ignore it, and a
    caller migrating to URE v2 can reach the full assessment through it
    without changing its call site first.
    """

    result_status: ResultStatus
    dominant_regime: ResultStatus
    composite_risk_score: float
    calculated_energy: float
    regime_entropy: float
    regime_confidence: float
    rationale_statement: str
    #: The modern assessment this result was projected from. Migration path.
    assessment: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_status": self.result_status.value,
            "dominant_regime": self.dominant_regime.value,
            "composite_risk_score": round(self.composite_risk_score, 4),
            "calculated_energy": round(self.calculated_energy, 4),
            "regime_entropy": round(self.regime_entropy, 4),
            "regime_confidence": round(self.regime_confidence, 4),
            "rationale_statement": self.rationale_statement,
        }


#: Composite-risk bands for the derived legacy status.
_REGRESSIVE_AT: float = 0.40
_RISK_INCREASING_AT: float = 0.20


class CompatGatewayHealthEngine:
    """Drop-in replacement for the engine GSA was written against.

    Construction signature matches the original (``backlog_capacity``), and
    :meth:`assess` matches the original call. Internally this is the full URE
    v2 pipeline, so a gateway that swaps this in gets trajectory analysis,
    AMX, BVE, and adaptive thresholds without changing a line -- it simply
    cannot *see* them through this interface.

    Composition rather than inheritance: subclassing the engine would expose
    both interfaces on one object and invite call sites to drift between them,
    which is how the two-interface problem started.
    """

    __slots__ = ("_adapter", "_engine", "backlog_capacity")

    def __init__(
        self,
        backlog_capacity: int = 256,
        *,
        engine: UREEngine | None = None,
        config: UREConfig | None = None,
    ) -> None:
        self.backlog_capacity = backlog_capacity
        self._engine = engine or UREEngine(config or UREConfig(queue_capacity=backlog_capacity))
        self._adapter = GSATelemetryAdapter(backlog_capacity=backlog_capacity)

    @property
    def engine(self) -> UREEngine:
        """The underlying URE engine, for callers ready to migrate."""
        return self._engine

    def assess(
        self,
        reject_rate: float,
        retry_rate: float,
        latency_ms: float,
        backlog: int,
    ) -> ClassificationResult:
        """The legacy call. Returns the legacy shape, computed the modern way."""
        frame = self._adapter.build_frame(
            reject_rate=reject_rate,
            retry_rate=retry_rate,
            latency_ms=latency_ms,
            backlog=backlog,
        )
        assessment = self._engine.observe(frame)

        energy = float(assessment.lyapunov_energy)
        composite_risk = clamp(energy / ENERGY_MAX)

        engine_status = assessment.regime.health_status
        if composite_risk >= _REGRESSIVE_AT:
            status = "REGRESSIVE"
        elif composite_risk >= _RISK_INCREASING_AT:
            status = "RISK_INCREASING"
        else:
            status = engine_status

        # Never report calmer than the engine itself.
        if engine_status == "REGRESSIVE":
            status = "REGRESSIVE"
        elif engine_status == "RISK_INCREASING" and status == "NEUTRAL":
            status = "RISK_INCREASING"

        return ClassificationResult(
            result_status=ResultStatus(status),
            dominant_regime=ResultStatus(assessment.regime.value),
            composite_risk_score=composite_risk,
            calculated_energy=energy,
            regime_entropy=float(assessment.regime_entropy),
            regime_confidence=float(assessment.regime_confidence),
            rationale_statement=assessment.explanation,
            assessment=assessment,
        )

    # The modern names, so a call site can migrate incrementally.
    def classify(self, telemetry: Mapping[str, Any]) -> Any:
        """Modern entry point: full :class:`~ure_engine.assessment.UREAssessment`."""
        return self._engine.observe(telemetry)

    async def evaluate(self, state: Any) -> Any:
        """Async entry point satisfying :class:`~ure_engine.engine.GatewayHealthEngine`."""
        return await self._engine.evaluate(state)


def binary_entropy(p: float) -> float:
    """Normalized binary Shannon entropy of the split ``(p, 1-p)``.

    Retained from the archive's compatibility shim, where it stood in for a
    real distribution entropy the old engine could not compute. URE now
    computes the genuine six-way entropy, so this is exported only for
    callers that were importing it.
    """
    p = min(1.0, max(0.0, p))
    if p in (0.0, 1.0):
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
