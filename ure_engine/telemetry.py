"""
telemetry.py, projecting raw signals onto the pressure state space.
====================================================================

This is the only place in URE that knows what a "queue" or a "circuit breaker"
is. Everything downstream sees six normalized pressures and nothing else. That
containment is what lets the same engine sit behind an LLM gateway, a payments
API, or a drone control link: swap the adapter, keep the engine.

WS3, the universal telemetry layer in the AEGIS hierarchy, feeds events,
metrics, anomalies, and attestations in through :class:`TelemetryFrame`.

Two adapters ship here
----------------------
:class:`TelemetryAdapter`
    The general mapping from a telemetry dict to a StateVector. Keys are all
    optional and default safely, so a partial feed degrades to a partial
    picture rather than an exception.

:class:`SmoothingAdapter`
    Wraps any adapter with an exponential moving average. Gateway telemetry is
    spiky at request granularity; a single 900 ms outlier should move the
    latency pressure, not define it.

A note on the units problem
---------------------------
The predecessor mixed rates in [0, 1] with unbounded counts in one weighted
sum, so ``backlog_depth`` at 45 dominated every rate term and the weights had
to be re-tuned per deployment. Every mapping below normalizes first and is
documented with the scale it assumes, so a deployment with different capacity
changes a constructor argument instead of the physics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .state_vector import PolicyTelemetry, StateVector, clamp

__all__ = [
    "SmoothingAdapter",
    "TelemetryAdapter",
    "TelemetryFrame",
    "TelemetrySource",
]


@dataclass(slots=True)
class TelemetryFrame:
    """One observation window's worth of signals from the host system.

    Every field is optional. A caller that only has latency and an error rate
    gets a coherent (if partial) state vector rather than a validation error --
    telemetry completeness is not something a resilience engine gets to demand
    of a system that is currently on fire.
    """

    #: Policy risk score for the request, [0, 1].
    risk_score: float = 0.0
    #: Fraction of recent requests blocked, [0, 1].
    blocked_rate: float = 0.0
    #: Fraction of recent requests retried, [0, 1].
    retry_rate: float = 0.0
    #: Observed latency in milliseconds.
    latency_ms: float = 0.0
    #: Normalized latency variability, [0, 1].
    latency_volatility: float = 0.0
    #: Work queue fill fraction, [0, 1].
    queue_saturation: float = 0.0
    #: Substrate health, 1.0 healthy .. 0.0 dead.
    substrate_health: float = 1.0
    #: Circuit breaker state: CLOSED, HALF_OPEN, or OPEN.
    circuit_state: str = "CLOSED"
    #: Count of explicitly adversarial events in the window.
    adversarial_events: float = 0.0
    #: Distribution drift metric, [0, 1].
    drift_metric: float = 0.0
    #: Error rate, [0, 1].
    error_rate: float = 0.0
    #: Opaque signatures for AMX correlation.
    signatures: tuple[str, ...] = ()
    #: Opaque behavioural markers for BVE.
    markers: tuple[str, ...] = ()
    #: Optional policy outcome for this window.
    policy: PolicyTelemetry | None = None
    #: Anything the adapter does not recognize, preserved for audit.
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TelemetryFrame:
        """Build a frame from a loose dict, ignoring unknown keys safely.

        This is the ingestion path for WS3 and for the legacy compatibility
        layer, both of which hand URE plain dictionaries.
        """
        known = {
            "risk_score",
            "blocked_rate",
            "retry_rate",
            "latency_ms",
            "latency_volatility",
            "queue_saturation",
            "substrate_health",
            "circuit_state",
            "adversarial_events",
            "drift_metric",
            "error_rate",
        }
        return cls(
            risk_score=_as_float(data.get("risk_score")),
            blocked_rate=_as_float(data.get("blocked_rate")),
            retry_rate=_as_float(data.get("retry_rate")),
            latency_ms=_as_float(data.get("latency_ms")),
            latency_volatility=_as_float(data.get("latency_volatility")),
            queue_saturation=_as_float(data.get("queue_saturation")),
            substrate_health=_as_float(data.get("substrate_health"), default=1.0),
            circuit_state=str(data.get("circuit_state", "CLOSED")),
            adversarial_events=_as_float(data.get("adversarial_events")),
            drift_metric=_as_float(data.get("drift_metric")),
            error_rate=_as_float(data.get("error_rate")),
            signatures=tuple(str(s) for s in data.get("signatures", ()) or ()),
            markers=tuple(str(m) for m in data.get("markers", ()) or ()),
            policy=data.get("policy") if isinstance(data.get("policy"), PolicyTelemetry) else None,
            extra={
                k: v
                for k, v in data.items()
                if k not in known and k not in {"signatures", "markers", "policy"}
            },
        )


def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce to float, falling back to ``default`` for junk.

    Telemetry arrives from systems under stress; a ``None`` where a float was
    expected is a symptom, not a reason to raise inside the resilience engine.
    """
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result  # NaN check


class TelemetrySource(Protocol):
    """Anything that can project a telemetry frame into the state space."""

    def to_state_vector(self, frame: TelemetryFrame, timestamp: float) -> StateVector:
        ...


class TelemetryAdapter:
    """Maps a :class:`TelemetryFrame` onto the six pressures.

    Each mapping is documented with the assumption it encodes, because these
    constants are the engine's contact with reality and silent ones are how a
    resilience engine ends up confidently wrong.
    """

    __slots__ = ("_adversarial_scale", "_latency_ceiling_ms")

    def __init__(
        self,
        latency_ceiling_ms: float = 1000.0,
        adversarial_scale: float = 10.0,
    ) -> None:
        if latency_ceiling_ms <= 0:
            raise ValueError("latency_ceiling_ms must be positive")
        if adversarial_scale <= 0:
            raise ValueError("adversarial_scale must be positive")
        self._latency_ceiling_ms = latency_ceiling_ms
        self._adversarial_scale = adversarial_scale

    def to_state_vector(self, frame: TelemetryFrame, timestamp: float) -> StateVector:
        # THREAT: what the policy layer thinks of this request, plus how much
        # the gateway is currently rejecting. Weighted toward the score because
        # the block rate is a lagging consequence of it.
        threat = frame.risk_score * 0.7 + frame.blocked_rate * 0.3

        # LATENCY: absolute latency against a configured ceiling, plus
        # variability. Variability is added rather than averaged: a system with
        # acceptable mean latency and wild variance is worse off than its mean
        # suggests, and the mean is what hides that.
        latency = frame.latency_ms / self._latency_ceiling_ms + frame.latency_volatility

        # FAILURE: retries and errors, with an open circuit as a large step.
        # An open breaker is a discrete statement that something is broken,
        # not a rate, so it contributes a fixed 0.5 rather than scaling.
        circuit_penalty = {"OPEN": 0.5, "HALF_OPEN": 0.2}.get(
            frame.circuit_state.upper(), 0.0
        )
        failure = frame.retry_rate + frame.error_rate * 0.5 + circuit_penalty

        # DRIFT: reported directly by the host; URE has no view of the
        # underlying distribution and does not pretend to.
        drift = frame.drift_metric

        # ADVERSARIAL: explicit hostile events, normalized against a scale that
        # says "this many events in a window is saturation". AMX and BVE add to
        # this dimension downstream in the engine.
        adversarial = frame.adversarial_events / self._adversarial_scale

        # RESOURCE: queue fill plus substrate degradation. Both are already
        # fractions; summed because they are independent ways to run out.
        resource = frame.queue_saturation + (1.0 - clamp(frame.substrate_health))

        return StateVector.create(
            threat_pressure=threat,
            latency_pressure=latency,
            failure_pressure=failure,
            drift_pressure=drift,
            adversarial_pressure=adversarial,
            resource_pressure=resource,
            timestamp=timestamp,
        )


class SmoothingAdapter:
    """Exponential moving average over any other adapter.

    ``alpha`` is the weight given to the newest observation. 1.0 disables
    smoothing; smaller values make the engine steadier but slower to react.
    The default of 0.4 reaches ~87% of a step change within four samples, which
    is fast enough for regime transitions and slow enough to ignore one
    pathological request.
    """

    __slots__ = ("_alpha", "_inner", "_previous")

    def __init__(self, inner: TelemetrySource, alpha: float = 0.4) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._inner = inner
        self._alpha = alpha
        self._previous: StateVector | None = None

    def to_state_vector(self, frame: TelemetryFrame, timestamp: float) -> StateVector:
        raw = self._inner.to_state_vector(frame, timestamp)
        if self._previous is None:
            self._previous = raw
            return raw
        smoothed = self._previous.blend(raw, self._alpha)
        self._previous = smoothed
        return smoothed

    def reset(self) -> None:
        self._previous = None
