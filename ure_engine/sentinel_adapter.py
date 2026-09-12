"""
sentinel_adapter.py, sentinel_os telemetry into URE's state space.
==================================================================

sentinel_os already measures the two things URE needs and has nowhere to send
them. ``CircuitBreaker.snapshot()`` knows what is being rejected and what is
failing; ``TransmissionQueue.stats()`` knows what is backing up and how stale
it has become. Neither surface draws a conclusion. This adapter projects them
onto URE's six pressures so something can.

URE does not import sentinel_os and sentinel_os does not import URE. The
contract is the shape of two mappings, which is the same dependency inversion
``governance_adapter.py`` uses for GSA: the resilience engine must not know
what it is watching, or it can only ever watch one thing.

THE COUNTER PROBLEM
-------------------
sentinel_os's counters are lifetime cumulative. URE wants rates on [0, 1].

A naive ``total_failures / total_calls`` is a lifetime average, and a lifetime
average converges to a constant and then stops responding to anything: after a
million healthy calls, a thousand consecutive failures barely move it. The
system would be visibly on fire and the number would be fine.

So the adapter holds the previous sample and differentiates. That is the single
most important thing in this file, and skipping it is the most likely way to
get a confidently wrong assessment out of a correct engine.

THE SCALE PROBLEM
-----------------
URE's own ``TelemetryAdapter`` treats 1000ms as full latency pressure, which is
right for the LLM gateway it was written for. sentinel_os's
``oldest_pending_age_s`` is queue staleness measured in seconds to tens of
seconds. Passing it through raw saturates URE at roughly one second of queue
age, and a 7% load ramp reads STRESSED.

Every ceiling on :class:`SentinelConfig` exists for that reason: they are the
deployment's scales, not tuning knobs in the usual sense, and a deployment that
inherits the defaults without checking them against its own queue will get
assessments that are precisely computed from the wrong units.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .state_vector import clamp
from .telemetry import TelemetryFrame

__all__ = ["SentinelAdapter", "SentinelConfig"]


@dataclass(frozen=True, slots=True)
class SentinelConfig:
    """The deployment's scales. Check every one against the real system."""

    #: Ready-queue depth treated as full resource saturation.
    queue_capacity: int = 128
    #: ``oldest_pending_age_s`` treated as full latency pressure.
    latency_ceiling_s: float = 60.0
    #: Dead-letter arrivals per minute treated as full failure pressure.
    dlq_rate_ceiling: float = 20.0
    #: ``processing_overdue`` count treated as the reaper being fully behind.
    overdue_ceiling: int = 20


class SentinelAdapter:
    """Differentiates cumulative counters and projects them onto URE pressures.

    Stateful: it remembers the previous counter sample. One adapter instance per
    monitored breaker, and do not share one across scopes, or the differences
    are taken between unrelated systems.
    """

    __slots__ = ("_config", "_previous")

    def __init__(self, config: SentinelConfig | None = None) -> None:
        self._config = config or SentinelConfig()
        self._previous: dict[str, int] | None = None

    @property
    def config(self) -> SentinelConfig:
        return self._config

    def _rates(self, snapshot: Mapping[str, Any]) -> tuple[float, float]:
        """``(blocked_rate, error_rate)`` over the interval since the last sample."""
        current = {
            "calls": int(snapshot.get("total_calls", 0)),
            "failures": int(snapshot.get("total_failures", 0)),
            "rejections": int(snapshot.get("total_rejections", 0)),
        }
        previous, self._previous = self._previous, current

        # First sample establishes the baseline. Reporting a lifetime average
        # here as if it were current is the failure this class exists to avoid,
        # so it reports nothing instead.
        if previous is None:
            return 0.0, 0.0

        delta_calls = current["calls"] - previous["calls"]
        delta_failures = current["failures"] - previous["failures"]
        delta_rejections = current["rejections"] - previous["rejections"]

        # Rejections never reach the callable, so they are absent from
        # total_calls and have to be added back to get attempts.
        attempts = delta_calls + delta_rejections
        if attempts <= 0:
            return 0.0, 0.0
        return (
            clamp(delta_rejections / attempts),
            clamp(delta_failures / max(1, delta_calls)),
        )

    def to_frame(
        self,
        snapshot: Mapping[str, Any],
        stats: Mapping[str, Any],
        *,
        dlq_per_minute: float = 0.0,
        injection_hits: float = 0.0,
    ) -> TelemetryFrame:
        """Build one URE telemetry frame from one sentinel_os poll."""
        config = self._config
        blocked_rate, error_rate = self._rates(snapshot)

        depth = float(stats.get("depth_ready", 0) or 0)
        overdue = float(stats.get("processing_overdue", 0) or 0)
        age_s = float(stats.get("oldest_pending_age_s") or 0.0)

        # FAILURE: real errors plus dead-letter arrival rate. The circuit-state
        # step is deliberately NOT added here. URE's own TelemetryAdapter
        # already applies it from `circuit_state`, and adding it in both places
        # double-counts an open breaker into failure pressure.
        failure = clamp(error_rate + clamp(dlq_per_minute / config.dlq_rate_ceiling) * 0.5)

        # RESOURCE: queue backlog plus the reaper falling behind. Two
        # independent ways to run out, so they sum rather than average.
        resource = (
            depth / config.queue_capacity
            + clamp(overdue / config.overdue_ceiling) * 0.5
        )

        return TelemetryFrame(
            # THREAT. URE weights this as risk_score * 0.7 + blocked_rate * 0.3,
            # which assumes risk_score is a per-request policy score that spikes
            # hard. sentinel_os has no such score at this surface; it has a
            # rejection RATE. Routing rejections only into blocked_rate gives
            # them a 0.3 weight, and a sustained 30% rejection rate then reads
            # NOMINAL, which is the exact failure GSATelemetryAdapter was
            # written to fix for the GSA gateway. From a gateway's point of
            # view a sustained rejection rate IS the threat signal.
            risk_score=clamp(injection_hits + blocked_rate * 0.5),
            blocked_rate=blocked_rate,
            retry_rate=failure,
            # Pre-normalized to URE's millisecond scale; see the module
            # docstring on why passing age_s through raw does not work.
            latency_ms=clamp(age_s / config.latency_ceiling_s) * 1000.0,
            queue_saturation=clamp(resource),
            substrate_health=1.0,
            circuit_state=str(snapshot.get("state", "closed")).upper(),
            error_rate=error_rate,
            # sentinel_os exposes no distribution signal at this surface, and
            # reporting a guess as a measurement is worse than reporting none.
            drift_metric=0.0,
        )

    def reset(self) -> None:
        """Forget the previous sample, so the next one re-establishes baseline."""
        self._previous = None
