"""
URE, the Universal Resilience Engine.
======================================

URE is the resilience subsystem of the AEGIS architecture. It sits behind a
governance gateway and answers a different question than a risk scorer does.
A risk scorer asks "how dangerous is this request". URE asks:

    Given everything happening to this system right now and everything it
    remembers, what state is it in, where is it heading, how much can it still
    absorb, and what should be done about it?

It computes **stability, trajectory, regime, resilience, recovery, and
adaptation** -- not a risk score.

Quick start
-----------
::

    from ure_engine import UREEngine

    engine = UREEngine()
    assessment = engine.observe({
        "risk_score": 0.9,
        "blocked_rate": 0.7,
        "retry_rate": 0.1,
        "latency_ms": 60,
        "queue_saturation": 0.2,
        "signatures": ["sig-a1b2"],
        "markers": ["ignore_previous", "developer_override"],
    })

    assessment.regime               # SystemRegime.ATTACKED
    assessment.resilience_index     # 0.0 .. 1.0
    assessment.recommended_action   # RecoveryAction.ISOLATE
    assessment.recommended_threshold
    print(assessment.explanation)

Legacy GSA integration
----------------------
The gateway's original, long-unsatisfiable import works as written::

    from ure_engine import ClassificationResult, GatewayHealthEngine

``GatewayHealthEngine`` resolves to the compatibility engine, which exposes the
legacy ``assess(reject_rate, retry_rate, latency_ms, backlog)`` call and is
driven by the full v2 pipeline underneath. The structural protocol of the same
name is available as :class:`GatewayHealthEngineProtocol` for type checking.

Architecture
------------
::

    AEGIS
     |-- GSA        Governance Gateway
     |-- URE        Universal Resilience Engine     <- this package
     |-- LTE        Lyapunov Trajectory Engine
     |-- AMX        Attack Memory Exchange
     |-- BVE        Behavioral Vaccine Engine
     |-- FORTRESS   Recovery Authority
     |-- WS3        Universal Telemetry Layer
     `-- Executive Dashboard

See ``docs/ARCHITECTURE.md`` for the full design and ``docs/PROVENANCE.md`` for
where each component came from.
"""

from __future__ import annotations

from .assessment import GovernanceDecision, UREAssessment
from .attack_memory import AttackMemoryExchange, AttackProfile, MemoryRecall
from .compat import ClassificationResult, CompatGatewayHealthEngine, ResultStatus
from .engine import GatewayHealthEngine as GatewayHealthEngineProtocol
from .engine import UREConfig, UREEngine
from .governance_adapter import GovernanceOutcome, GSATelemetryAdapter
from .lyapunov import (
    ENERGY_MAX,
    ENERGY_WEIGHTS,
    LyapunovTrajectoryEngine,
    compute_energy,
    energy_contributions,
)
from .recovery import RecoveryAction, RecoveryPlanner, RecoveryVector
from .regimes import RegimeClassifier, RegimeProfile, SystemRegime
from .resilience import ResilienceBreakdown, ResilienceIndex
from .state_vector import PRESSURE_NAMES, PolicyTelemetry, StateVector, clamp
from .telemetry import SmoothingAdapter, TelemetryAdapter, TelemetryFrame
from .thresholds import AdaptiveThresholdController, ThresholdDecision
from .trajectory import TrajectoryEngine, TrajectorySnapshot
from .vaccines import (
    BehavioralVaccine,
    BehavioralVaccineEngine,
    Immunization,
    Observation,
)

__version__ = "2.0.0"

#: The name GSA imports. Bound to the concrete compatibility engine so the
#: legacy `from ure_engine import GatewayHealthEngine` resolves to something
#: constructible; the structural Protocol is GatewayHealthEngineProtocol.
GatewayHealthEngine = CompatGatewayHealthEngine

__all__ = [
    "__version__",
    # engine
    "UREEngine",
    "UREConfig",
    "UREAssessment",
    "GovernanceDecision",
    # state
    "StateVector",
    "PolicyTelemetry",
    "PRESSURE_NAMES",
    "clamp",
    # lyapunov / trajectory
    "LyapunovTrajectoryEngine",
    "TrajectoryEngine",
    "TrajectorySnapshot",
    "compute_energy",
    "energy_contributions",
    "ENERGY_MAX",
    "ENERGY_WEIGHTS",
    # regimes
    "SystemRegime",
    "RegimeClassifier",
    "RegimeProfile",
    # memory and learning
    "AttackMemoryExchange",
    "AttackProfile",
    "MemoryRecall",
    "BehavioralVaccineEngine",
    "BehavioralVaccine",
    "Immunization",
    "Observation",
    # recovery and resilience
    "RecoveryAction",
    "RecoveryPlanner",
    "RecoveryVector",
    "ResilienceIndex",
    "ResilienceBreakdown",
    "AdaptiveThresholdController",
    "ThresholdDecision",
    # telemetry
    "TelemetryFrame",
    "TelemetryAdapter",
    "SmoothingAdapter",
    # governance
    "GSATelemetryAdapter",
    "GovernanceOutcome",
    # legacy compatibility
    "GatewayHealthEngine",
    "GatewayHealthEngineProtocol",
    "CompatGatewayHealthEngine",
    "ClassificationResult",
    "ResultStatus",
]
