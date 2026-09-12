"""
engine.py, the URE orchestrator.
=================================

Wires the subsystems into the execution flow the architecture specifies::

    URE.observe()
        |-- State Vector Update
        |-- LTE Energy Calculation
        |-- Trajectory Update
        |-- AMX Correlation
        |-- BVE Activation
        |-- Regime Classification
        |-- Resilience Index Computation
        `-- Recovery Recommendation
              |
              v
         UREAssessment

Order matters, and not only for readability. AMX and BVE run *before* regime
classification because their output raises ``adversarial_pressure``, which
changes the energy the classifier sees. Running them afterwards would mean
memory and learning could annotate an assessment but never alter it, which is
precisely the dead-subsystem failure mode the predecessor build removed them
for. Here they are upstream of the decision or they are not wired at all.

Concurrency
-----------
:meth:`UREEngine.observe` is synchronous and cheap -- the whole pipeline is
arithmetic over bounded windows, with no I/O. :meth:`UREEngine.evaluate` is the
async entry point the ``GatewayHealthEngine`` protocol requires; it does not
spawn tasks, it simply satisfies the await-able contract so callers on an event
loop are not forced into a thread. A lock guards the mutable history so
concurrent observers cannot interleave into a torn trajectory.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from .assessment import GovernanceDecision, UREAssessment
from .attack_memory import AttackMemoryExchange, MemoryRecall
from .dwell import HostileDwell
from .lyapunov import ENERGY_MAX, LyapunovTrajectoryEngine, energy_contributions
from .recovery import RecoveryPlanner, RecoveryVector
from .regimes import RegimeClassifier, RegimeProfile, SystemRegime
from .resilience import ResilienceIndex
from .state_vector import PolicyTelemetry, StateVector, clamp
from .telemetry import SmoothingAdapter, TelemetryAdapter, TelemetryFrame, TelemetrySource
from .thresholds import AdaptiveThresholdController
from .trajectory import TrajectoryEngine, TrajectorySnapshot
from .vaccines import BehavioralVaccineEngine, Immunization, Observation

__all__ = ["GatewayHealthEngine", "UREConfig", "UREEngine"]


@runtime_checkable
class GatewayHealthEngine(Protocol):
    """The contract GSA depends on.

    Defined as a Protocol so the gateway depends on a shape, not on URE's
    concrete class -- the dependency inversion the architecture requires. Any
    object with this method can stand in, including a test double.
    """

    async def evaluate(self, state: StateVector) -> UREAssessment:
        ...


class UREConfig:
    """Tuning surface for a URE deployment.

    Grouped into one object so a deployment is configured in a single place
    rather than through a dozen constructor arguments threaded down the stack.
    """

    __slots__ = (
        "adversarial_scale",
        "amx_capacity",
        "amx_half_life",
        "enable_learning",
        "history_window",
        "latency_ceiling_ms",
        "queue_capacity",
        "restore_patience",
        "slope_window",
        "smoothing_alpha",
        "synthesis_threshold",
        "vaccine_capacity",
    )

    def __init__(
        self,
        *,
        history_window: int = 64,
        slope_window: int = 8,
        smoothing_alpha: float = 0.4,
        latency_ceiling_ms: float = 1000.0,
        adversarial_scale: float = 10.0,
        queue_capacity: int = 256,
        amx_capacity: int = 4096,
        amx_half_life: float = 6 * 60 * 60.0,
        vaccine_capacity: int = 512,
        synthesis_threshold: int = 3,
        restore_patience: int = 3,
        enable_learning: bool = True,
    ) -> None:
        self.history_window = history_window
        self.slope_window = slope_window
        self.smoothing_alpha = smoothing_alpha
        self.latency_ceiling_ms = latency_ceiling_ms
        self.adversarial_scale = adversarial_scale
        self.queue_capacity = queue_capacity
        self.amx_capacity = amx_capacity
        self.amx_half_life = amx_half_life
        self.vaccine_capacity = vaccine_capacity
        self.synthesis_threshold = synthesis_threshold
        self.restore_patience = restore_patience
        self.enable_learning = enable_learning


class UREEngine:
    """The Universal Resilience Engine.

    Construct once per deployment and call :meth:`observe` (or
    :meth:`evaluate`) on every request or telemetry tick. The engine holds all
    the history; callers hold nothing.
    """

    __slots__ = (
        "_amx",
        "_bve",
        "_classifier",
        "_config",
        "_dwell",
        "_last",
        "_lock",
        "_lyapunov",
        "_observations",
        "_planner",
        "_resilience",
        "_telemetry",
        "_thresholds",
        "_trajectory",
    )

    def __init__(
        self,
        config: UREConfig | None = None,
        *,
        telemetry: TelemetrySource | None = None,
        attack_memory: AttackMemoryExchange | None = None,
        vaccines: BehavioralVaccineEngine | None = None,
    ) -> None:
        self._config = config or UREConfig()

        self._telemetry: TelemetrySource = telemetry or SmoothingAdapter(
            TelemetryAdapter(
                latency_ceiling_ms=self._config.latency_ceiling_ms,
                adversarial_scale=self._config.adversarial_scale,
            ),
            alpha=self._config.smoothing_alpha,
        )
        self._lyapunov = LyapunovTrajectoryEngine(
            window=self._config.history_window, slope_window=self._config.slope_window
        )
        self._trajectory = TrajectoryEngine(window=self._config.history_window)
        self._classifier = RegimeClassifier(energy_max=ENERGY_MAX)
        self._dwell = HostileDwell()
        self._amx = attack_memory or AttackMemoryExchange(
            capacity=self._config.amx_capacity, half_life=self._config.amx_half_life
        )
        self._bve = vaccines or BehavioralVaccineEngine(
            capacity=self._config.vaccine_capacity,
            synthesis_threshold=self._config.synthesis_threshold,
        )
        self._resilience = ResilienceIndex()
        self._planner = RecoveryPlanner(restore_patience=self._config.restore_patience)
        self._thresholds = AdaptiveThresholdController()

        self._observations = 0
        self._last: UREAssessment | None = None
        self._lock = threading.RLock()

    # -- subsystem access -------------------------------------------------

    @property
    def attack_memory(self) -> AttackMemoryExchange:
        """AMX, for direct correlation queries and profile exchange."""
        return self._amx

    @property
    def vaccines(self) -> BehavioralVaccineEngine:
        """BVE, for feedback (:meth:`~BehavioralVaccineEngine.confirm`) and export."""
        return self._bve

    @property
    def thresholds(self) -> AdaptiveThresholdController:
        """The adaptive threshold controller, for false-discovery feedback."""
        return self._thresholds

    @property
    def last_assessment(self) -> UREAssessment | None:
        return self._last

    @property
    def observations(self) -> int:
        return self._observations

    # -- the pipeline -----------------------------------------------------

    def observe(
        self,
        telemetry: TelemetryFrame | Mapping[str, Any],
        *,
        now: float | None = None,
        scope: str = "global",
    ) -> UREAssessment:
        """Run the full pipeline over one telemetry observation."""
        timestamp = time.time() if now is None else now
        frame = (
            telemetry
            if isinstance(telemetry, TelemetryFrame)
            else TelemetryFrame.from_mapping(telemetry)
        )

        with self._lock:
            self._observations += 1

            # 1. State vector.
            state = self._telemetry.to_state_vector(frame, timestamp)

            # 2. AMX correlation, 3. BVE activation. Both run before energy so
            #    learned adversarial history is part of the state being scored,
            #    not a footnote attached to it afterwards.
            recall = self._amx.recall(frame.signatures, now=timestamp)
            immunization = self._bve.immunize(frame.markers, now=timestamp)
            ambient = self._amx.ambient_pressure(now=timestamp)
            learned = clamp(
                1.0 - (1.0 - recall.pressure) * (1.0 - immunization.pressure) * (1.0 - ambient)
            )
            if learned > 0.0:
                state = state.raised("adversarial_pressure", learned)

            # 4. Lyapunov energy and derivative.
            energy = self._lyapunov.observe(state)
            derivative = self._lyapunov.derivative

            # 5. Trajectory.
            motion = self._trajectory.update(timestamp, energy, state)

            # 5b. Hostile dwell. Updated after the learned pressure has been
            #     folded into the state, so a signature AMX recognises counts
            #     toward exposure the same way a directly observed one does.
            dwell = self._dwell.update(state, timestamp)

            # 6. Regime classification.
            profile = self._classifier.classify(
                state,
                energy,
                derivative,
                volatility=motion.volatility,
                recall_strength=max(recall.strength, immunization.best_match),
                dwell=dwell,
            )

            # 7. Resilience index.
            breakdown = self._resilience.compute(
                energy=energy,
                derivative=derivative,
                profile=profile,
                trajectory_stability=self._trajectory.stability_score(),
                adaptation_score=self._bve.adaptation_score(),
                recovery_score=self._planner.recovery_score(),
                memory_depth=len(self._amx),
                memory_recall=recall.pressure,
            )

            # 8. Recovery recommendation.
            vector = self._planner.plan(
                state,
                profile,
                energy,
                derivative,
                breakdown.index,
                campaign=recall.campaign,
                scope=scope,
            )

            # 9. Adaptive threshold for the gateway's next decision.
            threshold = self._thresholds.compute(
                regime=profile.dominant,
                resilience=breakdown.index,
                attack_pressure=max(state.threat_pressure, state.adversarial_pressure),
                recovering=profile.dominant is SystemRegime.RECOVERING,
            )

            # 10. Learning from this observation, if enabled and permitted.
            #     The recovery vector can switch learning off: an engine that
            #     keeps learning through its own collapse learns the collapse.
            learning_off = bool(vector.controls.get("disable_learning"))
            if self._config.enable_learning and not learning_off:
                self._learn(frame, profile, timestamp)

            assessment = UREAssessment(
                resilience_index=breakdown.index,
                regime=profile.dominant,
                lyapunov_energy=energy,
                lyapunov_derivative=derivative,
                trajectory_velocity=motion.velocity,
                trajectory_acceleration=motion.acceleration,
                trajectory_volatility=motion.volatility,
                active_attack_profiles=list(recall.matches),
                active_vaccines=list(immunization.activated),
                recommended_action=vector.action,
                explanation=self._explain(
                    state, profile, breakdown, motion, vector, recall, immunization, energy
                ),
                timestamp=timestamp,
                state=state,
                regime_confidence=profile.confidence,
                regime_entropy=profile.entropy,
                regime_profile=profile,
                resilience_breakdown=breakdown,
                recovery_vector=vector,
                recommended_threshold=threshold.threshold,
                decision=GovernanceDecision.from_recovery(vector.action),
                time_to_saturation=self._lyapunov.time_to_threshold(ENERGY_MAX * 0.95),
                hostile_dwell=dwell,
            )
            self._last = assessment
            return assessment

    async def evaluate(self, state: StateVector) -> UREAssessment:
        """Assess a pre-built state vector. Satisfies :class:`GatewayHealthEngine`.

        Used by callers that already project their own telemetry and want URE
        to reason over the result. The AMX/BVE stages are skipped because there
        are no signatures or markers on a bare state vector -- learned pressure
        must already be baked into ``adversarial_pressure`` by the caller.
        """
        with self._lock:
            self._observations += 1
            timestamp = state.timestamp or time.time()

            energy = self._lyapunov.observe(state)
            derivative = self._lyapunov.derivative
            motion = self._trajectory.update(timestamp, energy, state)
            dwell = self._dwell.update(state, timestamp)
            profile = self._classifier.classify(
                state, energy, derivative, volatility=motion.volatility, dwell=dwell
            )
            breakdown = self._resilience.compute(
                energy=energy,
                derivative=derivative,
                profile=profile,
                trajectory_stability=self._trajectory.stability_score(),
                adaptation_score=self._bve.adaptation_score(),
                recovery_score=self._planner.recovery_score(),
                memory_depth=len(self._amx),
                memory_recall=0.0,
            )
            vector = self._planner.plan(
                state, profile, energy, derivative, breakdown.index
            )
            threshold = self._thresholds.compute(
                regime=profile.dominant,
                resilience=breakdown.index,
                attack_pressure=max(state.threat_pressure, state.adversarial_pressure),
                recovering=profile.dominant is SystemRegime.RECOVERING,
            )

            assessment = UREAssessment(
                resilience_index=breakdown.index,
                regime=profile.dominant,
                lyapunov_energy=energy,
                lyapunov_derivative=derivative,
                trajectory_velocity=motion.velocity,
                trajectory_acceleration=motion.acceleration,
                trajectory_volatility=motion.volatility,
                active_attack_profiles=[],
                active_vaccines=[],
                recommended_action=vector.action,
                explanation=self._explain(
                    state,
                    profile,
                    breakdown,
                    motion,
                    vector,
                    MemoryRecall.empty(),
                    Immunization.none(),
                    energy,
                ),
                timestamp=timestamp,
                state=state,
                regime_confidence=profile.confidence,
                regime_entropy=profile.entropy,
                regime_profile=profile,
                resilience_breakdown=breakdown,
                recovery_vector=vector,
                recommended_threshold=threshold.threshold,
                decision=GovernanceDecision.from_recovery(vector.action),
                time_to_saturation=self._lyapunov.time_to_threshold(ENERGY_MAX * 0.95),
                hostile_dwell=dwell,
            )
            self._last = assessment
            return assessment

    # convenience alias: the engine is callable on a telemetry frame.
    __call__ = observe

    # -- policy signal ----------------------------------------------------

    def observe_policy_signal(self, telemetry: PolicyTelemetry) -> None:
        """Feed a policy outcome into AMX and BVE without exposing policy internals.

        This is the extension the archive's review specified. URE learns that
        *something* identified by an opaque signature produced an adverse
        outcome. It does not learn the rule, the domain, or the content -- the
        policy seam holds.
        """
        if not self._config.enable_learning:
            return
        with self._lock:
            if telemetry.adverse and telemetry.signature:
                self._amx.remember(
                    telemetry.signature,
                    severity=clamp(telemetry.input_score if not telemetry.hard_block else 1.0),
                    succeeded=telemetry.output_acceptable and not telemetry.hard_block,
                    now=telemetry.timestamp,
                    tags=(f"policy:{telemetry.policy_name}",),
                )

    def _learn(
        self, frame: TelemetryFrame, profile: RegimeProfile, timestamp: float
    ) -> None:
        """Record this observation into AMX and BVE. Caller holds the lock."""
        adverse = profile.dominant.is_hostile or frame.risk_score >= 0.5 or (
            frame.policy is not None and frame.policy.adverse
        )

        if frame.signatures and adverse:
            severity = max(frame.risk_score, frame.blocked_rate, 0.5)
            for signature in frame.signatures:
                self._amx.remember(signature, severity=severity, now=timestamp)

        if frame.markers:
            self._bve.observe(
                Observation(
                    markers=frame.markers,
                    adverse=adverse,
                    timestamp=timestamp,
                    signature=frame.signatures[0] if frame.signatures else "",
                )
            )

        if frame.policy is not None:
            self.observe_policy_signal(frame.policy)

    # -- narrative --------------------------------------------------------

    @staticmethod
    def _explain(
        state: StateVector,
        profile: RegimeProfile,
        breakdown: Any,
        motion: TrajectorySnapshot,
        vector: RecoveryVector,
        recall: MemoryRecall,
        immunization: Immunization,
        energy: float,
    ) -> str:
        """Compose a human-readable account of the assessment.

        Written for an operator reading an alert at 3am: what regime, how
        confident, what is driving it, what is being recommended.
        """
        contributions = energy_contributions(state)
        driver = max(contributions, key=lambda name: contributions[name])
        share = contributions[driver]
        if motion.velocity > 0.02:
            direction = "rising"
        elif motion.velocity < -0.02:
            direction = "falling"
        else:
            direction = "steady"

        parts = [
            f"{profile.dominant.value} at {profile.confidence:.0%} confidence "
            f"(entropy {profile.entropy:.2f}).",
            f"Energy {energy:.3f}/{ENERGY_MAX:.1f} {direction}"
            f" ({motion.velocity:+.4f}/s).",
        ]

        if share > 0.0:
            parts.append(
                f"Dominant pressure: {driver.replace('_pressure', '')} "
                f"at {getattr(state, driver):.2f} ({share:.0%} of energy)."
            )

        if profile.ambiguous:
            parts.append(
                f"Classification is close: {profile.runner_up.value} trails by "
                f"{profile.margin:.2f}."
            )

        if motion.unstable:
            parts.append(
                f"Trajectory is oscillating ({motion.oscillations} reversals, "
                f"volatility {motion.volatility:.3f}); the mean trend understates the instability."
            )

        if recall.matches:
            parts.append(
                f"AMX recalls {len(recall.matches)} known signature(s)"
                + (" from a recurring campaign" if recall.campaign else "")
                + f" (pressure {recall.pressure:.2f})."
            )

        if immunization.activated:
            parts.append(
                f"BVE fired {len(immunization.activated)} vaccine(s)"
                + (
                    f" on a partial pattern match ({immunization.best_match:.0%}) "
                    "before the sequence completed"
                    if immunization.preemptive
                    else ""
                )
                + "."
            )

        parts.append(
            f"Resilience {breakdown.index:.2f} ({breakdown.band}); "
            f"weakest term is {breakdown.weakest_term}."
        )
        parts.append(vector.rationale)
        return " ".join(parts)

    # -- lifecycle --------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Operational summary for a dashboard or a /health endpoint."""
        with self._lock:
            return {
                "observations": self._observations,
                "energy": round(self._lyapunov.energy, 4),
                "derivative": round(self._lyapunov.derivative, 6),
                "headroom": round(self._lyapunov.headroom(), 4),
                "trajectory_depth": self._trajectory.depth,
                "attack_profiles": len(self._amx),
                "vaccines": len(self._bve),
                "adaptation": round(self._bve.adaptation_score(), 4),
                "current_action": self._planner.current_action.value,
                "observed_fdr": round(self._thresholds.observed_fdr, 4),
                "last_regime": self._last.regime.value if self._last else None,
                "last_resilience": (
                    round(self._last.resilience_index, 4) if self._last else None
                ),
            }

    def reset(self, *, keep_memory: bool = True) -> None:
        """Clear trajectory history.

        ``keep_memory`` defaults to True: restarting a process should not
        discard what the deployment has learned about its adversaries. Pass
        False only when the memory is known to be poisoned.
        """
        with self._lock:
            self._lyapunov.reset()
            self._trajectory.reset()
            self._planner.reset()
            self._dwell.reset()
            self._thresholds.reset()
            if isinstance(self._telemetry, SmoothingAdapter):
                self._telemetry.reset()
            self._observations = 0
            self._last = None
            if not keep_memory:
                self._amx.clear()
