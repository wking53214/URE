#!/usr/bin/env python3
"""
A synthetic incident, narrated by URE.

Run it::

    python examples/incident_walkthrough.py

The scenario runs a gateway through five phases and prints what URE concludes
at each step. It is the fastest way to see why the regime model earns its
complexity: phases 2 and 4 have comparable energy and would be indistinguishable
to a scalar risk score, but they call for opposite responses.

    1. Quiet          nominal traffic
    2. Load           a legitimate traffic surge      -> STRESSED, throttle
    3. Recovery       the surge passes                -> RECOVERING, restore
    4. Attack         a probing campaign begins       -> ATTACKED, isolate
    5. Cascade        the attack induces failures     -> CASCADING, quarantine

Watch the `action` column at phases 2 and 4. Same pressure, opposite answer:
throttling an overloaded system relieves it, and throttling an attacked system
delivers the denial of service the attacker wanted.
"""

from __future__ import annotations

from ure_engine import RecoveryAction, UREConfig, UREEngine

ATTACK_MARKERS = ("ignore_previous", "developer_override", "show_prompt")

HEADER = (
    f"{'t':>4}  {'phase':<10} {'regime':<11} {'E':>6} {'dV/dt':>8} "
    f"{'RI':>5} {'thr':>5}  {'action':<11} notes"
)


def phases() -> list[tuple[str, dict[str, object]]]:
    """Build the timeline as (phase, telemetry) pairs."""
    timeline: list[tuple[str, dict[str, object]]] = []

    # 1. Quiet.
    for _ in range(6):
        timeline.append(
            ("quiet", {"risk_score": 0.02, "retry_rate": 0.01, "latency_ms": 15.0})
        )

    # 2. A legitimate surge: latency, retries and queue climb; risk does not.
    for step in range(8):
        ramp = (step + 1) / 8
        timeline.append(
            (
                "load",
                {
                    "risk_score": 0.05,
                    "retry_rate": 0.6 * ramp,
                    "latency_ms": 900.0 * ramp,
                    "queue_saturation": 0.95 * ramp,
                    "substrate_health": 1.0 - 0.5 * ramp,
                },
            )
        )

    # 3. The surge passes.
    for step in range(8):
        decay = 1.0 - (step + 1) / 8
        timeline.append(
            (
                "recovery",
                {
                    "risk_score": 0.05,
                    "retry_rate": 0.6 * decay,
                    "latency_ms": 900.0 * decay,
                    "queue_saturation": 0.95 * decay,
                    "substrate_health": 1.0 - 0.5 * decay,
                },
            )
        )

    # 4. A probing campaign. Same signature and markers every time, so AMX and
    #    BVE both have something to latch onto.
    for step in range(8):
        ramp = (step + 1) / 8
        timeline.append(
            (
                "attack",
                {
                    "risk_score": 0.50 + 0.20 * ramp,
                    "blocked_rate": 0.45 + 0.15 * ramp,
                    "retry_rate": 0.05,
                    "latency_ms": 40.0,
                    "queue_saturation": 0.1,
                    "signatures": ["probe-7f3a"],
                    "markers": ATTACK_MARKERS,
                },
            )
        )

    # 5. The attack starts inducing failures.
    for step in range(8):
        ramp = (step + 1) / 8
        timeline.append(
            (
                "cascade",
                {
                    "risk_score": 0.95,
                    "blocked_rate": 0.9,
                    "retry_rate": 0.5 + 0.45 * ramp,
                    "latency_ms": 200.0 + 1800.0 * ramp,
                    "queue_saturation": 0.4 + 0.6 * ramp,
                    "substrate_health": 0.6 - 0.6 * ramp,
                    "circuit_state": "OPEN" if ramp > 0.4 else "CLOSED",
                    "adversarial_events": 10.0 * ramp,
                    "signatures": ["probe-7f3a"],
                    "markers": ATTACK_MARKERS,
                },
            )
        )

    return timeline


def main() -> None:
    engine = UREEngine(UREConfig(synthesis_threshold=3))

    print(__doc__.split("Watch the")[0].rstrip())
    print(HEADER)
    print("-" * len(HEADER))

    previous_phase = ""
    for step, (phase, telemetry) in enumerate(phases()):
        assessment = engine.observe(telemetry, now=float(step))

        notes: list[str] = []
        if assessment.active_attack_profiles:
            notes.append(f"AMX:{len(assessment.active_attack_profiles)}")
        if assessment.active_vaccines:
            notes.append(f"BVE:{','.join(assessment.active_vaccines)}")
        if assessment.regime_profile and assessment.regime_profile.ambiguous:
            notes.append("ambiguous")
        if assessment.time_to_saturation is not None:
            notes.append(f"saturate~{assessment.time_to_saturation:.0f}s")

        marker = " *" if phase != previous_phase else "  "
        previous_phase = phase

        print(
            f"{step:>4}{marker}{phase:<10} {assessment.regime.value:<11} "
            f"{assessment.lyapunov_energy:>6.3f} {assessment.lyapunov_derivative:>+8.4f} "
            f"{assessment.resilience_index:>5.2f} {assessment.recommended_threshold:>5.2f}  "
            f"{assessment.recommended_action.value:<11} {' '.join(notes)}"
        )

    final = engine.last_assessment
    assert final is not None

    print()
    print("Final assessment")
    print("-" * 60)
    print(final.explanation)

    print()
    print("Engine state")
    print("-" * 60)
    for key, value in engine.snapshot().items():
        print(f"  {key:<20} {value}")

    if final.recommended_action is RecoveryAction.QUARANTINE:
        print()
        print("Recovery controls handed to FORTRESS:")
        assert final.recovery_vector is not None
        for control, value in sorted(final.recovery_vector.controls.items()):
            print(f"  {control:<24} {value}")


if __name__ == "__main__":
    main()
