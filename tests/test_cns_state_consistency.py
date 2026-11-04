"""F05: the CNS gate judges the assessment's real state, not a defaulted decision.

``UREAssessment.decision`` defaults to ``ALLOW``. A hand-built assessment that
says ATTACKED and recommends QUARANTINE, but leaves ``decision`` at its
default, must not read as a recommendation to proceed. When the stated
decision is less restrictive than the state implies, state and decision
disagree, and that is a breach.
"""

from __future__ import annotations

import os
import time

import pytest

from ure_engine import UREEngine
from ure_engine.assessment import GovernanceDecision, UREAssessment
from ure_engine.recovery import RecoveryAction
from ure_engine.regimes import SystemRegime

if os.environ.get("URE_REQUIRE_CNS") == "1":  # CI: a missing cns is a failure, not a skip
    import cns.gate as gate
else:
    gate = pytest.importorskip("cns.gate")

from ure_engine.cns_connector import SUBJECT, UREGate, assessment_digest

PASS = gate.GateOutcome.PASS
RETRY = gate.GateOutcome.RETRY
BREACH = gate.GateOutcome.TERMINAL_BREACH


def hand_built(**over: object) -> UREAssessment:
    base = dict(
        resilience_index=0.9, regime=SystemRegime.NOMINAL, lyapunov_energy=0.1,
        lyapunov_derivative=0.0, trajectory_velocity=0.0, trajectory_acceleration=0.0,
        trajectory_volatility=0.0, active_attack_profiles=[], active_vaccines=[],
        recommended_action=RecoveryAction.NONE, explanation="hand built",
        timestamp=time.time(),
    )
    base.update(over)
    return UREAssessment(**base)  # type: ignore[arg-type]


def test_attacked_and_quarantine_with_a_defaulted_decision_is_a_breach():
    a = hand_built(regime=SystemRegime.ATTACKED, recommended_action=RecoveryAction.QUARANTINE)
    assert a.decision is GovernanceDecision.ALLOW  # the default the audit found
    result = UREGate().check(a)
    assert result.outcome is BREACH
    assert "disagree" in result.reason
    assert result.binds(SUBJECT, assessment_digest(a))


@pytest.mark.parametrize("action", list(RecoveryAction))
def test_a_decision_less_restrictive_than_the_recommended_action_is_a_breach(action):
    implied = GovernanceDecision.from_recovery(action)
    for decision in GovernanceDecision:
        if decision.restriction >= implied.restriction:
            continue
        result = UREGate().check(hand_built(recommended_action=action, decision=decision))
        assert result.outcome is BREACH, (action, decision)


@pytest.mark.parametrize(
    "regime, floor",
    [
        (SystemRegime.CASCADING, GovernanceDecision.QUARANTINE),
        (SystemRegime.ATTACKED, GovernanceDecision.ISOLATE),
        (SystemRegime.STRESSED, GovernanceDecision.DEGRADE),
        (SystemRegime.UNKNOWN, GovernanceDecision.DEGRADE),
    ],
)
def test_a_hostile_or_unknown_regime_with_a_weaker_decision_is_a_breach(regime, floor):
    weaker = [d for d in GovernanceDecision if d.restriction < floor.restriction]
    for decision in weaker:
        result = UREGate().check(
            hand_built(regime=regime, recommended_action=RecoveryAction.NONE, decision=decision)
        )
        assert result.outcome is BREACH and "disagree" in result.reason, (regime, decision)


def test_a_decision_more_restrictive_than_the_state_still_maps_by_the_table():
    result = UREGate().check(hand_built(decision=GovernanceDecision.REVIEW))
    assert result.outcome is RETRY


def test_a_consistent_healthy_assessment_still_passes():
    assert UREGate().check(hand_built()).outcome is PASS


@pytest.mark.parametrize(
    "field, value",
    [("regime", "NOMINAL"), ("recommended_action", "NONE"), ("regime", None)],
)
def test_a_state_field_of_the_wrong_type_is_an_unbound_breach(field, value):
    result = UREGate().check(hand_built(**{field: value}))
    assert result.outcome is BREACH and not result.bound()


def test_real_engine_assessments_are_never_called_inconsistent():
    engine = UREEngine()
    start = time.time() - 30
    loads = [(50, 0.0)] * 3 + [(1000, 1.0)] * 12 + [(50, 0.0)] * 12
    for i, (latency, error) in enumerate(loads):
        a = engine.observe({"latency_ms": latency, "error_rate": error}, now=start + i)
        result = UREGate().check(a)
        assert "disagree" not in result.reason, a.to_dict()
