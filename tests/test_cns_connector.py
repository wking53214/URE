"""URE's recommendation restated as a CNS gate.

These skip where ``cns`` is not installed. CI installs the extra and sets URE_REQUIRE_CNS=1,
so there they cannot skip.
"""

from __future__ import annotations

import dataclasses
import math
import os

import pytest

from ure_engine import UREEngine
from ure_engine.assessment import GovernanceDecision, UREAssessment
from ure_engine.recovery import RecoveryAction
from ure_engine.regimes import SystemRegime

if os.environ.get("URE_REQUIRE_CNS") == "1":  # CI: a missing cns is a failure, not a skip
    import cns.gate as gate
else:
    gate = pytest.importorskip("cns.gate")

from ure_engine.cns_connector import (
    SUBJECT,
    UREGate,
    assessment_digest,
    to_cns_result,
)

PASS = gate.GateOutcome.PASS
RETRY = gate.GateOutcome.RETRY
BREACH = gate.GateOutcome.TERMINAL_BREACH

EXPECTED = {
    GovernanceDecision.ALLOW: PASS,
    GovernanceDecision.REVIEW: RETRY,
    GovernanceDecision.THROTTLE: RETRY,
    GovernanceDecision.DEGRADE: RETRY,
    GovernanceDecision.ISOLATE: BREACH,
    GovernanceDecision.QUARANTINE: BREACH,
    GovernanceDecision.DENY: BREACH,
}


def assessment(decision: object = GovernanceDecision.ALLOW, **over: object) -> UREAssessment:
    base = dict(
        resilience_index=0.9, regime=SystemRegime.NOMINAL, lyapunov_energy=0.1,
        lyapunov_derivative=0.0, trajectory_velocity=0.0, trajectory_acceleration=0.0,
        trajectory_volatility=0.0, active_attack_profiles=[], active_vaccines=[],
        recommended_action=RecoveryAction.NONE, explanation="all quiet", timestamp=1000.0,
        decision=decision,
    )
    base.update(over)
    return UREAssessment(**base)  # type: ignore[arg-type]


def test_the_table_decides_every_governance_decision_there_is():
    assert set(EXPECTED) == set(GovernanceDecision)


@pytest.mark.parametrize("decision, outcome", list(EXPECTED.items()))
def test_each_decision_maps_to_its_outcome_and_is_bound(decision, outcome):
    a = assessment(decision)
    result = UREGate().check(a)
    assert result.outcome is outcome and result.binds(SUBJECT, assessment_digest(a))
    assert decision.value in result.reason and "all quiet" in result.reason


def test_an_unknown_decision_is_a_breach_and_unbound():
    result = UREGate().check(assessment("WAVE_THROUGH"))
    assert result.outcome is BREACH and not result.bound()


def test_the_gate_satisfies_the_protocol_and_defaults_to_the_omega_end():
    g = UREGate()
    assert isinstance(g, gate.Gate) and g.name == "ure.assessment"
    assert g.position is gate.GatePosition.OMEGA
    assert UREGate("alpha").position is gate.GatePosition.ALPHA
    assert UREGate("alpha").check(assessment()).position is gate.GatePosition.ALPHA


@pytest.mark.parametrize("bad", ["", "beta", "OMEGA", None])
def test_a_position_that_is_not_alpha_or_omega_is_refused_at_construction(bad):
    with pytest.raises(ValueError, match="alpha' or 'omega"):
        UREGate(bad)  # type: ignore[arg-type]


def test_a_verdict_binds_to_that_observation_and_not_to_a_later_one():
    a, later = assessment(), assessment(timestamp=1001.0)
    assert assessment_digest(a) != assessment_digest(later)
    assert not UREGate().check(a).binds(SUBJECT, assessment_digest(later))


def test_an_assessment_that_cannot_be_described_for_binding_is_an_unbound_breach():
    result = UREGate().check(assessment(lyapunov_energy=math.nan))
    assert result.outcome is BREACH and not result.bound() and "nothing was judged" in result.reason


@pytest.mark.parametrize("bad", [None, {"decision": "ALLOW"}, "ALLOW", GovernanceDecision.ALLOW])
def test_something_that_is_not_an_assessment_is_a_type_error_not_a_pass(bad):
    with pytest.raises(TypeError, match="UREAssessment"):
        UREGate().check(bad)
    with pytest.raises(TypeError):
        to_cns_result(bad)


# ------------------------------------------------------------ against the real engine

def test_a_healthy_system_passes_and_a_stressed_one_does_not():
    engine = UREEngine()
    calm = engine.observe({"latency_ms": 50, "error_rate": 0.0}, now=1000.0)
    assert UREGate().check(calm).outcome is PASS
    engine = UREEngine()
    for i in range(12):
        hot = engine.observe({"latency_ms": 1000, "error_rate": 1.0}, now=2000.0 + i)
    verdict = UREGate().check(hot)
    assert verdict.outcome is EXPECTED[hot.decision] and verdict.outcome is not PASS
    assert verdict.binds(SUBJECT, assessment_digest(hot))


def test_the_gate_never_changes_what_the_engine_concluded():
    engine = UREEngine()
    a = engine.observe({"latency_ms": 900, "error_rate": 0.6}, now=1000.0)
    before = a.to_dict()
    UREGate().check(a)
    assert a.to_dict() == before and engine.last_assessment is a
    assert dataclasses.is_dataclass(a)
