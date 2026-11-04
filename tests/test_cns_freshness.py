"""F06: the CNS gate refuses stale and future-dated assessments.

An assessment describes the state of a system at one moment. Read long after,
it says nothing about the state now; read from the future, its clock is not
to be trusted. Either is a breach, never a pass.
"""

from __future__ import annotations

import math
import os
import time

import pytest

from ure_engine.assessment import UREAssessment
from ure_engine.recovery import RecoveryAction
from ure_engine.regimes import SystemRegime

if os.environ.get("URE_REQUIRE_CNS") == "1":  # CI: a missing cns is a failure, not a skip
    import cns.gate as gate
else:
    gate = pytest.importorskip("cns.gate")

from ure_engine.cns_connector import (
    DEFAULT_MAX_AGE_SECONDS,
    SUBJECT,
    UREGate,
    assessment_digest,
    to_cns_result,
)

PASS = gate.GateOutcome.PASS
BREACH = gate.GateOutcome.TERMINAL_BREACH


def at(timestamp: object) -> UREAssessment:
    return UREAssessment(
        resilience_index=0.9, regime=SystemRegime.NOMINAL, lyapunov_energy=0.1,
        lyapunov_derivative=0.0, trajectory_velocity=0.0, trajectory_acceleration=0.0,
        trajectory_volatility=0.0, active_attack_profiles=[], active_vaccines=[],
        recommended_action=RecoveryAction.NONE, explanation="all quiet",
        timestamp=timestamp,  # type: ignore[arg-type]
    )


def test_the_default_max_age_is_a_safe_finite_bound():
    assert 0 < DEFAULT_MAX_AGE_SECONDS <= 300


def test_a_1970_assessment_is_a_bound_breach():
    a = at(1000.0)
    result = UREGate().check(a)
    assert result.outcome is BREACH and "stale" in result.reason
    assert result.binds(SUBJECT, assessment_digest(a))
    assert to_cns_result(a).outcome is BREACH


def test_a_fresh_assessment_passes():
    assert UREGate().check(at(time.time())).outcome is PASS


def test_an_assessment_older_than_the_gates_max_age_is_a_breach():
    a = at(time.time() - 120)
    assert UREGate(max_age=60).check(a).outcome is BREACH
    assert UREGate(max_age=600).check(a).outcome is PASS
    assert to_cns_result(a, max_age=60).outcome is BREACH


def test_a_future_dated_assessment_is_a_breach():
    result = UREGate().check(at(time.time() + 3600))
    assert result.outcome is BREACH and "future" in result.reason


@pytest.mark.parametrize("ts", [math.nan, math.inf, -math.inf, "now", None, True])
def test_a_timestamp_that_is_not_a_finite_number_is_a_breach(ts):
    assert UREGate().check(at(ts)).outcome is BREACH


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf, None, True, "60"])
def test_a_max_age_that_is_not_a_positive_finite_number_is_refused_at_construction(bad):
    with pytest.raises(ValueError, match="max_age"):
        UREGate(max_age=bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_age"):
        to_cns_result(at(time.time()), max_age=bad)  # type: ignore[arg-type]
