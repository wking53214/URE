"""
Persistence of the learned hostile-pressure baseline, and the sentinel_os preset.

The baseline is what lets URE tell "this deployment's normal" from "an attack",
and it is learned, so a restart loses it. These pin what carrying it across a
restart buys, and what a hostile or stale record is and is not able to do.
"""

from __future__ import annotations

import pytest

from ure_engine import UREConfig, UREEngine
from ure_engine.dwell import STATE_VERSION, HostileDwell

BENIGN_REGIMES = frozenset({"NOMINAL", "ADAPTING", "RECOVERING"})
QUIET = {"risk_score": 0.02, "blocked_rate": 0.02, "retry_rate": 0.01,
         "latency_ms": 15.0, "queue_saturation": 0.03}


def hostile(level: float) -> dict:
    return {"risk_score": level, "blocked_rate": level, "retry_rate": 0.02,
            "latency_ms": 20.0, "queue_saturation": 0.05}


def feed(engine: UREEngine, telemetry: dict, steps: int, start: int = 0):
    assessment = None
    for step in range(start, start + steps):
        assessment = engine.observe(telemetry, now=float(step))
    assert assessment is not None
    return assessment, start + steps


def test_a_restart_during_an_attack_no_longer_calibrates_to_it() -> None:
    """The whole point of persisting the baseline.

    An engine that comes up while the attack is already under way learns the
    attack as normal, so an adversary who can crash it gets a free calibration
    window. Seeding it from a baseline learned in quiet closes that.
    """
    veteran = UREEngine(UREConfig.for_sentinel_os())
    feed(veteran, QUIET, 100)
    saved = veteran.export_dwell()

    restarted = UREEngine(UREConfig.for_sentinel_os())
    assert restarted.import_dwell(saved) is True
    cold = UREEngine(UREConfig.for_sentinel_os())

    attack = hostile(0.30)
    seeded, _ = feed(restarted, attack, 80)
    unseeded, _ = feed(cold, attack, 80)

    assert seeded.regime.value not in BENIGN_REGIMES, (
        "a restarted engine seeded with a quiet baseline did not see a 30% attack"
    )
    assert unseeded.regime.value in BENIGN_REGIMES, (
        "precondition: an engine started mid-attack should calibrate to it; if "
        "it no longer does, this test no longer demonstrates anything"
    )


def test_export_then_import_round_trips_the_learned_state() -> None:
    engine = UREEngine(UREConfig.for_sentinel_os())
    feed(engine, hostile(0.20), 150)
    saved = engine.export_dwell()

    other = UREEngine(UREConfig.for_sentinel_os())
    assert other.import_dwell(saved) is True
    assert other.export_dwell() == saved
    assert saved["version"] == STATE_VERSION
    assert 0.0 < float(saved["baseline"]) <= 0.26  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "baseline",
        [],
        {},
        {"version": 99, "baseline": 0.1, "value": 0.0, "observations": 5, "last_timestamp": None},
        {"version": STATE_VERSION, "baseline": float("nan"), "value": 0.0,
         "observations": 5, "last_timestamp": None},
        {"version": STATE_VERSION, "baseline": float("inf"), "value": 0.0,
         "observations": 5, "last_timestamp": None},
        {"version": STATE_VERSION, "baseline": True, "value": 0.0,
         "observations": 5, "last_timestamp": None},
        {"version": STATE_VERSION, "baseline": 0.1, "value": 0.0,
         "observations": -1, "last_timestamp": None},
        {"version": STATE_VERSION, "baseline": 0.1, "value": 0.0,
         "observations": 5.5, "last_timestamp": None},
        {"version": STATE_VERSION, "baseline": 0.1, "value": 0.0,
         "observations": 5, "last_timestamp": float("nan")},
        {"version": STATE_VERSION, "baseline": "0.1", "value": 0.0,
         "observations": 5, "last_timestamp": None},
    ],
)
def test_a_malformed_record_is_refused_whole_and_changes_nothing(bad: object) -> None:
    engine = UREEngine(UREConfig.for_sentinel_os())
    feed(engine, hostile(0.15), 60)
    before = engine.export_dwell()

    assert engine.import_dwell(bad) is False
    assert engine.export_dwell() == before


def test_an_imported_baseline_cannot_exceed_the_engine_ceiling() -> None:
    """The record is untrusted: it is the one input that can blind the engine.

    A tampered baseline is clamped to this engine's own ceiling, so it can do
    no more than a cold start during an attack already could, which is hide
    traffic below ceiling + margin and never above it.
    """
    engine = UREEngine(UREConfig.for_sentinel_os())
    forged = {"version": STATE_VERSION, "baseline": 0.99, "value": 0.0,
              "observations": 10_000, "last_timestamp": None}
    assert engine.import_dwell(forged) is True
    assert engine.export_dwell()["baseline"] == 0.26

    # And the clamp is honest end to end: 0.45 sits above ceiling + margin and
    # is still caught immediately, however the baseline got there.
    assessment, _ = feed(engine, hostile(0.45), 10)
    assert assessment.regime.value not in BENIGN_REGIMES


def test_a_configured_ceiling_bounds_the_import() -> None:
    engine = UREEngine(UREConfig.for_sentinel_os(dwell_baseline_ceiling=0.10))
    forged = {"version": STATE_VERSION, "baseline": 0.25, "value": 0.0,
              "observations": 100, "last_timestamp": None}
    assert engine.import_dwell(forged) is True
    assert engine.export_dwell()["baseline"] == 0.10


def test_downtime_decays_the_accumulator_rather_than_freezing_it() -> None:
    """A restart after a long outage must not resume mid-incident forever.

    The first update after an import takes its time step from the restored
    timestamp, clamped, so the accumulator decays across the gap by the same
    rule that decays it while running.
    """
    engine = UREEngine(UREConfig.for_sentinel_os())
    feed(engine, QUIET, 100)
    charged, t = feed(engine, hostile(0.60), 60, start=100)
    assert charged.hostile_dwell > 0.9
    saved = engine.export_dwell()

    restarted = UREEngine(UREConfig.for_sentinel_os())
    assert restarted.import_dwell(saved) is True
    resumed = restarted.observe(QUIET, now=float(t) + 86_400.0)

    assert resumed.hostile_dwell < 0.02, (
        f"a day of downtime left dwell at {resumed.hostile_dwell:.3f}"
    )


def test_the_baseline_is_visible_on_the_assessment() -> None:
    """An operator looking at an ATTACKED call needs to see what it was judged
    against. The baseline used to be reachable from nowhere."""
    engine = UREEngine(UREConfig.for_sentinel_os())
    assessment, _ = feed(engine, hostile(0.20), 200)

    assert assessment.hostile_baseline == pytest.approx(engine.export_dwell()["baseline"])
    assert 0.15 < assessment.hostile_baseline <= 0.26
    assert assessment.to_dict()["hostile_baseline"] == round(assessment.hostile_baseline, 4)


def test_state_round_trips_on_the_bare_accumulator_too() -> None:
    dwell = HostileDwell()
    assert dwell.import_state(dwell.export_state()) is True
    assert dwell.baseline == 0.0 and dwell.value == 0.0 and not dwell.calibrated


# -- the preset ---------------------------------------------------------------


def test_the_preset_carries_the_tuned_values_and_leaves_defaults_alone() -> None:
    tuned = UREConfig.for_sentinel_os()
    default = UREConfig()

    assert (tuned.smoothing_alpha, tuned.slope_window, tuned.history_window,
            tuned.restore_patience) == (0.6, 12, 64, 5)
    # The point of a preset: global defaults were not silently replaced by
    # numbers fitted to a simulation.
    assert (default.smoothing_alpha, default.slope_window, default.restore_patience) == (
        0.4, 8, 3,
    )


def test_a_preset_override_wins_and_an_unknown_name_is_refused() -> None:
    assert UREConfig.for_sentinel_os(slope_window=20).slope_window == 20
    assert UREConfig.for_sentinel_os(slope_window=20).smoothing_alpha == 0.6
    with pytest.raises(TypeError, match="slope_windw"):
        UREConfig.for_sentinel_os(slope_windw=20)
