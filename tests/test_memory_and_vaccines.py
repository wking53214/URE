"""Tests for AMX (attack memory) and BVE (behavioural vaccines).

These are the two subsystems the hardened predecessor deleted as dead code.
They are only worth having if they are actually wired and actually learn, so
these tests assert behaviour -- decay, discrimination, waning -- rather than
merely that the classes instantiate.
"""

from __future__ import annotations

import pytest

from ure_engine.attack_memory import (
    DEFAULT_HALF_LIFE,
    FORGET_THRESHOLD,
    AttackMemoryExchange,
)
from ure_engine.vaccines import BehavioralVaccineEngine, Observation


class TestAttackProfile:
    def test_remember_creates_a_profile(self) -> None:
        amx = AttackMemoryExchange()
        profile = amx.remember("sig-a", severity=0.8, now=1000.0)
        assert profile.signature == "sig-a"
        assert profile.severity == 0.8
        assert profile.frequency == 1
        assert len(amx) == 1

    def test_repeat_sightings_accumulate_frequency(self) -> None:
        amx = AttackMemoryExchange()
        for index in range(3):
            profile = amx.remember("sig-a", severity=0.5, now=1000.0 + index)
        assert profile.frequency == 3
        assert len(amx) == 1

    def test_severity_takes_the_maximum_not_the_latest(self) -> None:
        # An attack does not become less dangerous because its most recent
        # instance happened to score lower.
        amx = AttackMemoryExchange()
        amx.remember("sig-a", severity=0.9, now=1000.0)
        profile = amx.remember("sig-a", severity=0.2, now=1001.0)
        assert profile.severity == pytest.approx(0.9, abs=1e-3)

    def test_success_rate_is_a_running_average(self) -> None:
        amx = AttackMemoryExchange()
        amx.remember("sig-a", severity=0.5, succeeded=True, now=1000.0)
        amx.remember("sig-a", severity=0.5, succeeded=False, now=1001.0)
        profile = amx.lookup("sig-a", now=1001.0)
        assert profile is not None
        assert profile.success_rate == pytest.approx(0.5)

    def test_campaign_requires_recurrence_over_time(self) -> None:
        amx = AttackMemoryExchange()
        for index in range(3):
            amx.remember("burst", severity=0.7, now=1000.0 + index)
        burst = amx.lookup("burst", now=1003.0)
        assert burst is not None and not burst.is_campaign

        for index in range(3):
            amx.remember("campaign", severity=0.7, now=1000.0 + index * 60)
        campaign = amx.lookup("campaign", now=1120.0)
        assert campaign is not None and campaign.is_campaign


class TestDecay:
    def test_severity_halves_over_one_half_life(self) -> None:
        amx = AttackMemoryExchange(half_life=100.0)
        profile = amx.remember("sig-a", severity=0.8, now=0.0)
        assert profile.effective_severity(100.0, 100.0) == pytest.approx(0.4)
        assert profile.effective_severity(200.0, 100.0) == pytest.approx(0.2)

    def test_lookup_hides_fully_decayed_profiles(self) -> None:
        amx = AttackMemoryExchange(half_life=10.0)
        amx.remember("sig-a", severity=0.5, now=0.0)
        assert amx.lookup("sig-a", now=0.0) is not None
        assert amx.lookup("sig-a", now=10_000.0) is None

    def test_decay_evicts_exhausted_profiles(self) -> None:
        amx = AttackMemoryExchange(half_life=10.0)
        amx.remember("old", severity=0.5, now=0.0)
        amx.remember("new", severity=0.9, now=10_000.0)
        dropped = amx.decay(now=10_000.0)
        assert dropped == 1
        assert "new" in amx and "old" not in amx

    def test_a_pause_does_not_erase_memory(self) -> None:
        # Pausing to evade a memory window is the obvious counter-move; the
        # half-life makes it expensive without making memory permanent.
        amx = AttackMemoryExchange(half_life=DEFAULT_HALF_LIFE)
        amx.remember("sig-a", severity=1.0, now=0.0)
        one_hour_later = amx.lookup("sig-a", now=3600.0)
        assert one_hour_later is not None
        assert one_hour_later.effective_severity(3600.0) > FORGET_THRESHOLD

    def test_forget_is_immediate_and_reports_presence(self) -> None:
        amx = AttackMemoryExchange()
        amx.remember("sig-a", severity=0.5, now=0.0)
        assert amx.forget("sig-a") is True
        assert amx.forget("sig-a") is False


class TestRecall:
    def test_empty_recall_for_unknown_signatures(self) -> None:
        amx = AttackMemoryExchange()
        recall = amx.recall(["unknown"], now=0.0)
        assert recall.pressure == 0.0
        assert recall.matches == ()

    def test_recall_reports_matching_signatures(self) -> None:
        amx = AttackMemoryExchange()
        amx.remember("sig-a", severity=0.8, now=0.0)
        recall = amx.recall(["sig-a", "sig-unknown"], now=0.0)
        assert recall.matches == ("sig-a",)
        assert recall.pressure == pytest.approx(0.8)

    def test_multiple_matches_combine_probabilistically(self) -> None:
        # 1 - prod(1 - s), not a sum: two half-severity matches read as worse
        # than either alone but short of certainty, and the measure keeps
        # discriminating past the second match.
        amx = AttackMemoryExchange()
        amx.remember("a", severity=0.5, now=0.0)
        amx.remember("b", severity=0.5, now=0.0)
        recall = amx.recall(["a", "b"], now=0.0)
        assert recall.pressure == pytest.approx(0.75)
        assert recall.pressure < 1.0

    def test_ambient_pressure_reflects_deployment_history(self) -> None:
        amx = AttackMemoryExchange()
        assert amx.ambient_pressure(now=0.0) == 0.0
        for index in range(5):
            amx.remember(f"sig-{index}", severity=0.9, now=0.0)
        assert amx.ambient_pressure(now=0.0) > 0.0

    def test_ambient_pressure_is_damped_and_capped(self) -> None:
        amx = AttackMemoryExchange()
        for index in range(500):
            amx.remember(f"sig-{index}", severity=0.9, now=0.0)
        # Capped to the strongest few and halved, so a large memory of weak
        # signatures cannot manufacture pressure by sheer volume.
        assert amx.ambient_pressure(now=0.0) <= 0.5


class TestCapacityAndExchange:
    def test_capacity_is_enforced_by_evicting_the_weakest(self) -> None:
        amx = AttackMemoryExchange(capacity=3)
        amx.remember("weak", severity=0.1, now=0.0)
        for index in range(5):
            amx.remember(f"strong-{index}", severity=0.9, now=0.0)
        assert len(amx) <= 3
        assert "weak" not in amx

    def test_profiles_round_trip_through_export_and_import(self) -> None:
        source = AttackMemoryExchange()
        source.remember("sig-a", severity=0.7, now=100.0)
        source.remember("sig-b", severity=0.4, now=100.0)

        peer = AttackMemoryExchange()
        accepted = peer.import_profiles(source.export_profiles())
        assert accepted == 2
        assert peer.lookup("sig-a", now=100.0) is not None

    def test_import_never_weakens_local_memory(self) -> None:
        local = AttackMemoryExchange()
        local.remember("sig-a", severity=0.9, now=100.0)
        local.import_profiles([{"signature": "sig-a", "severity": 0.1, "frequency": 1}])
        profile = local.lookup("sig-a", now=100.0)
        assert profile is not None
        assert profile.severity == pytest.approx(0.9)

    def test_import_skips_records_without_a_signature(self) -> None:
        amx = AttackMemoryExchange()
        assert amx.import_profiles([{"severity": 0.5}]) == 0


# ---------------------------------------------------------------------------
# BVE
# ---------------------------------------------------------------------------

ATTACK = ("ignore_previous", "developer_override", "show_prompt")


def feed(
    engine: BehavioralVaccineEngine,
    markers: tuple[str, ...],
    adverse: bool,
    n: int,
    t0: float = 0.0,
) -> None:
    for index in range(n):
        engine.observe(
            Observation(markers=markers, adverse=adverse, timestamp=t0 + index)
        )


class TestSynthesis:
    def test_no_vaccine_from_a_single_incident(self) -> None:
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        created = bve.observe(Observation(markers=ATTACK, adverse=True, timestamp=0.0))
        assert created == []
        assert len(bve) == 0

    def test_vaccine_synthesized_after_the_threshold(self) -> None:
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=True, n=3)
        assert len(bve) == 1
        vaccine = bve.vaccines[0]
        assert vaccine.vaccine_id.startswith("Vaccine-")
        assert vaccine.confidence >= 0.70

    def test_benign_episodes_never_synthesize(self) -> None:
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=False, n=10)
        assert len(bve) == 0

    def test_patterns_common_to_benign_traffic_are_rejected(self) -> None:
        # Discrimination, not frequency. A pattern that appears just as often
        # in healthy traffic predicts nothing and would be pure false positives.
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        common = ("login", "fetch_profile")
        feed(bve, common, adverse=False, n=20)
        feed(bve, common, adverse=True, n=3)
        assert len(bve) == 0

    def test_only_one_vaccine_per_incident(self) -> None:
        # Synthesizing every sub-sequence would flood the store with
        # near-duplicates that all fire together and triple-count one incident.
        bve = BehavioralVaccineEngine(synthesis_threshold=2)
        created = []
        for index in range(2):
            created += bve.observe(
                Observation(markers=ATTACK, adverse=True, timestamp=float(index))
            )
        assert len(created) <= 1

    def test_short_sequences_are_not_vaccinated(self) -> None:
        bve = BehavioralVaccineEngine(synthesis_threshold=2)
        feed(bve, ("solo",), adverse=True, n=5)
        assert len(bve) == 0

    def test_rejects_a_degenerate_threshold(self) -> None:
        with pytest.raises(ValueError):
            BehavioralVaccineEngine(synthesis_threshold=1)


class TestImmunization:
    def test_no_pressure_without_vaccines(self) -> None:
        bve = BehavioralVaccineEngine()
        assert bve.immunize(ATTACK).pressure == 0.0

    def test_full_match_fires(self) -> None:
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=True, n=3)
        result = bve.immunize(ATTACK, now=100.0)
        assert result.pressure > 0.0
        assert result.activated

    def test_partial_match_fires_preemptively(self) -> None:
        # This is the entire point of the subsystem: acting at marker two of
        # three rather than after marker three.
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=True, n=3)
        result = bve.immunize(ATTACK[:2], now=100.0)
        assert result.pressure > 0.0
        assert result.preemptive
        assert result.best_match == pytest.approx(2 / 3)

    def test_partial_match_is_weaker_than_a_full_one(self) -> None:
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=True, n=3)
        partial = bve.immunize(ATTACK[:2], now=100.0)
        full = bve.immunize(ATTACK, now=101.0)
        assert full.pressure > partial.pressure
        # Pinned, not merely ordered: potency is 1.0 here, so a two-of-three
        # match contributes exactly 2/3 and a full match saturates at 1.0.
        assert partial.pressure == pytest.approx(2 / 3)
        assert partial.best_match == pytest.approx(2 / 3)
        assert full.pressure == pytest.approx(1.0)

    def test_interleaved_markers_still_match(self) -> None:
        # Subsequence matching: padding a known sequence with benign markers
        # is still executing the known sequence.
        #
        # Every field is pinned, not just `pressure > 0`. A vaccine synthesized
        # from three adverse observations and no benign ones has confidence and
        # effectiveness of exactly 1.0, so potency is 1.0 and a full match
        # saturates pressure at 1.0. "Greater than zero" is therefore true for
        # almost any perturbation of the constants in immunize(), which is what
        # mutation analysis proved: it is an assertion that cannot fail.
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=True, n=3)
        interleaved = ("hello", "ignore_previous", "weather", "developer_override",
                       "thanks", "show_prompt")

        result = bve.immunize(interleaved, now=100.0)

        assert result.activated == (bve.vaccines[0].vaccine_id,)
        # All three pattern markers found in order despite the padding.
        assert result.best_match == pytest.approx(1.0)
        # A complete match is not an early warning; preemptive is for partials.
        assert result.preemptive is False
        assert result.pressure == pytest.approx(1.0)

    def test_out_of_order_markers_do_not_fully_match(self) -> None:
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=True, n=3)
        assert bve.immunize(tuple(reversed(ATTACK)), now=100.0).best_match < 1.0

    def test_unrelated_markers_do_not_fire(self) -> None:
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=True, n=3)
        assert bve.immunize(("weather", "recipe"), now=100.0).pressure == 0.0

    def test_activation_count_increments(self) -> None:
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=True, n=3)
        vaccine_id = bve.vaccines[0].vaccine_id
        bve.immunize(ATTACK, now=100.0)
        bve.immunize(ATTACK, now=101.0)
        vaccine = bve.get(vaccine_id)
        assert vaccine is not None and vaccine.activation_count == 2


class TestWaning:
    def test_repeated_false_positives_erode_effectiveness(self) -> None:
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=True, n=3)
        vaccine_id = bve.vaccines[0].vaccine_id
        before = bve.get(vaccine_id)
        assert before is not None

        for _ in range(20):
            bve.confirm([vaccine_id], correct=False)

        after = bve.get(vaccine_id)
        assert after is not None
        assert after.effectiveness < before.effectiveness
        assert after.retired

    def test_a_single_mistake_does_not_kill_a_vaccine(self) -> None:
        # Laplace smoothing: one early error must not zero out a vaccine, and
        # one early success must not certify one.
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=True, n=3)
        vaccine_id = bve.vaccines[0].vaccine_id
        bve.confirm([vaccine_id], correct=False)
        vaccine = bve.get(vaccine_id)
        assert vaccine is not None and not vaccine.retired

    def test_retired_vaccines_stop_firing(self) -> None:
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=True, n=3)
        vaccine_id = bve.vaccines[0].vaccine_id
        for _ in range(20):
            bve.confirm([vaccine_id], correct=False)
        assert bve.immunize(ATTACK, now=100.0).pressure == 0.0

    def test_retire_ineffective_removes_them(self) -> None:
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=True, n=3)
        vaccine_id = bve.vaccines[0].vaccine_id
        for _ in range(20):
            bve.confirm([vaccine_id], correct=False)
        assert bve.retire_ineffective() == [vaccine_id]
        assert len(bve) == 0

    def test_confirming_an_unknown_vaccine_is_a_no_op(self) -> None:
        bve = BehavioralVaccineEngine()
        bve.confirm(["Vaccine-999"], correct=True)  # must not raise


class TestAdaptationScore:
    def test_fresh_engine_has_not_adapted(self) -> None:
        assert BehavioralVaccineEngine().adaptation_score() == 0.0

    def test_learning_raises_the_score(self) -> None:
        # Pinned to the exact weighted sum. "Greater than zero" passes for any
        # inflation of the weights, so it proved nothing about the formula --
        # mutation analysis showed the constants could all shift by +100 and
        # the assertion still held, because the result merely clamped to 1.0.
        bve = BehavioralVaccineEngine(synthesis_threshold=2)
        for index in range(4):
            markers = (f"a{index}", f"b{index}", f"c{index}")
            feed(bve, markers, adverse=True, n=2, t0=index * 10)

        assert len(bve) == 4
        # Four vaccines, each synthesized from two adverse and zero benign
        # observations, so confidence and effectiveness are both 1.0 and mean
        # potency is 1.0. Breadth saturates at eight, so 4/8 == 0.5.
        #   0.70 * mean_potency + 0.30 * breadth == 0.70 * 1.0 + 0.30 * 0.5
        assert bve.adaptation_score() == pytest.approx(0.85)

    def test_freeze_stops_learning_without_losing_knowledge(self) -> None:
        bve = BehavioralVaccineEngine(synthesis_threshold=3)
        feed(bve, ATTACK, adverse=True, n=3)
        assert len(bve) == 1
        bve.freeze()
        # Knowledge retained, observation history cleared.
        assert len(bve) == 1
        feed(bve, ("x", "y", "z"), adverse=True, n=2)
        assert len(bve) == 1


class TestUntrustedPeerImport:
    """Imported profiles are untrusted input and must not break the exchange.

    A malformed record from a peer feed must not raise from inside the lock and
    take the whole memory down with it.
    """

    def test_malformed_numeric_fields_fall_back_to_defaults(self) -> None:
        amx = AttackMemoryExchange()
        accepted = amx.import_profiles(
            [
                {
                    "signature": "sig-junk",
                    "severity": "not-a-number",
                    "frequency": None,
                    "success_rate": {},
                    "last_seen": "yesterday",
                    "first_seen": [],
                }
            ]
        )
        assert accepted == 1
        profile = amx.lookup("sig-junk", now=0.0)
        # Severity floors to 0.0, which is below the forget threshold.
        assert profile is None or profile.severity == 0.0

    def test_malformed_tags_become_empty(self) -> None:
        amx = AttackMemoryExchange()
        amx.import_profiles(
            [{"signature": "sig-a", "severity": 0.9, "tags": "not-a-list"}]
        )
        profile = amx.lookup("sig-a", now=0.0)
        assert profile is not None
        assert profile.tags == frozenset()

    def test_nan_severity_does_not_poison_memory(self) -> None:
        amx = AttackMemoryExchange()
        amx.import_profiles([{"signature": "sig-a", "severity": float("nan")}])
        profile = amx.lookup("sig-a", now=0.0)
        assert profile is None or profile.severity == 0.0

    def test_frequency_never_drops_below_one(self) -> None:
        amx = AttackMemoryExchange()
        amx.import_profiles(
            [{"signature": "sig-a", "severity": 0.9, "frequency": -50}]
        )
        profile = amx.lookup("sig-a", now=0.0)
        assert profile is not None
        assert profile.frequency >= 1
