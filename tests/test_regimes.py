"""Tests for regime classification.

The reachability sweep is carried forward from the archive, where it was the
check that exposed the uncentered-energy bug. It is the guard that keeps a
future edit to the energy function or the affinity weights from silently
making a regime impossible to reach.
"""

from __future__ import annotations

import itertools

import pytest

from ure_engine.lyapunov import compute_energy
from ure_engine.regimes import (
    CLASSIFIABLE,
    RegimeClassifier,
    RegimeProfile,
    SystemRegime,
)
from ure_engine.state_vector import StateVector


@pytest.fixture()
def classifier() -> RegimeClassifier:
    return RegimeClassifier()


def state(**pressures: float) -> StateVector:
    return StateVector.create(timestamp=0.0, **pressures)


class TestSystemRegime:
    def test_health_status_projection(self) -> None:
        assert SystemRegime.NOMINAL.health_status == "NEUTRAL"
        assert SystemRegime.RECOVERING.health_status == "NEUTRAL"
        assert SystemRegime.ADAPTING.health_status == "NEUTRAL"
        assert SystemRegime.STRESSED.health_status == "RISK_INCREASING"
        assert SystemRegime.ATTACKED.health_status == "RISK_INCREASING"
        assert SystemRegime.CASCADING.health_status == "REGRESSIVE"
        assert SystemRegime.UNKNOWN.health_status == "UNKNOWN"

    def test_severity_is_ordered(self) -> None:
        assert SystemRegime.NOMINAL.severity < SystemRegime.ADAPTING.severity
        assert SystemRegime.STRESSED.severity < SystemRegime.ATTACKED.severity
        assert SystemRegime.ATTACKED.severity < SystemRegime.CASCADING.severity

    def test_unknown_is_not_treated_as_mild(self) -> None:
        # An engine that cannot classify itself is not in a benign condition.
        assert SystemRegime.UNKNOWN.severity > SystemRegime.STRESSED.severity

    def test_hostility_flag(self) -> None:
        assert SystemRegime.ATTACKED.is_hostile
        assert SystemRegime.CASCADING.is_hostile
        assert not SystemRegime.STRESSED.is_hostile

    def test_serializes_as_its_value(self) -> None:
        assert SystemRegime.ATTACKED.value == "ATTACKED"
        assert f"{SystemRegime.ATTACKED}".endswith("ATTACKED")


class TestReachability:
    """Every regime must be reachable; UNKNOWN must never appear."""

    def test_full_state_space_sweep(self, classifier: RegimeClassifier) -> None:
        levels = (0.0, 0.25, 0.5, 0.75, 1.0)
        seen: set[SystemRegime] = set()

        for combination in itertools.product(levels, repeat=6):
            vector = StateVector(*combination, timestamp=0.0)
            energy = compute_energy(vector)
            for derivative in (-0.08, -0.01, 0.0, 0.01, 0.08):
                for volatility in (0.0, 0.05, 0.2):
                    profile = classifier.classify(vector, energy, derivative, volatility)
                    seen.add(profile.dominant)

        missing = set(CLASSIFIABLE) - seen
        assert not missing, f"unreachable regimes: {sorted(r.value for r in missing)}"
        assert SystemRegime.UNKNOWN not in seen, "UNKNOWN dead zone reintroduced"

    def test_idle_system_is_nominal(self, classifier: RegimeClassifier) -> None:
        # The specific case the uncentered-energy bug broke.
        vector = StateVector.zero()
        profile = classifier.classify(vector, compute_energy(vector), 0.0, 0.0)
        assert profile.dominant is SystemRegime.NOMINAL
        assert profile.confidence > 0.8


class TestClassification:
    def test_hostile_pressure_yields_attacked(self, classifier: RegimeClassifier) -> None:
        vector = state(threat_pressure=0.9, adversarial_pressure=0.8)
        profile = classifier.classify(vector, compute_energy(vector), 0.0, 0.0)
        assert profile.dominant is SystemRegime.ATTACKED

    def test_operational_pressure_yields_stressed_not_attacked(
        self, classifier: RegimeClassifier
    ) -> None:
        # The asymmetry the whole regime model exists for: load is not malice,
        # and the two call for opposite responses.
        vector = state(latency_pressure=0.9, failure_pressure=0.9, resource_pressure=0.9)
        profile = classifier.classify(vector, compute_energy(vector), 0.0, 0.0)
        assert profile.dominant is SystemRegime.STRESSED

    def test_high_rising_broad_failure_yields_cascading(
        self, classifier: RegimeClassifier
    ) -> None:
        vector = state(
            threat_pressure=0.8,
            latency_pressure=0.9,
            failure_pressure=1.0,
            resource_pressure=1.0,
            adversarial_pressure=0.7,
        )
        profile = classifier.classify(vector, compute_energy(vector), 0.10, 0.05)
        assert profile.dominant is SystemRegime.CASCADING

    def test_falling_energy_yields_recovering(self, classifier: RegimeClassifier) -> None:
        vector = state(drift_pressure=0.9)
        profile = classifier.classify(vector, compute_energy(vector), -0.08, 0.0)
        assert profile.dominant is SystemRegime.RECOVERING

    def test_mid_band_with_volatility_yields_adapting(
        self, classifier: RegimeClassifier
    ) -> None:
        vector = state(drift_pressure=0.9)
        profile = classifier.classify(vector, compute_energy(vector), 0.0, 0.20)
        assert profile.dominant is SystemRegime.ADAPTING

    def test_receding_attack_is_not_still_attacked(
        self, classifier: RegimeClassifier
    ) -> None:
        # An adversary backing off should read as recovery, not as ongoing
        # attack, or the engine can never leave the ATTACKED regime.
        vector = state(threat_pressure=0.75)
        rising = classifier.classify(vector, compute_energy(vector), 0.08, 0.0)
        falling = classifier.classify(vector, compute_energy(vector), -0.08, 0.0)
        assert rising.dominant is SystemRegime.ATTACKED
        assert falling.probability(SystemRegime.RECOVERING) > rising.probability(
            SystemRegime.RECOVERING
        )


class TestDistribution:
    def test_probabilities_sum_to_one(self, classifier: RegimeClassifier) -> None:
        vector = state(threat_pressure=0.5, failure_pressure=0.4)
        profile = classifier.classify(vector, compute_energy(vector), 0.01, 0.05)
        assert sum(profile.distribution.values()) == pytest.approx(1.0)

    def test_every_candidate_gets_a_probability(
        self, classifier: RegimeClassifier
    ) -> None:
        vector = state(threat_pressure=0.5)
        profile = classifier.classify(vector, compute_energy(vector), 0.0, 0.0)
        assert set(profile.distribution) == set(CLASSIFIABLE)

    def test_confidence_is_the_dominant_mass(self, classifier: RegimeClassifier) -> None:
        vector = state(threat_pressure=0.9)
        profile = classifier.classify(vector, compute_energy(vector), 0.0, 0.0)
        assert profile.confidence == pytest.approx(
            profile.distribution[profile.dominant]
        )

    def test_margin_separates_the_top_two(self, classifier: RegimeClassifier) -> None:
        vector = state(threat_pressure=0.9)
        profile = classifier.classify(vector, compute_energy(vector), 0.0, 0.0)
        assert profile.margin == pytest.approx(
            profile.confidence - profile.distribution[profile.runner_up]
        )
        assert profile.margin >= 0.0

    def test_clear_case_has_low_entropy(self, classifier: RegimeClassifier) -> None:
        vector = StateVector.zero()
        profile = classifier.classify(vector, compute_energy(vector), 0.0, 0.0)
        assert profile.entropy < 0.3

    def test_entropy_is_normalized(self, classifier: RegimeClassifier) -> None:
        uniform = {regime: 1.0 / len(CLASSIFIABLE) for regime in CLASSIFIABLE}
        assert RegimeClassifier.entropy(uniform) == pytest.approx(1.0)
        certain = {regime: 0.0 for regime in CLASSIFIABLE}
        certain[SystemRegime.NOMINAL] = 1.0
        assert RegimeClassifier.entropy(certain) == pytest.approx(0.0)

    def test_ambiguity_flag_tracks_margin_and_confidence(self) -> None:
        clear = RegimeProfile(
            dominant=SystemRegime.NOMINAL,
            distribution={SystemRegime.NOMINAL: 0.9},
            confidence=0.9,
            entropy=0.1,
            runner_up=SystemRegime.ADAPTING,
            margin=0.8,
        )
        assert not clear.ambiguous

        close = RegimeProfile(
            dominant=SystemRegime.STRESSED,
            distribution={SystemRegime.STRESSED: 0.35},
            confidence=0.35,
            entropy=0.9,
            runner_up=SystemRegime.ATTACKED,
            margin=0.02,
        )
        assert close.ambiguous


class TestMemoryInput:
    def test_recall_sharpens_attacked_but_cannot_invent_it(
        self, classifier: RegimeClassifier
    ) -> None:
        # Memory enters multiplicatively alongside observed hostile pressure,
        # so a false AMX match on a quiet system cannot manufacture an attack.
        quiet = StateVector.zero()
        without = classifier.classify(quiet, compute_energy(quiet), 0.0, 0.0)
        with_recall = classifier.classify(
            quiet, compute_energy(quiet), 0.0, 0.0, recall_strength=1.0
        )
        assert without.dominant is SystemRegime.NOMINAL
        assert with_recall.dominant is SystemRegime.NOMINAL

    def test_recall_raises_attacked_probability_when_hostility_is_present(
        self, classifier: RegimeClassifier
    ) -> None:
        vector = state(threat_pressure=0.5)
        energy = compute_energy(vector)
        without = classifier.classify(vector, energy, 0.0, 0.0)
        with_recall = classifier.classify(vector, energy, 0.0, 0.0, recall_strength=1.0)
        assert with_recall.probability(SystemRegime.ATTACKED) > without.probability(
            SystemRegime.ATTACKED
        )


class TestDegenerateInput:
    def test_non_finite_energy_yields_unknown(self, classifier: RegimeClassifier) -> None:
        profile = classifier.classify(StateVector.zero(), float("nan"), 0.0, 0.0)
        assert profile.dominant is SystemRegime.UNKNOWN
        assert profile.confidence == 0.0

    def test_unknown_profile_reports_maximum_entropy(self) -> None:
        profile = RegimeProfile.unknown()
        assert profile.entropy == 1.0
        assert profile.confidence == 0.0

    def test_rejects_non_positive_energy_max(self) -> None:
        with pytest.raises(ValueError):
            RegimeClassifier(energy_max=0.0)


class TestSerialization:
    def test_profile_round_trips_to_json_safe_dict(
        self, classifier: RegimeClassifier
    ) -> None:
        import json

        vector = state(threat_pressure=0.6)
        profile = classifier.classify(vector, compute_energy(vector), 0.0, 0.0)
        payload = json.loads(json.dumps(profile.to_dict()))
        assert payload["dominant"] == profile.dominant.value
        assert set(payload["distribution"]) == {r.value for r in CLASSIFIABLE}
