"""
vaccines.py, BVE, the Behavioral Vaccine Engine.
=================================================

AMX remembers *what* attacked. BVE learns the *shape* of the attack, so the
next variant is caught before it completes rather than after.

The mechanism
-------------
A vaccine is an ordered sequence of behavioural markers that has repeatedly
preceded a bad outcome. The archive's worked example::

    "ignore previous instructions"
    "developer override"
    "show prompt"

Seen together often enough, that sequence becomes ``Vaccine-17``. On a later
encounter, the *second* marker is enough to raise risk, because the engine has
learned what follows. That is the whole value: acting at marker two of three
instead of after marker three.

Immunology, not pattern matching
--------------------------------
Three properties distinguish this from a blocklist, and each maps onto the
biological metaphor the name invokes:

* **Acquired.** Vaccines are synthesized from observed incidents, not authored.
  Nobody writes them; the engine derives them from clusters of bad outcomes.
* **Graded.** A vaccine has confidence and measured effectiveness, and a
  partial sequence match confers partial protection. A blocklist entry is a
  boolean.
* **Waning.** Vaccines whose predictions stop coming true lose effectiveness
  and are eventually retired. This is the property that keeps the engine from
  accumulating superstitions -- a pattern that coincided with three incidents
  and has been wrong ever since is actively removed, not merely outvoted.

Markers are opaque tokens supplied by the integrator, exactly like AMX
signatures. BVE learns sequence and co-occurrence; it never learns content.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from .state_vector import clamp

__all__ = [
    "BehavioralVaccine",
    "BehavioralVaccineEngine",
    "Immunization",
    "Observation",
]

#: Minimum number of incidents sharing a pattern before a vaccine is synthesized.
DEFAULT_SYNTHESIS_THRESHOLD: int = 3

#: Benign sightings of a pattern after which it will never be vaccinated
#: against, whatever the adverse count. See ``_benign_evidence``.
BENIGN_VETO: int = 3

#: Distinct benign marker sets retained as durable evidence. Independent of the
#: observation window, because the window is what the poisoning attack flushes.
BENIGN_LEDGER_CAPACITY: int = 4096

#: Share of the ledger reserved for marker sets seen only once. Without it the
#: ledger ossifies: a full ledger of well-corroborated entries evicts every
#: newcomer at count 1, so a newcomer can never reach count 2, and legitimate
#: new traffic is never recorded as benign. That would make new traffic
#: eligible for false vaccination, which is the failure this ledger exists to
#: prevent, reintroduced one level down.
BENIGN_PROBATION_FRACTION: float = 0.25

#: Vaccines below this effectiveness are retired.
RETIREMENT_FLOOR: float = 0.15

#: Longest marker sequence a vaccine will encode.
MAX_PATTERN_LENGTH: int = 6


@dataclass(frozen=True, slots=True)
class Observation:
    """One behavioural episode reported to BVE by the integrating gateway."""

    markers: tuple[str, ...]
    #: True when this episode ended badly (blocked, exploited, failed policy).
    adverse: bool
    timestamp: float
    #: Optional correlation id linking this episode to an AMX signature.
    signature: str = ""


@dataclass(frozen=True, slots=True)
class BehavioralVaccine:
    """An acquired defence against a recurring behavioural pattern."""

    vaccine_id: str
    #: Ordered marker sequence this vaccine recognizes.
    pattern: tuple[str, ...]
    #: How strongly the pattern predicted an adverse outcome when synthesized.
    confidence: float
    #: How many times this vaccine has fired.
    activation_count: int
    #: Rolling accuracy: of the times it fired, how often it was right.
    effectiveness: float
    created_at: float
    last_activated: float
    #: Firings that were later confirmed correct / incorrect.
    true_positives: int = 0
    false_positives: int = 0

    @property
    def potency(self) -> float:
        """The multiplier this vaccine contributes to risk, in [0, 1].

        Confidence and effectiveness multiply rather than average: a vaccine
        that was confident at synthesis but has been wrong in the field should
        collapse toward zero influence, not settle at the midpoint.
        """
        return clamp(self.confidence * self.effectiveness)

    @property
    def retired(self) -> bool:
        """True once field performance has fallen below the retirement floor."""
        return self.effectiveness < RETIREMENT_FLOOR

    def match(self, markers: Sequence[str]) -> float:
        """Fraction of the pattern present, in order, within ``markers``.

        Subsequence matching rather than contiguous: an attacker interleaving
        benign markers between the steps of a known sequence is still executing
        the known sequence. Returns 0.0 for an empty pattern.
        """
        if not self.pattern:
            return 0.0
        matched = 0
        iterator = iter(markers)
        for element in self.pattern:
            for candidate in iterator:
                if candidate == element:
                    matched += 1
                    break
            else:
                break
        return matched / len(self.pattern)

    def to_dict(self) -> dict[str, object]:
        return {
            "vaccine_id": self.vaccine_id,
            "pattern": list(self.pattern),
            "confidence": round(self.confidence, 4),
            "activation_count": self.activation_count,
            "effectiveness": round(self.effectiveness, 4),
            "potency": round(self.potency, 4),
            "created_at": self.created_at,
            "last_activated": self.last_activated,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
        }


@dataclass(frozen=True, slots=True)
class Immunization:
    """BVE's verdict on a live observation."""

    #: Additional adversarial pressure to apply, in [0, 1].
    pressure: float
    #: Vaccines that fired, strongest first.
    activated: tuple[str, ...]
    #: Highest partial-match fraction across all vaccines.
    best_match: float
    #: True when a vaccine fired on a *partial* pattern -- the early-warning case.
    preemptive: bool

    @classmethod
    def none(cls) -> Immunization:
        return cls(pressure=0.0, activated=(), best_match=0.0, preemptive=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "pressure": round(self.pressure, 4),
            "activated": list(self.activated),
            "best_match": round(self.best_match, 4),
            "preemptive": self.preemptive,
        }


class BehavioralVaccineEngine:
    """Synthesizes, applies, and retires behavioural vaccines.

    Thread-safe. Holds a bounded window of recent observations from which it
    derives candidate patterns.
    """

    __slots__ = (
        "_activation_threshold",
        "_benign_evidence",
        "_benign_index",
        "_capacity",
        "_counter",
        "_lock",
        "_observations",
        "_synthesis_threshold",
        "_vaccines",
    )

    def __init__(
        self,
        capacity: int = 512,
        synthesis_threshold: int = DEFAULT_SYNTHESIS_THRESHOLD,
        observation_window: int = 1024,
        activation_threshold: float = 0.5,
    ) -> None:
        if synthesis_threshold < 2:
            raise ValueError("synthesis_threshold must be at least 2")
        self._capacity = capacity
        self._synthesis_threshold = synthesis_threshold
        self._activation_threshold = clamp(activation_threshold)
        self._observations: deque[Observation] = deque(maxlen=observation_window)
        # Durable, frequency-weighted record of marker sets that have behaved
        # benignly. Deliberately NOT the observation window: the window is
        # bounded, and flushing it by flooding adverse episodes is precisely
        # how an adversary got false vaccines synthesized against legitimate
        # traffic. Evidence that traffic is innocent has to outlive the ring
        # buffer or it is not evidence, it is a recency effect.
        self._benign_evidence: dict[tuple[str, ...], list[float]] = {}
        # marker -> ledger keys containing it. Without this, checking one
        # candidate pattern scans the whole ledger, and synthesis considers
        # ~21 patterns per adverse episode: an adversary would only have to
        # send adverse traffic to make the engine expensive.
        self._benign_index: dict[str, set[tuple[str, ...]]] = {}
        self._vaccines: dict[str, BehavioralVaccine] = {}
        self._counter = 0
        self._lock = threading.RLock()

    # -- learning ---------------------------------------------------------

    def observe(self, observation: Observation) -> list[BehavioralVaccine]:
        """Record an episode and synthesize any vaccines it completes.

        Returns newly created vaccines (usually empty).
        """
        with self._lock:
            self._observations.append(observation)
            if not observation.adverse:
                self._record_benign(observation)
                return []
            return self._synthesize(observation)

    def _record_benign(self, observation: Observation) -> None:
        """Add one benign sighting to the durable ledger. Caller holds the lock."""
        key = tuple(observation.markers)
        record = self._benign_evidence.get(key)
        if record is None:
            self._benign_evidence[key] = [1.0, observation.timestamp]
            for marker in set(key):
                self._benign_index.setdefault(marker, set()).add(key)
        else:
            record[0] += 1.0
            record[1] = max(record[1], observation.timestamp)

        if len(self._benign_evidence) <= BENIGN_LEDGER_CAPACITY:
            return
        self._trim_benign()

    def _drop_benign(self, key: tuple[str, ...]) -> None:
        del self._benign_evidence[key]
        for marker in set(key):
            bucket = self._benign_index.get(marker)
            if bucket is not None:
                bucket.discard(key)
                if not bucket:
                    del self._benign_index[marker]

    def _trim_benign(self) -> None:
        """Bring the ledger back within capacity. Caller holds the lock.

        Partitioned the same way AMX partitions attack memory, and for the same
        reason. Ranking on sighting count alone protects well-corroborated
        evidence, which is what the poisoning attack needs to displace, but it
        also means a full ledger evicts every newcomer at count 1 before it can
        ever reach count 2. Reserving a share for newcomers keeps the ledger
        able to learn without making established evidence cheap to flush.

        Within each region: fewest sightings first, oldest breaking ties. A
        marker set seen hundreds of times is close to immovable, and that is
        the exchange rate. Burying it means out-generating it in *successful*
        traffic, not merely in adverse episodes, and an adversary whose traffic
        succeeds that often did not need the vaccine attack.
        """
        established = [k for k, v in self._benign_evidence.items() if v[0] >= 2.0]
        probation = [k for k, v in self._benign_evidence.items() if v[0] < 2.0]
        weakest = lambda keys: sorted(  # noqa: E731
            keys, key=lambda k: (self._benign_evidence[k][0], self._benign_evidence[k][1])
        )

        allowance = max(
            int(BENIGN_LEDGER_CAPACITY * BENIGN_PROBATION_FRACTION),
            BENIGN_LEDGER_CAPACITY - len(established),
        )
        for key in weakest(probation)[: max(0, len(probation) - allowance)]:
            self._drop_benign(key)

        overflow = len(self._benign_evidence) - BENIGN_LEDGER_CAPACITY
        if overflow > 0:
            for key in weakest(established)[:overflow]:
                self._drop_benign(key)

    def _benign_support(self, pattern: Sequence[str]) -> int:
        """Total durable benign sightings of any marker set containing ``pattern``.

        Narrowed through the marker index first. Containing the pattern
        requires containing every marker in it, so intersecting the index
        buckets gives a small candidate set, and only those are order-checked.
        """
        candidates: set[tuple[str, ...]] | None = None
        for marker in pattern:
            bucket = self._benign_index.get(marker)
            if not bucket:
                return 0
            candidates = bucket.copy() if candidates is None else (candidates & bucket)
            if not candidates:
                return 0
        if not candidates:
            return 0
        return int(
            sum(
                self._benign_evidence[markers][0]
                for markers in candidates
                if _contains(markers, pattern)
            )
        )

    def _synthesize(self, trigger: Observation) -> list[BehavioralVaccine]:
        """Derive vaccines from patterns recurring across adverse episodes.

        Caller holds the lock. Considers every contiguous sub-sequence of the
        triggering episode's markers, from longest to shortest, and promotes
        those that appear in enough adverse episodes and are discriminative --
        that is, they do *not* also appear throughout benign traffic.
        """
        created: list[BehavioralVaccine] = []
        markers = trigger.markers[:MAX_PATTERN_LENGTH]
        if len(markers) < 2:
            return created

        adverse = [o for o in self._observations if o.adverse]
        benign = [o for o in self._observations if not o.adverse]

        for length in range(len(markers), 1, -1):
            for start in range(0, len(markers) - length + 1):
                pattern = markers[start : start + length]
                if self._has_pattern(pattern):
                    continue

                adverse_hits = sum(1 for o in adverse if _contains(o.markers, pattern))
                if adverse_hits < self._synthesis_threshold:
                    continue

                # Counted from the durable ledger, not only the window. The
                # window is bounded, so an adversary who floods adverse
                # episodes past its length evicts the benign observations that
                # would otherwise block synthesis, and the ratio below then
                # sees an unopposed pattern.
                windowed = sum(1 for o in benign if _contains(o.markers, pattern))
                benign_hits = max(windowed, self._benign_support(pattern))

                # An absolute veto, not just a ratio. A ratio is whoever
                # generates more volume, and the adversary controls the adverse
                # side of it: with durable benign evidence alone they need only
                # out-produce it by a factor of two or three. But a pattern
                # known to occur in legitimate traffic must never be vaccinated
                # against at any ratio, because the vaccine will fire on that
                # traffic, and the cost of blocking real users is not something
                # a higher adverse count makes acceptable. If an attack really
                # does reuse a benign pattern, it has to be stopped by
                # signature or policy, not by a rule that also hits customers.
                if benign_hits >= BENIGN_VETO:
                    continue

                total = adverse_hits + benign_hits
                # Discriminative power: P(adverse | pattern). A pattern that
                # shows up just as often in benign traffic predicts nothing,
                # and vaccinating against it would be pure false positives.
                confidence = adverse_hits / total if total else 0.0
                if confidence < 0.70:
                    continue

                self._counter += 1
                vaccine = BehavioralVaccine(
                    vaccine_id=f"Vaccine-{self._counter}",
                    pattern=pattern,
                    confidence=clamp(confidence),
                    activation_count=0,
                    # Start at confidence: the vaccine is presumed as effective
                    # as it was discriminative until the field says otherwise.
                    effectiveness=clamp(confidence),
                    created_at=trigger.timestamp,
                    last_activated=0.0,
                )
                self._vaccines[vaccine.vaccine_id] = vaccine
                created.append(vaccine)
                self._evict_if_needed()
                # One vaccine per episode: synthesizing every sub-sequence of a
                # single incident would flood the store with near-duplicates
                # that all fire together and triple-count the same evidence.
                return created

        return created

    @staticmethod
    def _pattern_key(pattern: Sequence[str]) -> tuple[str, ...]:
        return tuple(pattern)

    def _has_pattern(self, pattern: Sequence[str]) -> bool:
        key = self._pattern_key(pattern)
        return any(v.pattern == key for v in self._vaccines.values())

    def _evict_if_needed(self) -> None:
        """Drop the least potent vaccines when over capacity. Caller holds lock."""
        if len(self._vaccines) <= self._capacity:
            return
        ranked = sorted(self._vaccines.values(), key=lambda v: v.potency)
        for vaccine in ranked[: len(self._vaccines) - self._capacity]:
            self._vaccines.pop(vaccine.vaccine_id, None)

    # -- application ------------------------------------------------------

    def immunize(self, markers: Sequence[str], now: float | None = None) -> Immunization:
        """Evaluate live markers against the vaccine store.

        This is the call that happens *during* an attack, before it completes.
        """
        timestamp = time.time() if now is None else now
        if not markers:
            return Immunization.none()

        fired: list[tuple[str, float]] = []
        best_match = 0.0
        preemptive = False

        with self._lock:
            for vaccine in list(self._vaccines.values()):
                if vaccine.retired:
                    continue
                fraction = vaccine.match(markers)
                best_match = max(best_match, fraction)
                if fraction < self._activation_threshold:
                    continue

                # Partial match on a known escalation is the early warning the
                # whole subsystem exists to produce. Scale contribution by how
                # much of the pattern is present so a full match still
                # outweighs a partial one.
                if fraction < 1.0:
                    preemptive = True

                contribution = clamp(vaccine.potency * fraction)
                if contribution <= 0.0:
                    continue
                fired.append((vaccine.vaccine_id, contribution))
                self._vaccines[vaccine.vaccine_id] = replace(
                    vaccine,
                    activation_count=vaccine.activation_count + 1,
                    last_activated=timestamp,
                )

        if not fired:
            return Immunization(
                pressure=0.0, activated=(), best_match=best_match, preemptive=False
            )

        fired.sort(key=lambda item: item[1], reverse=True)
        union = 1.0
        for _, contribution in fired:
            union *= 1.0 - contribution
        return Immunization(
            pressure=clamp(1.0 - union),
            activated=tuple(vid for vid, _ in fired),
            best_match=best_match,
            preemptive=preemptive,
        )

    # -- feedback ---------------------------------------------------------

    def confirm(self, vaccine_ids: Iterable[str], correct: bool) -> None:
        """Report whether a firing turned out to be right.

        This closes the loop that turns a static pattern store into a learning
        one. Effectiveness is a Laplace-smoothed success rate, so a single
        early mistake does not zero out a vaccine and a single early success
        does not certify one.
        """
        with self._lock:
            for vaccine_id in vaccine_ids:
                vaccine = self._vaccines.get(vaccine_id)
                if vaccine is None:
                    continue
                true_positives = vaccine.true_positives + (1 if correct else 0)
                false_positives = vaccine.false_positives + (0 if correct else 1)
                effectiveness = (true_positives + 1.0) / (
                    true_positives + false_positives + 2.0
                )
                self._vaccines[vaccine_id] = replace(
                    vaccine,
                    true_positives=true_positives,
                    false_positives=false_positives,
                    effectiveness=clamp(effectiveness),
                )

    def retire_ineffective(self) -> list[str]:
        """Remove vaccines that have fallen below the retirement floor."""
        with self._lock:
            retired = [v.vaccine_id for v in self._vaccines.values() if v.retired]
            for vaccine_id in retired:
                del self._vaccines[vaccine_id]
            return retired

    def freeze(self) -> None:
        """Stop learning without discarding what is already known.

        The recovery planner sets this during CASCADING and QUARANTINE: an
        engine that keeps synthesizing vaccines from the noise of its own
        collapse learns the collapse, not the attack.
        """
        with self._lock:
            self._observations.clear()
            self._benign_evidence.clear()
            self._benign_index.clear()

    # -- introspection ----------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._vaccines)

    @property
    def vaccines(self) -> tuple[BehavioralVaccine, ...]:
        with self._lock:
            return tuple(self._vaccines.values())

    def get(self, vaccine_id: str) -> BehavioralVaccine | None:
        with self._lock:
            return self._vaccines.get(vaccine_id)

    def adaptation_score(self) -> float:
        """How well-adapted the engine is, in [0, 1].

        Feeds the "adaptation" term of the resilience index. A system with
        effective vaccines has genuinely learned from what hit it; a system
        with none, or with only discredited ones, has not.
        """
        with self._lock:
            live = [v for v in self._vaccines.values() if not v.retired]
        if not live:
            return 0.0
        mean_potency = sum(v.potency for v in live) / len(live)
        # Breadth matters but saturates: eight good vaccines is a mature
        # immune memory, eight hundred is not a hundred times better.
        breadth = clamp(len(live) / 8.0)
        return clamp(mean_potency * 0.7 + breadth * 0.3)

    def export_vaccines(self) -> list[dict[str, object]]:
        with self._lock:
            return [v.to_dict() for v in self._vaccines.values()]


def _contains(markers: Sequence[str], pattern: Sequence[str]) -> bool:
    """True when ``pattern`` appears as an ordered subsequence of ``markers``."""
    iterator = iter(markers)
    return all(any(candidate == element for candidate in iterator) for element in pattern)
