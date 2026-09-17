"""
attack_memory.py, AMX, the Attack Memory Exchange.
===================================================

AMX is what makes URE historically aware. Without it the engine is purely
reactive: every request is the first request it has ever seen, and an adversary
who stays below the instantaneous threshold is invisible no matter how many
times they probe. AMX remembers signatures, ages them out, and raises
adversarial pressure when something familiar comes back.

Why this was missing
--------------------
The hardened predecessor removed AMX with the note that it was "instantiated
but never called". That was an accurate description of the code at the time and
the right call for that build -- dead subsystems are worse than absent ones.
But the reason it was never called is that nothing fed it: the engine had no
signature channel. Here it has one, through
:class:`~ure_engine.state_vector.PolicyTelemetry`, and AMX's output lands on
``adversarial_pressure`` where the energy function already weights it most
heavily. It is wired, or it is not shipped.

What a signature is
-------------------
An opaque string supplied by the integrating gateway -- typically a hash of
normalized request features. URE never sees request content and cannot
reconstruct it from the signature. That boundary is what keeps AMX compatible
with the policy seam: AMX correlates on identity without learning meaning.

Decay
-----
Profiles lose severity exponentially with a configurable half-life. An attack
campaign that stopped six hours ago should not still be inflating adversarial
pressure, but neither should it be forgotten the instant it pauses -- pausing
to evade a memory window is the obvious counter-move. Exponential decay with a
multi-hour half-life makes that counter-move expensive without making the
memory permanent.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace

from .state_vector import clamp

__all__ = ["AttackMemoryExchange", "AttackProfile", "MemoryRecall"]


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce an imported field to float, falling back to ``default``.

    Peer profile feeds are untrusted input: a malformed record must not raise
    from inside the exchange's lock and take the whole memory down with it.
    """
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) else result


def _as_int(value: object, default: int = 0) -> int:
    """Integer counterpart of :func:`_as_float`."""
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default


def _as_tags(value: object) -> tuple[object, ...]:
    """Coerce an imported tags field to a tuple; anything unusable becomes empty."""
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return ()

#: Default half-life for profile severity, in seconds (6 hours).
DEFAULT_HALF_LIFE: float = 6 * 60 * 60.0

#: Profiles below this effective severity are eligible for eviction.
FORGET_THRESHOLD: float = 0.01

#: Half-life of the new-signature admission counter, in seconds. Sets what
#: counts as "arriving in a crowd" for the purposes of eviction ranking.
ADMISSION_HALF_LIFE: float = 60.0

#: Recent distinct new signatures at which admission is treated as a full
#: flood. A deployment discovering a new signature every few minutes sits near
#: zero; one discovering several a second saturates it.
BURST_SIGNATURES: float = 20.0

#: Fraction of capacity reserved for signatures seen only once. Uncorroborated
#: profiles compete only for this share, which is what makes a flood of minted
#: signatures structurally unable to displace established memory rather than
#: merely unlikely to. See :meth:`AttackMemoryExchange._evict_if_needed`.
PROBATION_FRACTION: float = 0.25

#: Persistence, in seconds, at which recurrence counts as half its maximum
#: evidence. Recurring across five minutes is materially stronger than
#: recurring across one second, and elapsed time is the one thing an adversary
#: cannot mint: they can claim any severity they like, but they cannot claim to
#: have been here longer than they have.

#: Window over which a full memory being replaced counts as a flood.
FLOOD_WINDOW: float = 300.0
PERSISTENCE_SCALE: float = 300.0


@dataclass(frozen=True, slots=True)
class AttackProfile:
    """One remembered adversarial signature.

    Immutable: every update produces a new profile, so a recall taken during
    an update always sees a coherent snapshot rather than a half-written one.
    """

    signature: str
    #: Peak severity observed, in [0, 1]. Decays with age.
    severity: float
    #: How many times this signature has been seen.
    frequency: int
    #: Fraction of encounters where the attack got through.
    success_rate: float
    #: Epoch seconds of the most recent sighting.
    last_seen: float
    #: Epoch seconds of the first sighting.
    first_seen: float
    #: Free-form labels supplied by the integrator (campaign id, source ASN).
    tags: frozenset[str] = field(default_factory=frozenset)
    #: How crowded this signature's first sighting was, in [0, 1]. See
    #: :attr:`corroboration`.
    admission_burst: float = 0.0

    def effective_severity(self, now: float, half_life: float = DEFAULT_HALF_LIFE) -> float:
        """Severity discounted for time since ``last_seen``.

        A signature seen once an hour ago carries less weight than the same
        signature seen a minute ago, without either being deleted.
        """
        if half_life <= 0.0:
            return self.severity
        age = max(0.0, now - self.last_seen)
        return self.severity * math.pow(0.5, age / half_life)

    @property
    def persistence(self) -> float:
        """How long this signature has been recurring, in seconds."""
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def corroboration(self) -> float:
        """How much independent evidence stands behind this profile, in [0, 1].

        This exists because **severity is attacker-supplied and corroboration
        is not**. ``remember`` takes the severity its caller reports, and an
        adversary minting signatures reports whatever they like. Ranking
        eviction on severity alone therefore lets anyone who can claim 0.99
        evict a real profile sitting at 0.95, which is exactly the flooding
        attack in ``tests/test_adversarial.py``.

        Recurrence cannot be claimed, only spent. A campaign has been seen
        repeatedly across a span of time, which costs the adversary both
        patience and repeated exposure, so it ranks above anything asserted
        once.

        The hard case is a genuine first sighting, which carries no more
        evidence than a decoy does. The one thing that separates them is the
        company it arrived in: a signature admitted while hundreds of others
        are streaming in is far more likely to be part of the flood than one
        admitted during quiet. That is a probabilistic discount, not proof, so
        it is weighted rather than absolute, and it is not what actually stops
        the flood. The capacity partition in ``_evict_if_needed`` does that.

        Both components saturate rather than step, so there is no boundary for
        an adversary to sit exactly on top of.
        """
        if self.frequency < 2:
            return 0.25 * (1.0 - clamp(self.admission_burst))
        repeats = 1.0 - math.exp(-(self.frequency - 1) / 3.0)
        span = self.persistence / (self.persistence + PERSISTENCE_SCALE)
        return clamp(0.30 + 0.35 * repeats + 0.35 * span)

    def retention(self, now: float, half_life: float = DEFAULT_HALF_LIFE) -> float:
        """Eviction rank: what this profile is worth keeping. Lowest goes first.

        Corroboration dominates and severity only orders within a band of it.
        That split is the whole point: severity is a number the caller supplied
        and an adversary supplies whatever wins, so letting it decide rank hands
        them the memory for the price of claiming 0.99. Evidence that costs
        elapsed time cannot be supplied, only spent.
        """
        return self.corroboration + 0.15 * self.effective_severity(now, half_life)

    @property
    def is_campaign(self) -> bool:
        """True for signatures that recur over time rather than spike once.

        A campaign is materially more dangerous than a burst: it implies an
        adversary iterating against the defence rather than a scripted probe.
        """
        return self.frequency >= 3 and self.persistence >= 60.0

    def to_dict(self) -> dict[str, object]:
        return {
            "signature": self.signature,
            "severity": round(self.severity, 4),
            "frequency": self.frequency,
            "success_rate": round(self.success_rate, 4),
            "last_seen": self.last_seen,
            "first_seen": self.first_seen,
            "is_campaign": self.is_campaign,
            "admission_burst": round(self.admission_burst, 4),
            "tags": sorted(self.tags),
        }


@dataclass(frozen=True, slots=True)
class MemoryRecall:
    """What AMX has to say about the current moment."""

    #: Aggregate decayed severity across matching profiles, in [0, 1].
    pressure: float
    #: Confidence that this moment matches something remembered, in [0, 1].
    strength: float
    #: Signatures of the profiles that matched, strongest first.
    matches: tuple[str, ...]
    #: True when at least one match is a recurring campaign.
    campaign: bool

    @classmethod
    def empty(cls) -> MemoryRecall:
        return cls(pressure=0.0, strength=0.0, matches=(), campaign=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "pressure": round(self.pressure, 4),
            "strength": round(self.strength, 4),
            "matches": list(self.matches),
            "campaign": self.campaign,
        }


class AttackMemoryExchange:
    """Bounded, decaying, thread-safe store of adversarial signatures.

    "Exchange" rather than "store" because the design intends profiles to be
    shareable between deployments: :meth:`export_profiles` and
    :meth:`import_profiles` round-trip the memory so a fleet can pool what it
    has learned. Nothing in a profile is request content, so sharing them does
    not move user data.
    """

    __slots__ = (
        "_admissions",
        "_capacity",
        "_half_life",
        "_last_admission",
        "_lock",
        "_profiles",
    )

    def __init__(self, capacity: int = 4096, half_life: float = DEFAULT_HALF_LIFE) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        # Decaying count of recent first sightings, used to tell a signature
        # that arrived alone from one that arrived in a flood.
        self._admissions = 0.0
        self._last_admission: float | None = None
        self._half_life = half_life
        self._profiles: dict[str, AttackProfile] = {}
        self._lock = threading.RLock()

    # -- write ------------------------------------------------------------

    def remember(
        self,
        signature: str,
        severity: float,
        *,
        succeeded: bool = False,
        now: float | None = None,
        tags: Iterable[str] = (),
    ) -> AttackProfile:
        """Record a sighting, merging into any existing profile.

        Severity takes the maximum of the decayed existing value and the new
        observation: an attack does not become less dangerous because its most
        recent instance happened to score lower.
        """
        timestamp = time.time() if now is None else now
        clamped = clamp(severity)

        with self._lock:
            existing = self._profiles.get(signature)
            if existing is None:
                profile = AttackProfile(
                    signature=signature,
                    severity=clamped,
                    frequency=1,
                    success_rate=1.0 if succeeded else 0.0,
                    last_seen=timestamp,
                    first_seen=timestamp,
                    tags=frozenset(tags),
                    admission_burst=self._note_admission(timestamp),
                )
            else:
                decayed = existing.effective_severity(timestamp, self._half_life)
                frequency = existing.frequency + 1
                successes = existing.success_rate * existing.frequency + (1.0 if succeeded else 0.0)
                profile = replace(
                    existing,
                    severity=max(decayed, clamped),
                    frequency=frequency,
                    success_rate=successes / frequency,
                    last_seen=timestamp,
                    tags=existing.tags | frozenset(tags),
                )

            self._profiles[signature] = profile
            self._evict_if_needed(timestamp)
            return profile

    def forget(self, signature: str) -> bool:
        """Drop a signature outright. Returns True if it was present."""
        with self._lock:
            return self._profiles.pop(signature, None) is not None

    def decay(self, now: float | None = None) -> int:
        """Materialize decay and evict exhausted profiles; returns the count dropped.

        Recall applies decay lazily on read, so calling this is an optimization
        (bounding memory) rather than a correctness requirement.
        """
        timestamp = time.time() if now is None else now
        with self._lock:
            stale = [
                signature
                for signature, profile in self._profiles.items()
                if profile.effective_severity(timestamp, self._half_life) < FORGET_THRESHOLD
            ]
            for signature in stale:
                del self._profiles[signature]
            return len(stale)

    def clear(self) -> None:
        with self._lock:
            self._profiles.clear()

    def _note_admission(self, now: float) -> float:
        """Register a first sighting and report how crowded it was.

        Returns the burst level *before* counting this one, so the very first
        signature a deployment ever sees is never treated as part of a flood.
        Caller holds the lock.
        """
        previous = self._last_admission
        self._last_admission = now
        if previous is not None:
            elapsed = max(0.0, now - previous)
            self._admissions *= math.pow(0.5, elapsed / ADMISSION_HALF_LIFE)
        burst = clamp(self._admissions / BURST_SIGNATURES)
        self._admissions += 1.0
        return burst

    def _evict_if_needed(self, now: float) -> None:
        """Make room, taking it from the least corroborated memory first.

        Ranking on severity alone is what made the flooding attack work:
        severity is whatever the caller reported, so an adversary minting
        signatures at 0.99 evicted a real profile sitting at 0.95, and the
        memory forgot the one thing it most needed to remember. Ranking on
        :meth:`AttackProfile.retention` fixes the ordering, but ordering alone
        is a soft defence: an adversary who out-claims the incumbent by a
        hair still wins.

        So capacity is partitioned rather than merely ranked. Signatures seen
        exactly once compete only for ``PROBATION_FRACTION`` of the memory,
        and a flood of them can therefore evict nothing but each other, at any
        volume and any claimed severity. Established profiles are touched only
        when the established region is itself full, which no amount of minting
        can cause.

        Probation may borrow whatever the established region is not using, so
        a quiet deployment still gets the whole memory. Caller holds the lock.
        """
        if len(self._profiles) <= self._capacity:
            return

        def weakest_first(profiles: list[AttackProfile]) -> list[AttackProfile]:
            return sorted(profiles, key=lambda p: p.retention(now, self._half_life))

        established = [p for p in self._profiles.values() if p.frequency >= 2]
        probation = [p for p in self._profiles.values() if p.frequency < 2]

        # Probation gets its reserved share, plus anything established is not
        # using. It never gets to push established out.
        allowance = max(
            int(self._capacity * PROBATION_FRACTION),
            self._capacity - len(established),
        )
        for profile in weakest_first(probation)[: max(0, len(probation) - allowance)]:
            self._profiles.pop(profile.signature, None)

        # Only if the established region has genuinely outgrown the memory on
        # its own does corroborated history start being dropped.
        overflow = len(self._profiles) - self._capacity
        if overflow > 0:
            for profile in weakest_first(established)[:overflow]:
                self._profiles.pop(profile.signature, None)

    # -- read -------------------------------------------------------------

    def lookup(self, signature: str, now: float | None = None) -> AttackProfile | None:
        """Fetch a profile if it is still above the forget threshold."""
        timestamp = time.time() if now is None else now
        with self._lock:
            profile = self._profiles.get(signature)
            if profile is None:
                return None
            if profile.effective_severity(timestamp, self._half_life) < FORGET_THRESHOLD:
                return None
            return profile

    def recall(
        self, signatures: Iterable[str], now: float | None = None
    ) -> MemoryRecall:
        """Assess the current moment against memory.

        Aggregate pressure uses a probabilistic union (``1 - prod(1 - s)``)
        rather than a sum. Two independent half-severity matches should read as
        worse than either alone but still short of certainty; a sum would
        saturate at 1.0 after two matches and then stop distinguishing two
        matches from twenty.
        """
        timestamp = time.time() if now is None else now
        matched: list[tuple[str, float]] = []
        campaign = False

        with self._lock:
            for signature in signatures:
                profile = self._profiles.get(signature)
                if profile is None:
                    continue
                severity = profile.effective_severity(timestamp, self._half_life)
                if severity < FORGET_THRESHOLD:
                    continue
                matched.append((signature, severity))
                campaign = campaign or profile.is_campaign

        if not matched:
            return MemoryRecall.empty()

        matched.sort(key=lambda item: item[1], reverse=True)
        union = 1.0
        for _, severity in matched:
            union *= 1.0 - severity
        pressure = clamp(1.0 - union)

        # Strength is about identification confidence, not danger: the best
        # single match, nudged up when the sighting is part of a campaign.
        strength = clamp(matched[0][1] * (1.15 if campaign else 1.0))

        return MemoryRecall(
            pressure=pressure,
            strength=strength,
            matches=tuple(signature for signature, _ in matched),
            campaign=campaign,
        )

    def ambient_pressure(self, now: float | None = None, top_n: int = 8) -> float:
        """Background adversarial pressure from the whole memory, in [0, 1].

        Applied even when the current request matches nothing: a deployment
        that has been under sustained attack for an hour is in a different
        posture than one that has been idle, regardless of what this particular
        request looks like. Capped to the strongest ``top_n`` profiles so a
        large memory of weak signatures cannot manufacture pressure by volume.
        """
        timestamp = time.time() if now is None else now
        with self._lock:
            severities = sorted(
                (
                    profile.effective_severity(timestamp, self._half_life)
                    for profile in self._profiles.values()
                ),
                reverse=True,
            )[:top_n]
        if not severities:
            return 0.0
        union = 1.0
        for severity in severities:
            union *= 1.0 - severity
        # Damped: ambient history should colour the assessment, not dominate it.
        return clamp((1.0 - union) * 0.5)

    def flood_pressure(self, now: float | None = None) -> float:
        """Evidence that the memory itself is under a replacement attack.

        An adversary who can afford corroborated decoys cannot be stopped by
        eviction ranking: they will match whatever evidence shape the incumbent
        has and out-claim its severity by a hair. What they cannot do is hide.
        Displacing a full memory means minting capacity-many signatures and
        making each recur, which is thousands of adverse events in minutes.

        So this reports the one structural fact that follows: a memory that is
        **full** and mostly **new** has just been replaced. Both conditions
        matter. A deployment still filling its memory is trivially all-new and
        must not read as flooded, which is why capacity is checked first, and a
        full memory in steady state turns over slowly by definition.

        Scale-free on purpose. It is a fraction of this deployment's own
        memory, not a rate threshold, so it makes no claim about how many
        signatures per minute is normal anywhere.
        """
        timestamp = time.time() if now is None else now
        with self._lock:
            total = len(self._profiles)
            if total < self._capacity:
                return 0.0
            cutoff = timestamp - FLOOD_WINDOW
            recent = sum(1 for p in self._profiles.values() if p.first_seen >= cutoff)
        # Ramps in over the upper half: half a memory replaced in five minutes
        # is where this stops being explicable as ordinary discovery.
        return clamp((recent / total - 0.5) / 0.5)

    # -- introspection and exchange ---------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._profiles)

    def __contains__(self, signature: object) -> bool:
        with self._lock:
            return signature in self._profiles

    def __iter__(self) -> Iterator[AttackProfile]:
        with self._lock:
            return iter(list(self._profiles.values()))

    def top(self, n: int = 10, now: float | None = None) -> list[AttackProfile]:
        """The ``n`` most severe profiles right now, strongest first."""
        timestamp = time.time() if now is None else now
        with self._lock:
            return sorted(
                self._profiles.values(),
                key=lambda p: p.effective_severity(timestamp, self._half_life),
                reverse=True,
            )[:n]

    def export_profiles(self) -> list[dict[str, object]]:
        """Serialize the memory for sharing with peer deployments."""
        with self._lock:
            return [profile.to_dict() for profile in self._profiles.values()]

    def import_profiles(self, records: Iterable[dict[str, object]]) -> int:
        """Merge peer profiles in. Returns how many were accepted.

        Merge is max-severity and sum-frequency, so importing the same peer
        feed twice is close to idempotent in severity terms and never
        *weakens* local memory.
        """
        accepted = 0
        with self._lock:
            for record in records:
                signature = str(record.get("signature", ""))
                if not signature:
                    continue
                incoming = AttackProfile(
                    signature=signature,
                    severity=clamp(_as_float(record.get("severity"))),
                    frequency=max(1, _as_int(record.get("frequency"), 1)),
                    success_rate=clamp(_as_float(record.get("success_rate"))),
                    last_seen=_as_float(record.get("last_seen")),
                    first_seen=_as_float(record.get("first_seen")),
                    tags=frozenset(str(t) for t in _as_tags(record.get("tags"))),
                    admission_burst=clamp(_as_float(record.get("admission_burst"))),
                )
                existing = self._profiles.get(signature)
                if existing is None:
                    self._profiles[signature] = incoming
                else:
                    self._profiles[signature] = replace(
                        existing,
                        severity=max(existing.severity, incoming.severity),
                        frequency=existing.frequency + incoming.frequency,
                        last_seen=max(existing.last_seen, incoming.last_seen),
                        first_seen=min(
                            existing.first_seen or incoming.first_seen,
                            incoming.first_seen or existing.first_seen,
                        ),
                        tags=existing.tags | incoming.tags,
                    )
                accepted += 1
            self._evict_if_needed(time.time())
        return accepted
