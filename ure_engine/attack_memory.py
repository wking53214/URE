"""
attack_memory.py — AMX, the Attack Memory Exchange.
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

    __slots__ = ("_capacity", "_half_life", "_lock", "_profiles")

    def __init__(self, capacity: int = 4096, half_life: float = DEFAULT_HALF_LIFE) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
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

    def _evict_if_needed(self, now: float) -> None:
        """Evict the weakest profiles when over capacity. Caller holds the lock."""
        if len(self._profiles) <= self._capacity:
            return
        ranked = sorted(
            self._profiles.values(),
            key=lambda p: p.effective_severity(now, self._half_life),
        )
        for profile in ranked[: len(self._profiles) - self._capacity]:
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
