"""
dwell.py, how long has this been going on, relative to normal.
==============================================================

URE reasons about *change*. Energy, its derivative, velocity, acceleration,
volatility: every one is a statement about motion, and every one reads zero for
a quantity that is simply held constant.

That is a real hole. Hold a hostile rate steady just under the boundary where
affinity alone would name the regime ATTACKED, and there is nothing left to
detect: the level is too low, and the derivative of a constant is zero. Red
teaming measured the edge at 36.2% sustained hostile pressure, below which
evasion was indefinite, at any duration.

The missing quantity is exposure, so this module integrates instead of
differentiating. A hostile rate held for two observations is noise; the same
rate held for eighty is a campaign, and no amount of differentiating tells
those apart.

WHY THIS IS NOT A FIXED THRESHOLD, AND WHY THAT MATTERS MORE THAN THE ATTACK
----------------------------------------------------------------------------
The first version of this module charged on hostile pressure above a fixed
floor of 0.12. That floor is a claim about every deployment that will ever run
URE: that sustained hostile pressure above 12% means an adversary.

It does not. Plenty of healthy gateways sit permanently above it. Rate limits
reject a steady fraction of traffic. One misconfigured client retries forever.
A crawler trips a rule on every request and is not attacking anybody. Measured
against the fixed floor, a deployment whose normal rejection rate is 25% was
classified ATTACKED after 25 observations, with a recommended action of
ISOLATE, with no adversary anywhere near it.

That is worse than the attack it fixed. A missed slow boil costs you the
attacker's throughput. A false ISOLATE is an outage you inflicted on yourself,
caused by the defence, at a moment when nothing was wrong. A resilience engine
that does that gets switched off, and then it is not defending against the slow
boil either.

So there is no fixed floor here. This module learns the hostile pressure this
particular deployment shows when nobody is attacking it, and charges only on
the excess over that. A system's own steady state cannot be anomalous relative
to itself, so the steady-state false positive is gone by construction rather
than by choosing a better constant. There is no better constant: the quantity
is genuinely deployment-specific, and any single number is wrong somewhere.

THE BASELINE CEILING, AND WHY THIS IS NEVER A REGRESSION
--------------------------------------------------------
A learned baseline has an obvious failure: an engine started up *during* an
attack learns the attack as normal. So the baseline is capped at
``baseline_ceiling``, above which no deployment gets to call its own traffic
quiet.

The cap plus the margin is placed at 0.36, which is exactly where URE detected
on level alone before any of this existed. The guarantee that follows is worth
stating exactly:

* above 0.36 hostile pressure, detection is immediate, exactly as before, and
  no amount of patience moves that wall;
* between the learned baseline and 0.36, detection now happens over time,
  where before it never happened at all;
* below the learned baseline, nothing fires, and that band is by definition
  "what this deployment does when nobody is attacking it".

So this is never worse than the pre-dwell engine on either axis, and strictly
better on detection. No false positive is introduced anywhere, and none of the
detection the original engine had is traded away to buy that.

The cold-start hole self-heals: the baseline follows a quieter floor *down*
quickly, so the first genuine lull re-calibrates it, and a resumption then
reads as excess.

WALKING THE BASELINE UP
-----------------------
Any adaptive baseline can in principle be walked: ramp slowly enough and the
baseline absorbs it. The defence is an exchange rate, not a wall.

* The baseline rises far more slowly than it falls, so an adversary who
  alternates pressure with quiet loses ground faster than they gain it.
* It rises slower still once dwell is already elevated, so a walk cannot
  continue through the evidence it is generating.
* It cannot exceed ``baseline_ceiling`` at all, and that cap is placed so
  that reaching it wins the adversary nothing the original engine allowed.

The measured cost of a successful walk is in
``tests/test_adversarial.py::test_walking_the_baseline_upward_is_slow``. That
test exists to keep the exchange rate honest if these constants are ever
retuned.

DURATION, NOT MAGNITUDE
-----------------------
Above the margin the accumulator charges toward 1.0 at a fixed rate regardless
of how far above it the pressure sits. Magnitude is already fully represented:
it drives the pressure vector, the energy, and the affinity functions.
Weighting dwell by magnitude would double-count what URE already sees well, and
would leave the low-and-patient adversary, the only case this exists for,
accumulating most slowly.
"""

from __future__ import annotations

import math

from .state_vector import StateVector, clamp

__all__ = ["HostileDwell"]

#: Default hardest hostile pressure a deployment may learn as its own normal.
#:
#: Chosen so that ceiling + margin lands on 0.36, the level where URE detected
#: on affinity alone before this module existed. That makes the guarantee exact
#: rather than approximate: an adversary who spends tens of thousands of
#: observations walking the baseline to its cap arrives at the boundary the
#: original engine already enforced, and gains nothing at all for the patience.
#: No detection is traded away to buy the false-positive fix.
#:
#: Raising it buys tolerance for a deployment whose genuine normal rejection
#: rate is high, at the cost of letting a very patient adversary sit that much
#: higher. That is a real deployment-specific tradeoff, which is why it is a
#: parameter and not a constant: hardcoding the right answer for every
#: deployment is the exact mistake the fixed floor made.
_DEFAULT_BASELINE_CEILING: float = 0.26

#: Excess over the learned baseline before anything accumulates. Absorbs the
#: ordinary variance of a healthy system without needing to model it.
_MARGIN: float = 0.10

#: Width of the soft edge above the margin. A hard cutoff would make behaviour
#: at the boundary turn on the last decimal place of the smoother.
_BAND: float = 0.10

#: Observations spent learning the baseline before anything can accumulate. An
#: engine that alarms during its own warmup has not observed anything yet.
_WARMUP: int = 32

#: Baseline time constant when the observed pressure is BELOW it. Fast, because
#: a quieter floor is good news and should be believed.
_TAU_DOWN: float = 150.0

#: Baseline time constant when the observed pressure is ABOVE it. Slow, because
#: rising pressure is exactly what an adversary would want it to believe.
_TAU_UP: float = 600.0

#: And slower still once dwell is elevated, so a walk-up cannot proceed through
#: the evidence it is creating. Not infinite: a deployment that has genuinely
#: changed must be able to re-calibrate without an operator.
_TAU_UP_SUSPICIOUS: float = 6000.0

#: Dwell above which the baseline is treated as under attack.
_LOCK_AT: float = 0.25

#: Observations of continuous excess to reach ~63% charge.
_RISE: float = 25.0

#: Observations of quiet to decay by the same factor. Longer than the rise on
#: purpose: evidence should accumulate faster than it is forgiven, or an
#: adversary alternating pressure with quiet pays nothing.
_FALL: float = 60.0

#: Guard against a pathological or replayed timestamp saturating or flushing
#: the accumulator in a single observation.
_MAX_STEP: float = 300.0


class HostileDwell:
    """Leaky integrator over time spent above this deployment's own normal.

    Reports a scalar on [0, 1]: 0.0 means no sustained excess hostile exposure,
    1.0 means several multiples of ``_RISE`` spent under it.
    """

    __slots__ = ("_baseline", "_ceiling", "_last_timestamp", "_observations", "_value")

    def __init__(self, baseline_ceiling: float = _DEFAULT_BASELINE_CEILING) -> None:
        if not 0.0 <= baseline_ceiling < 1.0:
            raise ValueError("baseline_ceiling must be in [0, 1)")
        self._ceiling = baseline_ceiling
        self._baseline = 0.0
        self._value = 0.0
        self._observations = 0
        self._last_timestamp: float | None = None

    @property
    def value(self) -> float:
        """Current dwell, without advancing the accumulator."""
        return self._value

    @property
    def baseline(self) -> float:
        """Hostile pressure currently believed normal for this deployment.

        Exposed because an operator who sees an ATTACKED call needs to know
        what it was judged against, and because a baseline sitting at the
        ceiling is itself worth looking at.
        """
        return self._baseline

    @property
    def calibrated(self) -> bool:
        """False until enough observations exist to have a baseline at all."""
        return self._observations >= _WARMUP

    @staticmethod
    def _smoothstep(x: float) -> float:
        """Continuous at both ends, so there is no sharp corner to sit on."""
        x = clamp(x)
        return x * x * (3.0 - 2.0 * x)

    def _advance_baseline(self, hostile: float, dt: float) -> None:
        if self._observations <= _WARMUP:
            # Acquisition. A plain running mean, because the slow asymmetric
            # tracking below would leave a deployment with a genuinely high
            # normal rate sitting at a baseline of ~0 for its first several
            # hours, alarming the whole time. That is the same false positive
            # this module exists to remove, just moved to startup.
            n = self._observations
            self._baseline += (hostile - self._baseline) / max(1, n)
        else:
            if hostile <= self._baseline:
                tau = _TAU_DOWN
            elif self._value >= _LOCK_AT:
                tau = _TAU_UP_SUSPICIOUS
            else:
                tau = _TAU_UP
            self._baseline += (hostile - self._baseline) * (1.0 - math.exp(-dt / tau))
        self._baseline = min(self._ceiling, clamp(self._baseline))

    def update(self, state: StateVector, timestamp: float) -> float:
        """Advance the accumulator by one observation and return the new dwell."""
        hostile = max(state.threat_pressure, state.adversarial_pressure)
        if not math.isfinite(hostile) or not math.isfinite(timestamp):
            # A broken reading must not be able to charge the accumulator, move
            # the baseline, or flush either. Hold, and let the caller's own
            # non-finite handling be the thing that speaks.
            return self._value

        previous = self._last_timestamp
        self._last_timestamp = timestamp
        self._observations += 1
        # The first observation establishes the clock only. Charging from a
        # single sample would let a cold-started engine inherit exposure it
        # never actually saw.
        dt = 1.0 if previous is None else min(_MAX_STEP, max(0.0, timestamp - previous))

        self._advance_baseline(hostile, dt)

        if not self.calibrated:
            # Nothing is known yet about what normal looks like here, so there
            # is nothing to be in excess of.
            return self._value

        excess = hostile - self._baseline - _MARGIN
        target = self._smoothstep(excess / _BAND)
        tau = _RISE if target > self._value else _FALL
        self._value = clamp(self._value + (target - self._value) * (1.0 - math.exp(-dt / tau)))
        return self._value

    def reset(self) -> None:
        """Forget the baseline and the accumulator.

        Worth doing deliberately after an incident: if the engine was started
        during an attack, the baseline it learned is the attack.
        """
        self._baseline = 0.0
        self._value = 0.0
        self._observations = 0
        self._last_timestamp = None
