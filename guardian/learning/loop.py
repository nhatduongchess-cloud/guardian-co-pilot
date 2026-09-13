"""
loop.py
=======

THE LEARNING LOOP: THE CAR GETS TO KNOW *THIS* DRIVER.

One threshold for everyone is the second of the three gaps this project is
about. A novice and a twenty-year veteran get identical warnings; the veteran
learns the warnings are noise and switches them off; the feature's safety value
goes to zero. The fix is not a better global number - there isn't one - it is a
threshold that moves toward the person actually driving.

HOW IT ADAPTS
-------------
After each trip the loop sees what happened and nudges the driver's drowsiness
threshold:

  * warnings the driver dismissed, with no hazard behind them -> LOOSEN a little
  * a hazard the system failed to warn about               -> TIGHTEN a lot

That asymmetry is the whole safety argument. Loosening is a comfort improvement
and is made slowly, in small steps, only on repeated evidence. Tightening is a
safety correction and is made immediately, in one step. A system that forgets
a miss as easily as it forgets a false alarm is not a safety system.

WHAT IT MAY NEVER DO
--------------------
Personalisation is bounded. `SafetyBounds` is a hard floor and ceiling that no
amount of learning can cross, so a driver who dismisses every warning for a
month still cannot train the car into silence. This mirrors the rule the rest
of the architecture already follows - the planner proposes, the safety kernel
disposes - and it is enforced here, in `_clamp`, on every single update rather
than checked once at the end.

The loop tunes WHEN to warn. It has no route to the brake command itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

#: Shipped defaults, from the tuned rule engine. A driver with no history starts
#: exactly where the global engine starts - personalisation is a refinement of a
#: safe default, never a replacement for one.
DEFAULT_PERCLOS_THRESHOLD = 0.08


@dataclass(frozen=True)
class SafetyBounds:
    """
    The range a learned threshold may occupy. Not advisory.

    `ceiling` is the important one: it is the point past which the system would
    be too slow to be useful, and no sequence of dismissed warnings may push a
    driver's threshold above it.
    """

    floor: float = 0.04
    ceiling: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 < self.floor < self.ceiling:
            raise ValueError(f"invalid bounds: floor={self.floor} ceiling={self.ceiling}")


@dataclass(frozen=True)
class TripOutcome:
    """
    What one completed trip revealed. Counts, not opinions.

    `false_alarms` are warnings the driver dismissed with no hazard behind them.
    `missed_hazards` are hazards that occurred with no warning raised - the
    expensive kind of error, and the reason tightening is asymmetric.
    """

    trip_id: str
    warnings_raised: int = 0
    false_alarms: int = 0
    missed_hazards: int = 0
    #: This driver's own resting eye-closure on this trip, when measured.
    baseline_perclos: float | None = None

    def __post_init__(self) -> None:
        for name in ("warnings_raised", "false_alarms", "missed_hazards"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.false_alarms > self.warnings_raised:
            raise ValueError("false_alarms cannot exceed warnings_raised")

    @property
    def false_alarm_rate(self) -> float:
        """Share of warnings that were noise. No warnings means no noise."""
        return self.false_alarms / self.warnings_raised if self.warnings_raised else 0.0


@dataclass(frozen=True)
class DriverProfile:
    """
    What Guardian has learned about one driver. Immutable: every update returns
    a new profile, so a trip can never be half-applied.
    """

    driver_id: str
    perclos_threshold: float = DEFAULT_PERCLOS_THRESHOLD
    trips_seen: int = 0
    history: tuple[TripOutcome, ...] = field(default_factory=tuple)

    @property
    def is_personalised(self) -> bool:
        return self.trips_seen > 0

    @property
    def first_false_alarm_rate(self) -> float | None:
        return self.history[0].false_alarm_rate if self.history else None

    @property
    def latest_false_alarm_rate(self) -> float | None:
        return self.history[-1].false_alarm_rate if self.history else None

    def false_alarm_reduction(self) -> float | None:
        """
        Relative drop in false-alarm rate since the first trip, 0..1.

        This is the number the proposal puts a KPI on (20-30% after five trips).
        Returns None while there is nothing to compare against, rather than a
        flattering zero.
        """
        if len(self.history) < 2:
            return None
        first = self.first_false_alarm_rate or 0.0
        latest = self.latest_false_alarm_rate or 0.0
        if first <= 0.0:
            return None  # cannot improve on a trip that had no false alarms
        return max(0.0, (first - latest) / first)


class LearningLoop:
    """
    Turns trip outcomes into a personalised threshold.

    Deliberately not a learned model. With a handful of trips per driver there
    is nothing to fit, and the same trap that sank the CNN - memorising a tiny
    sample - applies here. This is a bounded controller: small corrections,
    hard limits, and every step explainable in one sentence.
    """

    def __init__(
        self,
        bounds: SafetyBounds | None = None,
        loosen_step: float = 0.005,
        tighten_step: float = 0.02,
        false_alarm_trigger: float = 0.30,
    ) -> None:
        self.bounds = bounds or SafetyBounds()
        #: Small, because loosening trades away sensitivity.
        self.loosen_step = loosen_step
        #: Four times larger, because a miss is the error that hurts.
        self.tighten_step = tighten_step
        #: Below this, the warnings are considered to be earning their place.
        self.false_alarm_trigger = false_alarm_trigger

        if tighten_step < loosen_step:
            raise ValueError("tightening must never be slower than loosening")

    def _clamp(self, value: float) -> float:
        return max(self.bounds.floor, min(self.bounds.ceiling, value))

    def adjust(self, threshold: float, outcome: TripOutcome) -> tuple[float, str]:
        """
        One trip's correction, with the reason for it.

        Returns the new threshold and a sentence explaining the change, because
        a personalisation that cannot be explained is a personalisation nobody
        will sign off.
        """
        # A miss outranks everything else in the trip. Even if the same trip was
        # full of false alarms, the system was not sensitive enough where it
        # counted, and comfort does not get a vote here.
        if outcome.missed_hazards > 0:
            new = self._clamp(threshold - self.tighten_step)
            return new, (
                f"tightened {threshold:.3f} -> {new:.3f}: "
                f"{outcome.missed_hazards} hazard(s) went unwarned"
            )

        if outcome.false_alarm_rate > self.false_alarm_trigger:
            new = self._clamp(threshold + self.loosen_step)
            if new == threshold:
                return new, (
                    f"held at {threshold:.3f}: false-alarm rate "
                    f"{outcome.false_alarm_rate:.0%} but the safety ceiling is reached"
                )
            return new, (
                f"loosened {threshold:.3f} -> {new:.3f}: false-alarm rate "
                f"{outcome.false_alarm_rate:.0%} with no missed hazards"
            )

        return threshold, f"held at {threshold:.3f}: warnings are earning their place"

    def update(self, profile: DriverProfile, outcome: TripOutcome) -> tuple[DriverProfile, str]:
        """Apply one trip to a profile, returning the new profile and the reason."""
        new_threshold, reason = self.adjust(profile.perclos_threshold, outcome)
        updated = replace(
            profile,
            perclos_threshold=new_threshold,
            trips_seen=profile.trips_seen + 1,
            history=profile.history + (outcome,),
        )
        return updated, reason

    def replay(
        self, profile: DriverProfile, outcomes: Iterable[TripOutcome]
    ) -> tuple[DriverProfile, list[str]]:
        """Apply a sequence of trips in order. Useful for evaluating the KPI."""
        reasons: list[str] = []
        for outcome in outcomes:
            profile, reason = self.update(profile, outcome)
            reasons.append(f"[{outcome.trip_id}] {reason}")
        return profile, reasons


def summarise(profile: DriverProfile) -> dict:
    """A small report: where the threshold landed and whether the KPI was met."""
    reduction = profile.false_alarm_reduction()
    return {
        "driver_id": profile.driver_id,
        "trips_seen": profile.trips_seen,
        "perclos_threshold": round(profile.perclos_threshold, 4),
        "shipped_default": DEFAULT_PERCLOS_THRESHOLD,
        "personalised": profile.is_personalised,
        "first_false_alarm_rate": (
            round(profile.first_false_alarm_rate, 4)
            if profile.first_false_alarm_rate is not None else None
        ),
        "latest_false_alarm_rate": (
            round(profile.latest_false_alarm_rate, 4)
            if profile.latest_false_alarm_rate is not None else None
        ),
        "false_alarm_reduction": round(reduction, 4) if reduction is not None else None,
    }
