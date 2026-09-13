"""
Tests for the learning loop.

The interesting tests here are the ones that try to make the loop UNSAFE:
train it into silence, reward it for forgetting a miss, or walk it out of
bounds. Those are the failures that would matter in a car.
"""

from __future__ import annotations

import pytest

from guardian.learning import (
    DEFAULT_PERCLOS_THRESHOLD,
    DriverProfile,
    LearningLoop,
    SafetyBounds,
    TripOutcome,
    summarise,
)


def noisy(trip_id="T", warnings=10, false_alarms=8, missed=0):
    return TripOutcome(trip_id=trip_id, warnings_raised=warnings,
                       false_alarms=false_alarms, missed_hazards=missed)


# --------------------------------------------------------------------------
# Safety: the properties that must hold no matter what the data says
# --------------------------------------------------------------------------

def test_a_driver_cannot_train_the_car_into_silence():
    """
    50 trips of nothing but dismissed warnings. The threshold must stop at the
    ceiling, not keep drifting toward "never warn".
    """
    loop = LearningLoop()
    profile = DriverProfile(driver_id="d1")
    profile, _ = loop.replay(profile, [noisy(f"T{i}") for i in range(50)])
    assert profile.perclos_threshold == loop.bounds.ceiling
    assert profile.perclos_threshold <= SafetyBounds().ceiling


def test_threshold_never_escapes_bounds_under_any_sequence():
    loop = LearningLoop()
    profile = DriverProfile(driver_id="d1")
    mixed = [noisy("a"), noisy("b", missed=3), noisy("c"), noisy("d", missed=9),
             noisy("e"), noisy("f"), noisy("g", missed=1)]
    for outcome in mixed * 5:
        profile, _ = loop.update(profile, outcome)
        assert loop.bounds.floor <= profile.perclos_threshold <= loop.bounds.ceiling


def test_a_missed_hazard_tightens_immediately():
    loop = LearningLoop()
    profile = DriverProfile(driver_id="d1", perclos_threshold=0.10)
    profile, reason = loop.update(profile, noisy(missed=1))
    assert profile.perclos_threshold < 0.10
    assert "tightened" in reason


def test_a_miss_outranks_false_alarms_in_the_same_trip():
    """
    A trip that was both noisy AND missed something must tighten. Comfort does
    not get a vote when the system was too slow where it counted.
    """
    loop = LearningLoop()
    profile = DriverProfile(driver_id="d1", perclos_threshold=0.10)
    profile, reason = loop.update(profile, noisy(warnings=20, false_alarms=19, missed=1))
    assert profile.perclos_threshold < 0.10, "must tighten despite heavy false alarms"
    assert "unwarned" in reason


def test_tightening_is_faster_than_loosening():
    """One miss must undo more than one noisy trip, or misses get forgotten."""
    loop = LearningLoop()
    start = 0.10
    loosened, _ = loop.adjust(start, noisy())
    tightened, _ = loop.adjust(start, noisy(missed=1))
    assert (start - tightened) > (loosened - start)


def test_a_loop_that_loosens_faster_than_it_tightens_is_rejected():
    with pytest.raises(ValueError):
        LearningLoop(loosen_step=0.05, tighten_step=0.01)


def test_invalid_bounds_are_rejected():
    with pytest.raises(ValueError):
        SafetyBounds(floor=0.2, ceiling=0.1)


# --------------------------------------------------------------------------
# Adaptation behaviour
# --------------------------------------------------------------------------

def test_new_driver_starts_at_the_shipped_default():
    profile = DriverProfile(driver_id="new")
    assert profile.perclos_threshold == DEFAULT_PERCLOS_THRESHOLD
    assert profile.is_personalised is False


def test_well_behaved_warnings_leave_the_threshold_alone():
    loop = LearningLoop()
    profile = DriverProfile(driver_id="d1")
    profile, reason = loop.update(profile, noisy(warnings=10, false_alarms=1))
    assert profile.perclos_threshold == DEFAULT_PERCLOS_THRESHOLD
    assert "earning their place" in reason


def test_at_the_ceiling_the_reason_says_so_rather_than_claiming_a_change():
    loop = LearningLoop()
    profile = DriverProfile(driver_id="d1", perclos_threshold=loop.bounds.ceiling)
    profile, reason = loop.update(profile, noisy())
    assert "ceiling" in reason


def test_update_does_not_mutate_the_original_profile():
    loop = LearningLoop()
    original = DriverProfile(driver_id="d1")
    updated, _ = loop.update(original, noisy())
    assert original.trips_seen == 0 and original.history == ()
    assert updated.trips_seen == 1


def test_every_update_explains_itself():
    loop = LearningLoop()
    profile = DriverProfile(driver_id="d1")
    _, reasons = loop.replay(profile, [noisy("a"), noisy("b", missed=1), noisy("c", false_alarms=0)])
    assert len(reasons) == 3
    assert all(r.strip() for r in reasons)


# --------------------------------------------------------------------------
# The KPI: false-alarm reduction over five trips
# --------------------------------------------------------------------------

def test_false_alarm_reduction_is_none_until_there_is_a_comparison():
    loop = LearningLoop()
    profile = DriverProfile(driver_id="d1")
    assert profile.false_alarm_reduction() is None
    profile, _ = loop.update(profile, noisy())
    assert profile.false_alarm_reduction() is None, "one trip is not a trend"


def test_reduction_is_measured_against_the_first_trip():
    loop = LearningLoop()
    profile = DriverProfile(driver_id="d1")
    trips = [
        noisy("T1", warnings=10, false_alarms=8),   # 0.80
        noisy("T2", warnings=10, false_alarms=6),
        noisy("T3", warnings=10, false_alarms=4),
        noisy("T4", warnings=10, false_alarms=3),
        noisy("T5", warnings=10, false_alarms=2),   # 0.20
    ]
    profile, _ = loop.replay(profile, trips)
    reduction = profile.false_alarm_reduction()
    assert reduction == pytest.approx(0.75), "0.80 -> 0.20 is a 75% reduction"
    assert reduction >= 0.20, "meets the proposal's 20-30% target after five trips"


def test_a_worsening_driver_reports_zero_not_a_negative_improvement():
    loop = LearningLoop()
    profile = DriverProfile(driver_id="d1")
    profile, _ = loop.replay(profile, [
        noisy("T1", warnings=10, false_alarms=2),
        noisy("T2", warnings=10, false_alarms=9),
    ])
    assert profile.false_alarm_reduction() == 0.0


def test_reduction_undefined_when_the_first_trip_had_no_false_alarms():
    loop = LearningLoop()
    profile = DriverProfile(driver_id="d1")
    profile, _ = loop.replay(profile, [
        noisy("T1", warnings=5, false_alarms=0),
        noisy("T2", warnings=5, false_alarms=1),
    ])
    assert profile.false_alarm_reduction() is None


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def test_impossible_outcomes_are_rejected():
    with pytest.raises(ValueError):
        TripOutcome(trip_id="x", warnings_raised=2, false_alarms=5)
    with pytest.raises(ValueError):
        TripOutcome(trip_id="x", missed_hazards=-1)


def test_no_warnings_means_no_false_alarm_rate():
    assert TripOutcome(trip_id="x").false_alarm_rate == 0.0


def test_summary_reports_the_threshold_and_the_kpi():
    loop = LearningLoop()
    profile = DriverProfile(driver_id="d7")
    profile, _ = loop.replay(profile, [
        noisy("T1", warnings=10, false_alarms=8),
        noisy("T2", warnings=10, false_alarms=2),
    ])
    report = summarise(profile)
    assert report["driver_id"] == "d7"
    assert report["trips_seen"] == 2
    assert report["personalised"] is True
    assert report["false_alarm_reduction"] == pytest.approx(0.75)
    assert report["shipped_default"] == DEFAULT_PERCLOS_THRESHOLD
