"""
test_state_machine.py
=====================

Unit tests for VERTICAL 1 -> MEMORY -> state_machine.py.

WHY THIS FILE IS WRITTEN FIRST (spec §6)
----------------------------------------
The state machine is the deterministic heart of personalisation. Every other
layer trusts its math: the transition COLD_START -> WARMING -> PERSONALIZED,
the EMA baseline, and — most importantly — the safety-floor clamp. A bug here
is a SAFETY bug, so we pin every branch before building anything on top.

HOW TO RUN (from personalization_engine/):
    pytest tests/test_state_machine.py -v
"""

import math

import pytest

from memory.state_machine import (
    GLOBAL_DEFAULTS,
    PersonalizationStateMachine as SM,
)
from models.driver_memory import DriverMemory, PersonalizationState
from models.telemetry import EventType, TelemetryEvent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fatigue_event(trip_id: str, perclos: float, driver="driver_001") -> TelemetryEvent:
    return TelemetryEvent(
        trip_id=trip_id,
        driver_id=driver,
        source="V3_PERCEPTION",
        event_type="FATIGUE_ALERT",
        risk_embedding=[0.1, 0.2, 0.3],
        scalar_features={"perclos": perclos},
    )


def _run_trips(perclos_per_trip: list[float]) -> DriverMemory:
    """Apply one FATIGUE event per trip and return the final memory."""
    mem = DriverMemory.new("driver_001")
    for i, p in enumerate(perclos_per_trip, start=1):
        trip = f"trip_{i:03d}"
        mem = SM.apply_completed_trip(mem, [_fatigue_event(trip, p)], trip)
    return mem


# ===========================================================================
# GROUP 1 — STATE TRANSITIONS (Constraint #2: personalise only after >=3 trips)
# ===========================================================================


def test_state_for_mapping():
    assert SM.state_for(0) == PersonalizationState.COLD_START
    assert SM.state_for(1) == PersonalizationState.COLD_START
    assert SM.state_for(2) == PersonalizationState.WARMING
    assert SM.state_for(3) == PersonalizationState.PERSONALIZED
    assert SM.state_for(10) == PersonalizationState.PERSONALIZED


def test_full_transition_cold_to_personalized():
    mem = DriverMemory.new("driver_001")
    assert mem.state == PersonalizationState.COLD_START

    mem = SM.apply_completed_trip(mem, [_fatigue_event("t1", 0.30)], "t1")
    assert mem.state == PersonalizationState.COLD_START  # 1 trip

    mem = SM.apply_completed_trip(mem, [_fatigue_event("t2", 0.32)], "t2")
    assert mem.state == PersonalizationState.WARMING     # 2 trips

    mem = SM.apply_completed_trip(mem, [_fatigue_event("t3", 0.34)], "t3")
    assert mem.state == PersonalizationState.PERSONALIZED  # 3 trips
    assert mem.trip_count == 3


def test_apply_completed_trip_does_not_mutate_input():
    mem = DriverMemory.new("driver_001")
    updated = SM.apply_completed_trip(mem, [_fatigue_event("t1", 0.4)], "t1")
    # The original is untouched; only the returned copy advanced.
    assert mem.trip_count == 0
    assert updated.trip_count == 1


def test_each_trip_appends_one_memory_snippet():
    mem = _run_trips([0.30, 0.32, 0.34])
    assert len(mem.memory_snippets) == 3
    assert mem.memory_snippets[0].trip_id == "trip_001"
    assert "Chuyến" in mem.memory_snippets[0].summary_vi


# ===========================================================================
# GROUP 2 — THE EMA BASELINE + Z-SCORE
# ===========================================================================


def test_first_sample_seeds_ema():
    mem = SM.apply_completed_trip(DriverMemory.new("d"), [_fatigue_event("t1", 0.40)], "t1")
    # The very first PERCLOS seeds the EMA exactly (no crawl-up from zero).
    assert mem.baseline.fatigue_ema == pytest.approx(0.40)
    assert mem.baseline.fatigue_var == pytest.approx(0.0)


def test_ema_tracks_toward_new_samples():
    mem = _run_trips([0.30, 0.50, 0.50])
    # EMA should sit between the first and later samples, and variance > 0.
    assert 0.30 < mem.baseline.fatigue_ema < 0.50
    assert mem.baseline.fatigue_var > 0.0


def test_zscore_zero_when_no_spread():
    mem = SM.apply_completed_trip(DriverMemory.new("d"), [_fatigue_event("t1", 0.4)], "t1")
    # Only one sample -> variance 0 -> z-score must be a safe 0.0, not a NaN.
    z = SM.fatigue_zscore(mem, 0.9)
    assert z == 0.0


def test_zscore_positive_for_high_value():
    mem = _run_trips([0.30, 0.34, 0.31, 0.33])
    z = SM.fatigue_zscore(mem, 0.60)  # well above this driver's ~0.32 normal
    assert z > 1.0


# ===========================================================================
# GROUP 3 — THRESHOLD SUGGESTION + BLEND WEIGHTS
# ===========================================================================


def test_blend_weights_grow_with_state():
    assert SM.blend_weight(PersonalizationState.COLD_START) == 0.0
    assert SM.blend_weight(PersonalizationState.WARMING) == 0.5
    assert SM.blend_weight(PersonalizationState.PERSONALIZED) == pytest.approx(0.85)


def test_cold_start_returns_global_default():
    mem = DriverMemory.new("driver_001")  # 0 trips, COLD_START
    thr, conf, used_global, floored = SM.suggest_threshold(mem, EventType.FATIGUE_ALERT)
    assert used_global is True
    assert thr == pytest.approx(GLOBAL_DEFAULTS[EventType.FATIGUE_ALERT])


def test_personalized_moves_away_from_global():
    # A driver whose normal PERCLOS is low should get a threshold BELOW the
    # global default (warn earlier), because the personal baseline is low.
    mem = _run_trips([0.30, 0.31, 0.30, 0.32])
    thr, conf, used_global, floored = SM.suggest_threshold(mem, EventType.FATIGUE_ALERT)
    assert used_global is False
    assert thr < GLOBAL_DEFAULTS[EventType.FATIGUE_ALERT]
    assert conf > 0.0


def test_confidence_grows_with_trips():
    assert SM.confidence_for(0) < SM.confidence_for(2) < SM.confidence_for(5)
    assert 0.0 <= SM.confidence_for(0) <= 1.0
    assert SM.confidence_for(100) == 1.0


# ===========================================================================
# GROUP 4 — SAFETY FLOOR (the most important guard)
# ===========================================================================


def test_fatigue_threshold_capped_by_safety_floor():
    # Force a driver baseline so high the raw suggestion exceeds the max fatigue
    # threshold; the clamp MUST cap it and report that it did.
    mem = _run_trips([0.80, 0.82, 0.84, 0.83])
    mem.state = PersonalizationState.PERSONALIZED
    thr, conf, used_global, floored = SM.suggest_threshold(mem, EventType.FATIGUE_ALERT)
    assert thr <= mem.safety_floor.max_fatigue_threshold
    if thr == mem.safety_floor.max_fatigue_threshold:
        assert floored is True


def test_object_risk_ttc_floored_from_below():
    # OBJECT_RISK is not personalised (we hold no TTC baseline) but the global
    # default (2.0) is above the min TTC floor (1.2), so it passes through.
    mem = _run_trips([0.30, 0.31, 0.32])
    thr, conf, used_global, floored = SM.suggest_threshold(mem, EventType.OBJECT_RISK)
    assert thr >= mem.safety_floor.min_ttc_threshold
    assert used_global is True


def test_object_risk_clamped_up_when_floor_raised():
    mem = _run_trips([0.30, 0.31, 0.32])
    mem.safety_floor.min_ttc_threshold = 3.0  # V4 raises the floor above default
    thr, conf, used_global, floored = SM.suggest_threshold(mem, EventType.OBJECT_RISK)
    assert thr == pytest.approx(3.0)
    assert floored is True


# ===========================================================================
# GROUP 5 — CONSENT (opt-out forces global defaults)
# ===========================================================================


def test_opt_out_forces_global_default():
    mem = _run_trips([0.30, 0.31, 0.30, 0.32])  # PERSONALIZED
    mem.consent.personalization_enabled = False
    thr, conf, used_global, floored = SM.suggest_threshold(mem, EventType.FATIGUE_ALERT)
    assert used_global is True
    assert conf == 0.0
    assert thr == pytest.approx(GLOBAL_DEFAULTS[EventType.FATIGUE_ALERT])
