"""
test_cockpit_state.py
=====================

Unit tests for VERTICAL 2 -> MODELS layer (cockpit_state.py).

WHY THIS FILE EXISTS
--------------------
These prove the cockpit vocabulary is safe: good data is accepted, bad data
is rejected, and objects survive the JSON trip they will make between V4 and
the display.

HOW TO RUN (from digital_cockpit/):
    pip install pydantic pytest
    pytest -v
"""

import pytest
from pydantic import ValidationError

from models.cockpit_state import (
    Alert,
    AlertSeverity,
    CockpitState,
    CockpitTheme,
    GearPosition,
    Telemetry,
)


# ===========================================================================
# GROUP 1 — DEFAULTS: a fresh cockpit (car just powered on)
# ===========================================================================


def test_fresh_state_has_safe_defaults():
    state = CockpitState()

    assert state.driver_name is None          # no driver yet
    assert state.theme == CockpitTheme.MINIMAL
    assert state.telemetry.speed_kmh == 0.0
    assert state.telemetry.gear == GearPosition.PARK
    assert state.alerts == []                  # no alerts yet


# ===========================================================================
# GROUP 2 — ALERT: required fields + enum safety
# ===========================================================================


def test_alert_requires_all_core_fields():
    """alert_id, severity, message, source are all required."""
    with pytest.raises(ValidationError):
        Alert(severity="critical", message="x", source="y")  # no alert_id


def test_alert_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        Alert(
            alert_id="a",
            severity="deadly",  # not a valid AlertSeverity
            message="x",
            source="y",
        )


def test_alert_rejects_empty_message():
    with pytest.raises(ValidationError):
        Alert(alert_id="a", severity="info", message="", source="y")


def test_alert_valid_construction_and_timestamp():
    alert = Alert(
        alert_id="collision_front",
        severity="critical",
        message="Obstacle ahead",
        source="safety_kernel",
    )
    assert alert.severity == AlertSeverity.CRITICAL
    # created_at is auto-stamped and timezone-aware.
    assert alert.created_at.tzinfo is not None


# ===========================================================================
# GROUP 3 — TELEMETRY: numeric range + gear enum
# ===========================================================================


def test_speed_cannot_be_negative():
    with pytest.raises(ValidationError):
        Telemetry(speed_kmh=-5.0)


def test_speed_above_max_rejected():
    with pytest.raises(ValidationError):
        Telemetry(speed_kmh=500.0)  # above 400 ceiling


def test_invalid_gear_rejected():
    with pytest.raises(ValidationError):
        Telemetry(gear="X")  # not P/R/N/D


def test_valid_telemetry():
    t = Telemetry(speed_kmh=42.5, gear="D")
    assert t.speed_kmh == 42.5
    assert t.gear == GearPosition.DRIVE


# ===========================================================================
# GROUP 4 — EXTRA FIELDS rejected (typo protection)
# ===========================================================================


def test_unknown_field_on_state_rejected():
    with pytest.raises(ValidationError):
        CockpitState(driverName="x")  # wrong key (should be driver_name)


# ===========================================================================
# GROUP 5 — ISOLATION: each state gets its OWN alerts list
# ===========================================================================


def test_alert_lists_are_not_shared():
    a = CockpitState()
    b = CockpitState()

    a.alerts.append(
        Alert(alert_id="x", severity="info", message="m", source="s")
    )

    # b must remain empty — lists are not shared between instances.
    assert b.alerts == []


# ===========================================================================
# GROUP 6 — JSON ROUND-TRIP (travels between V4 and the display)
# ===========================================================================


def test_state_json_round_trip():
    original = CockpitState(
        driver_name="Lan",
        theme=CockpitTheme.GUIDED,
        telemetry=Telemetry(speed_kmh=30.0, gear="D"),
        alerts=[
            Alert(
                alert_id="lane_departure",
                severity="warning",
                message="Drifting left",
                source="perception",
            )
        ],
    )

    as_json = original.model_dump_json()
    rebuilt = CockpitState.model_validate_json(as_json)

    assert rebuilt == original
