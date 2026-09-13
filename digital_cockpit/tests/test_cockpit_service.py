"""
test_cockpit_service.py
=======================

Unit tests for VERTICAL 2 -> SERVICE layer (cockpit_service.py).

WHY THIS FILE EXISTS
--------------------
The service holds the live display state and the safety-priority sorting of
alerts. We test every mutation, the alert upsert/sort logic, telemetry
validation, and — importantly — that get_state() returns an ISOLATED copy.

HOW TO RUN (from digital_cockpit/):
    pytest -v
"""

import pytest
from pydantic import ValidationError

from models.cockpit_state import Alert, AlertSeverity, CockpitTheme, GearPosition
from services.cockpit_service import CockpitService


def _alert(alert_id: str, severity: str, message: str = "m", source: str = "s") -> Alert:
    """Small helper to build an Alert quickly in tests."""
    return Alert(alert_id=alert_id, severity=severity, message=message, source=source)


# ===========================================================================
# GROUP 1 — DRIVER + THEME
# ===========================================================================


def test_set_driver_sets_name_and_theme():
    service = CockpitService()
    state = service.set_driver("Lan", theme="guided")

    assert state.driver_name == "Lan"
    assert state.theme == CockpitTheme.GUIDED


def test_set_driver_accepts_enum_theme():
    service = CockpitService()
    state = service.set_driver("Minh", theme=CockpitTheme.SPORT)
    assert state.theme == CockpitTheme.SPORT


def test_set_driver_rejects_invalid_theme_string():
    """A bad ui_theme string from V1 must be rejected, not stored."""
    service = CockpitService()
    with pytest.raises(ValueError):
        service.set_driver("X", theme="rainbow")


# ===========================================================================
# GROUP 2 — TELEMETRY (partial update + validation)
# ===========================================================================


def test_update_telemetry_partial_keeps_other_fields():
    service = CockpitService()
    service.update_telemetry(speed_kmh=30.0, gear="D")
    # Update only speed; gear must stay "D".
    state = service.update_telemetry(speed_kmh=55.0)

    assert state.telemetry.speed_kmh == 55.0
    assert state.telemetry.gear == GearPosition.DRIVE


def test_update_telemetry_rejects_negative_speed():
    service = CockpitService()
    with pytest.raises(ValidationError):
        service.update_telemetry(speed_kmh=-10.0)


# ===========================================================================
# GROUP 3 — ALERTS: add, upsert, sort by severity
# ===========================================================================


def test_alerts_sorted_critical_first():
    service = CockpitService()
    # Add in mixed order.
    service.raise_alert(_alert("info1", "info"))
    service.raise_alert(_alert("crit1", "critical"))
    service.raise_alert(_alert("warn1", "warning"))

    severities = [a.severity for a in service.get_state().alerts]
    assert severities == [
        AlertSeverity.CRITICAL,
        AlertSeverity.WARNING,
        AlertSeverity.INFO,
    ]


def test_same_alert_id_is_updated_not_duplicated():
    service = CockpitService()
    service.raise_alert(_alert("collision_front", "warning", message="far"))
    service.raise_alert(_alert("collision_front", "critical", message="close"))

    alerts = service.get_state().alerts
    assert len(alerts) == 1                      # not duplicated
    assert alerts[0].severity == AlertSeverity.CRITICAL
    assert alerts[0].message == "close"          # updated content


def test_dismiss_alert_removes_it():
    service = CockpitService()
    service.raise_alert(_alert("lane", "warning"))

    removed = service.dismiss_alert("lane")
    assert removed is True
    assert service.get_state().alerts == []


def test_dismiss_missing_alert_returns_false():
    service = CockpitService()
    assert service.dismiss_alert("ghost") is False


# ===========================================================================
# GROUP 4 — RESET
# ===========================================================================


def test_reset_clears_everything():
    service = CockpitService()
    service.set_driver("Lan", theme="guided")
    service.raise_alert(_alert("x", "critical"))

    state = service.reset()
    assert state.driver_name is None
    assert state.theme == CockpitTheme.MINIMAL
    assert state.alerts == []


# ===========================================================================
# GROUP 5 — ISOLATION: get_state() returns a copy, not the live object
# ===========================================================================


def test_get_state_returns_isolated_copy():
    service = CockpitService()
    service.raise_alert(_alert("x", "info"))

    # Mutate the returned copy aggressively.
    snapshot = service.get_state()
    snapshot.alerts.clear()
    snapshot.driver_name = "HACKED"

    # The service's internal state must be untouched.
    fresh = service.get_state()
    assert len(fresh.alerts) == 1
    assert fresh.driver_name is None
