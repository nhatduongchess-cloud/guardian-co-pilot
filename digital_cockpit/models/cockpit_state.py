"""
cockpit_state.py
================

VERTICAL 2 — DIGITAL COCKPIT
Layer: MODELS (the shared vocabulary of the cockpit)

This file defines the EXACT shape of everything the in-car screen can show:
  - Alert     : one warning/notification (collision, lane departure, ...)
  - Telemetry : live numbers (speed, gear)
  - CockpitState : the WHOLE screen at one instant (driver, theme, alerts, ...)

Vertical 4 (the central brain) pushes updates that change this state, and the
display (a screen or browser) reads CockpitState to draw the UI.

NOTE ON THEME
-------------
Vertical 1 sends a `ui_theme` string ("minimal"/"guided"/"sport"). We define
our OWN CockpitTheme enum with the SAME values. This keeps V2 decoupled from
V1's internals while staying perfectly compatible — a clean cross-vertical
contract based on shared string values, not shared code.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


# ===========================================================================
# SECTION 1 — ENUMS (fixed, safe vocabularies)
# ===========================================================================


class AlertSeverity(str, Enum):
    """
    How urgent an alert is. The display uses this to pick colour/priority,
    and the service uses it to sort alerts (critical shown first).
    """

    INFO = "info"          # neutral information (e.g. "Eco mode on")
    WARNING = "warning"    # caution (e.g. "Lane departure")
    CRITICAL = "critical"  # danger, act now (e.g. "Collision imminent")


class CockpitTheme(str, Enum):
    """Visual theme of the cockpit. Mirrors Vertical 1's ui_theme values."""

    MINIMAL = "minimal"  # clean, few elements — confident drivers
    GUIDED = "guided"    # extra hints & prompts — new drivers
    SPORT = "sport"      # performance-focused layout


class GearPosition(str, Enum):
    """The transmission gear currently selected."""

    PARK = "P"
    REVERSE = "R"
    NEUTRAL = "N"
    DRIVE = "D"


# ===========================================================================
# SECTION 2 — ALERT (one notification on the screen)
# ===========================================================================


class Alert(BaseModel):
    """A single alert/notification currently shown in the cockpit."""

    # extra="forbid" -> reject unknown fields (catches teammate typos early).
    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(
        ...,
        min_length=1,
        description="Stable id so the same alert can be updated/dismissed, "
        "e.g. 'collision_front'.",
    )

    severity: AlertSeverity = Field(
        ...,
        description="Urgency level; drives colour and sort order.",
    )

    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable text to display, e.g. 'Obstacle ahead!'.",
    )

    source: str = Field(
        ...,
        min_length=1,
        description="Which vertical raised it, e.g. 'perception' or "
        "'safety_kernel'. Useful for debugging and demos.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time the alert was raised.",
    )


# ===========================================================================
# SECTION 3 — TELEMETRY (live numeric readouts)
# ===========================================================================


class Telemetry(BaseModel):
    """Live vehicle numbers shown on the instrument cluster."""

    model_config = ConfigDict(extra="forbid")

    speed_kmh: float = Field(
        default=0.0,
        ge=0.0,
        le=400.0,
        description="Current speed in km/h (0..400).",
    )

    gear: GearPosition = Field(
        default=GearPosition.PARK,
        description="Selected transmission gear.",
    )


# ===========================================================================
# SECTION 4 — COCKPIT STATE (the whole screen at one instant)
# ===========================================================================


class CockpitState(BaseModel):
    """
    THE PUBLISHED CONTRACT of Vertical 2.

    This is the complete snapshot the display reads to render the UI, and the
    object the service mutates as V4 pushes updates.
    """

    model_config = ConfigDict(extra="forbid")

    # None until a driver is activated (e.g. car just powered on).
    driver_name: Optional[str] = Field(
        default=None,
        description="Name of the active driver, or None if not set yet.",
    )

    theme: CockpitTheme = Field(
        default=CockpitTheme.MINIMAL,
        description="Visual theme, set from the active driver's ui_theme.",
    )

    telemetry: Telemetry = Field(
        default_factory=Telemetry,
        description="Live vehicle numbers.",
    )

    # default_factory=list gives each state its OWN fresh list (never shared).
    alerts: list[Alert] = Field(
        default_factory=list,
        description="Active alerts, kept sorted by severity (critical first).",
    )

    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time this state last changed.",
    )


# ===========================================================================
# SECTION 5 — SELF-TEST (runs only when executed directly)
# ===========================================================================
if __name__ == "__main__":
    # Start with a default (car just powered on, no driver yet).
    state = CockpitState()
    print("Fresh cockpit state:")
    print(state.model_dump_json(indent=2))

    # Simulate what the service will do: set a driver + push an alert.
    state.driver_name = "Lan (Teen)"
    state.theme = CockpitTheme.GUIDED
    state.telemetry.speed_kmh = 42.0
    state.telemetry.gear = GearPosition.DRIVE
    state.alerts.append(
        Alert(
            alert_id="collision_front",
            severity=AlertSeverity.CRITICAL,
            message="Obstacle ahead - brake!",
            source="safety_kernel",
        )
    )

    print("\nCockpit state after V4 pushes updates:")
    print(state.model_dump_json(indent=2))
