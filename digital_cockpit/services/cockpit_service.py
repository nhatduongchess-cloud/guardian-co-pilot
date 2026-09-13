"""
cockpit_service.py
==================

VERTICAL 2 — DIGITAL COCKPIT
Layer: SERVICE (the in-memory "brain" of the cockpit)

This service OWNS the single, live CockpitState. Vertical 4 pushes updates
(set driver, update telemetry, raise/dismiss alerts) and the display reads
the current state to draw the UI.

TWO IMPORTANT DESIGN IDEAS
--------------------------
1) THREAD SAFETY
   FastAPI may handle several requests at once (V4 pushing telemetry while the
   display reads state). A threading.Lock ensures only ONE thread mutates the
   state at a time, so it never ends up half-updated.

2) RETURN COPIES, NEVER THE LIVE OBJECT
   Every method returns a DEEP COPY of the state. If we returned the internal
   object, a caller could mutate it directly, bypassing the lock and causing
   subtle bugs. Copies keep the internal state fully under our control.

There is NO repository/database here on purpose: cockpit state is ephemeral
(it resets when the car powers off), so it lives only in RAM.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import threading
from datetime import datetime, timezone
from typing import Optional

from models.cockpit_state import (
    Alert,
    AlertSeverity,
    CockpitState,
    CockpitTheme,
    GearPosition,
    Telemetry,
)


# ===========================================================================
# SECTION 1 — THE SERVICE
# ===========================================================================


class CockpitService:
    """Holds and safely mutates the one live CockpitState."""

    # Sort order for alerts: lower rank = shown higher on screen.
    # critical (0) appears before warning (1) before info (2).
    _SEVERITY_RANK = {
        AlertSeverity.CRITICAL: 0,
        AlertSeverity.WARNING: 1,
        AlertSeverity.INFO: 2,
    }

    def __init__(self) -> None:
        # The single source of truth for what the screen shows.
        self._state = CockpitState()
        # The "door lock" guarding every read/write of _state.
        self._lock = threading.Lock()

    # -- read ----------------------------------------------------------------

    def get_state(self) -> CockpitState:
        """Return a SAFE deep copy of the current cockpit state."""
        with self._lock:
            return self._state.model_copy(deep=True)

    # -- driver / theme ------------------------------------------------------

    def set_driver(
        self,
        driver_name: str,
        theme: CockpitTheme | str = CockpitTheme.MINIMAL,
    ) -> CockpitState:
        """
        Configure the cockpit for the active driver.

        `theme` may be a CockpitTheme OR a raw string like "guided" (which is
        exactly what Vertical 1 sends via `ui_theme`). We coerce a string into
        the enum, which also VALIDATES it (an unknown theme raises ValueError).
        """
        # Coerce/validate a string theme into the enum.
        theme_enum = CockpitTheme(theme) if isinstance(theme, str) else theme

        with self._lock:
            self._state.driver_name = driver_name
            self._state.theme = theme_enum
            self._touch()
            return self._state.model_copy(deep=True)

    # -- telemetry -----------------------------------------------------------

    def update_telemetry(
        self,
        speed_kmh: Optional[float] = None,
        gear: Optional[GearPosition | str] = None,
    ) -> CockpitState:
        """
        Partially update telemetry. Only the fields you pass are changed;
        the rest keep their current values.

        We rebuild a Telemetry object so Pydantic VALIDATES the new values
        (e.g. a negative speed is rejected here, not silently stored).
        """
        with self._lock:
            current = self._state.telemetry
            new_telemetry = Telemetry(
                speed_kmh=current.speed_kmh if speed_kmh is None else speed_kmh,
                gear=current.gear if gear is None else gear,
            )
            self._state.telemetry = new_telemetry
            self._touch()
            return self._state.model_copy(deep=True)

    # -- alerts --------------------------------------------------------------

    def raise_alert(self, alert: Alert) -> CockpitState:
        """
        Add a new alert, OR update an existing one with the same alert_id.

        Using alert_id as the identity means repeatedly reporting the same
        hazard (e.g. an obstacle that stays ahead) updates ONE alert instead
        of spamming duplicates. After the change we re-sort so the most
        severe alert sits at the top.
        """
        with self._lock:
            # Drop any existing alert with the same id (we will re-add it).
            self._state.alerts = [
                a for a in self._state.alerts if a.alert_id != alert.alert_id
            ]
            self._state.alerts.append(alert)
            self._sort_alerts()
            self._touch()
            return self._state.model_copy(deep=True)

    def dismiss_alert(self, alert_id: str) -> bool:
        """
        Remove an alert by id (e.g. the hazard has cleared).

        Returns:
            True if an alert was removed, False if no such id existed.
        """
        with self._lock:
            before = len(self._state.alerts)
            self._state.alerts = [
                a for a in self._state.alerts if a.alert_id != alert_id
            ]
            removed = len(self._state.alerts) < before
            if removed:
                self._touch()
            return removed

    # -- lifecycle -----------------------------------------------------------

    def reset(self) -> CockpitState:
        """
        Reset to a fresh cockpit (e.g. ignition off / driver logs out).
        Clears driver, theme, telemetry, and all alerts.
        """
        with self._lock:
            self._state = CockpitState()
            return self._state.model_copy(deep=True)

    # -- private helpers -----------------------------------------------------

    def _sort_alerts(self) -> None:
        """Sort alerts in place: critical first. Stable (keeps insertion order
        within the same severity)."""
        self._state.alerts.sort(
            key=lambda a: self._SEVERITY_RANK[a.severity]
        )

    def _touch(self) -> None:
        """Stamp the state as just-updated."""
        self._state.last_updated = datetime.now(timezone.utc)


# ===========================================================================
# SECTION 2 — SELF-TEST (runs only when executed directly)
# ===========================================================================
if __name__ == "__main__":
    service = CockpitService()

    # V4 sets the driver using a raw ui_theme string from V1.
    service.set_driver("Lan (Teen)", theme="guided")
    service.update_telemetry(speed_kmh=42.0, gear="D")

    # Alerts arrive out of order; the service must sort by severity.
    service.raise_alert(
        Alert(alert_id="eco_tip", severity="info",
              message="Eco mode active", source="cockpit")
    )
    service.raise_alert(
        Alert(alert_id="collision_front", severity="critical",
              message="Obstacle ahead - brake!", source="safety_kernel")
    )
    service.raise_alert(
        Alert(alert_id="lane", severity="warning",
              message="Lane departure", source="perception")
    )

    state = service.get_state()
    print(f"Driver: {state.driver_name} | Theme: {state.theme.value} | "
          f"Speed: {state.telemetry.speed_kmh} | Gear: {state.telemetry.gear.value}")
    print("Alerts (should be critical -> warning -> info):")
    for a in state.alerts:
        print(f"  [{a.severity.value:8s}] {a.alert_id}: {a.message}")

    # Hazard cleared.
    service.dismiss_alert("collision_front")
    print(f"\nAfter dismissing collision: {len(service.get_state().alerts)} alerts left")
