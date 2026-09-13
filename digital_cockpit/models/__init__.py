"""
models package — the shared vocabulary of the Digital Cockpit.

Re-exports let other layers write:
    from models import CockpitState, Alert, AlertSeverity
"""

from models.cockpit_state import (
    Alert,
    AlertSeverity,
    CockpitState,
    CockpitTheme,
    GearPosition,
    Telemetry,
)

__all__ = [
    "Alert",
    "AlertSeverity",
    "CockpitState",
    "CockpitTheme",
    "GearPosition",
    "Telemetry",
]
