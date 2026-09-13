"""
routes.py
=========

VERTICAL 2 — DIGITAL COCKPIT
Layer: API (the HTTP door of the cockpit)

Vertical 4 (the brain) PUSHES updates here; the display READS state here.

CRITICAL DESIGN POINT — A SHARED SINGLETON SERVICE
--------------------------------------------------
Unlike Vertical 1 (stateless -> a new service per request is fine), the
cockpit service holds LIVE state in RAM. Every request MUST share the SAME
service instance, or updates and reads would hit different (empty) states.

So we create ONE module-level CockpitService and return it from the
dependency. Tests override this dependency to inject a fresh service.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from models.cockpit_state import (
    Alert,
    CockpitState,
    CockpitTheme,
    GearPosition,
)
from services.cockpit_service import CockpitService


# ===========================================================================
# SECTION 1 — THE SHARED SINGLETON + ITS PROVIDER
# ===========================================================================
# This ONE instance lives for the whole life of the app. Every request that
# `Depends(get_cockpit_service)` receives THIS exact object, so they all share
# the same live state.
# ---------------------------------------------------------------------------
_cockpit_service = CockpitService()


def get_cockpit_service() -> CockpitService:
    """Return the single shared cockpit service (overridden in tests)."""
    return _cockpit_service


# ===========================================================================
# SECTION 2 — REQUEST MODELS (the shape of incoming JSON bodies)
# ===========================================================================
# These validate input at the HTTP boundary. Invalid values (bad theme, speed
# out of range) are auto-rejected by FastAPI with HTTP 422 before our code runs.
# ---------------------------------------------------------------------------


class SetDriverRequest(BaseModel):
    """Body for POST /cockpit/driver."""

    driver_name: str = Field(..., min_length=1)
    # Typed as the enum -> an invalid theme string is auto-rejected (422).
    theme: CockpitTheme = Field(default=CockpitTheme.MINIMAL)


class TelemetryUpdateRequest(BaseModel):
    """
    Body for POST /cockpit/telemetry. Both fields optional -> partial update.
    Range limits here give a clean 422 for bad values.
    """

    speed_kmh: Optional[float] = Field(default=None, ge=0.0, le=400.0)
    gear: Optional[GearPosition] = Field(default=None)


# ===========================================================================
# SECTION 3 — THE ROUTER
# ===========================================================================
router = APIRouter(prefix="/api/v1", tags=["cockpit"])


# -- health -----------------------------------------------------------------
@router.get("/health")
def health_check() -> dict:
    """Liveness probe."""
    return {"status": "ok", "service": "digital-cockpit"}


# -- read the whole screen --------------------------------------------------
@router.get("/cockpit/state", response_model=CockpitState)
def get_state(
    service: CockpitService = Depends(get_cockpit_service),
) -> CockpitState:
    """Return the full current cockpit state for the display to render."""
    return service.get_state()


# -- set the active driver (V1 -> V4 -> here) -------------------------------
@router.post("/cockpit/driver", response_model=CockpitState)
def set_driver(
    body: SetDriverRequest,
    service: CockpitService = Depends(get_cockpit_service),
) -> CockpitState:
    """
    Configure the cockpit for the active driver.

    V4 fills `theme` from Vertical 1's `ui_theme`, so the screen matches the
    driver's preference.
    """
    return service.set_driver(driver_name=body.driver_name, theme=body.theme)


# -- update telemetry (V4 / vehicle) ----------------------------------------
@router.post("/cockpit/telemetry", response_model=CockpitState)
def update_telemetry(
    body: TelemetryUpdateRequest,
    service: CockpitService = Depends(get_cockpit_service),
) -> CockpitState:
    """Partially update live numbers (speed, gear)."""
    return service.update_telemetry(speed_kmh=body.speed_kmh, gear=body.gear)


# -- raise / update an alert (V4 / V3) --------------------------------------
@router.post("/cockpit/alerts", response_model=CockpitState)
def raise_alert(
    alert: Alert,
    service: CockpitService = Depends(get_cockpit_service),
) -> CockpitState:
    """
    Push an alert onto the screen (or update one with the same alert_id).

    FastAPI has already validated the incoming JSON into a full Alert object.
    """
    return service.raise_alert(alert)


# -- dismiss an alert (V4) --------------------------------------------------
@router.post("/cockpit/alerts/{alert_id}/dismiss", response_model=CockpitState)
def dismiss_alert(
    alert_id: str,
    service: CockpitService = Depends(get_cockpit_service),
) -> CockpitState:
    """
    Remove an alert (the hazard has cleared).

    Returns 404 if no alert with that id is currently shown.
    """
    removed = service.dismiss_alert(alert_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active alert with id '{alert_id}'.",
        )
    return service.get_state()
