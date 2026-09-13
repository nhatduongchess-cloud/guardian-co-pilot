"""
routes.py
=========

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: API (the HTTP door of the whole vertical)

This file turns our PersonalizationService into real web endpoints using
FastAPI. The vehicle and the other verticals talk to Vertical 1 ONLY through
these HTTP routes — they never import our Python classes directly.

KEY FASTAPI IDEAS USED HERE
---------------------------
1) APIRouter
   A group of related endpoints. main.py will plug this router into the app.
   Keeping routes in their own file keeps main.py tiny and clean.

2) Depends (Dependency Injection)
   Each endpoint declares "I need a PersonalizationService". FastAPI calls
   `get_service()` to build one and injects it. In tests we OVERRIDE these
   providers to inject fakes — no real files needed.

3) response_model
   We tell FastAPI the exact shape each endpoint returns. FastAPI then
   validates our output and documents it automatically at /docs.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import os

# FastAPI building blocks:
#   APIRouter    -> a collection of endpoints.
#   Depends      -> declares a dependency to be injected.
#   HTTPException-> the way to return an HTTP error (e.g. 404) cleanly.
#   status       -> named HTTP status codes (readable, e.g. HTTP_404_NOT_FOUND).
from fastapi import APIRouter, Depends, HTTPException, status

from models.driver_profile import ActiveDriverProfile, DriverProfile
from repository.profile_repository import JsonProfileRepository, ProfileRepository
from services.personalization_service import (
    DriverNotFoundError,
    PersonalizationService,
)


# ===========================================================================
# SECTION 1 — DEPENDENCY PROVIDERS (the wiring)
# ===========================================================================
# These small functions build the objects our endpoints need. FastAPI calls
# them automatically. Because they are separate functions, tests can override
# them with fakes via `app.dependency_overrides`.
# ---------------------------------------------------------------------------

# Where the JSON "database" lives. Configurable via an environment variable
# so we never hard-code a path. Falls back to a sensible default that works
# out of the box here.
_DEFAULT_DB_PATH = os.getenv("GUARDIAN_DB_PATH", "mock_data/drivers.json")


def get_repository() -> ProfileRepository:
    """Build the storage backend. Swap this ONE function to change databases."""
    return JsonProfileRepository(_DEFAULT_DB_PATH)


def get_service(
    repository: ProfileRepository = Depends(get_repository),
) -> PersonalizationService:
    """
    Build the service and inject the repository into it.

    Notice: FastAPI resolves `Depends(get_repository)` first, then passes the
    result here. This is a small dependency CHAIN — exactly the clean wiring
    we designed in the service layer.
    """
    return PersonalizationService(repository)


# ===========================================================================
# SECTION 2 — THE ROUTER
# ===========================================================================
# `prefix="/api/v1"` -> every path below starts with /api/v1 (versioned API,
#                       so we can ship a v2 later without breaking clients).
# `tags=[...]`       -> groups these endpoints together in the /docs page.
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/api/v1", tags=["personalization"])


# -- health check -----------------------------------------------------------
@router.get("/health")
def health_check() -> dict:
    """
    Liveness probe. Returns immediately so the vehicle (or a monitor) can
    confirm the Personalization Engine is up. No data access involved.
    """
    return {"status": "ok", "service": "personalization-engine"}


# -- list all drivers -------------------------------------------------------
@router.get("/drivers", response_model=list[DriverProfile])
def list_drivers(
    service: PersonalizationService = Depends(get_service),
) -> list[DriverProfile]:
    """Return every known driver profile."""
    return service.list_drivers()


# -- get one driver ---------------------------------------------------------
@router.get("/drivers/{driver_id}", response_model=DriverProfile)
def get_driver(
    driver_id: str,
    service: PersonalizationService = Depends(get_service),
) -> DriverProfile:
    """
    Return one driver by id.

    If the driver does not exist, the service raises DriverNotFoundError,
    which we translate into a clean HTTP 404 response.
    """
    try:
        return service.get_driver(driver_id)
    except DriverNotFoundError as exc:
        # Convert a business error into the correct HTTP status code.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


# -- create or update a driver ---------------------------------------------
@router.post(
    "/drivers",
    response_model=DriverProfile,
    status_code=status.HTTP_201_CREATED,
)
def create_or_update_driver(
    profile: DriverProfile,
    service: PersonalizationService = Depends(get_service),
) -> DriverProfile:
    """
    Create a new driver or update an existing one.

    FastAPI has ALREADY validated the incoming JSON into a DriverProfile
    before this function runs — so `profile` is guaranteed well-formed.
    The service then applies the safety policy before saving.
    """
    return service.create_or_update_driver(profile)


# -- activate a driver (THE STAR) ------------------------------------------
@router.post("/drivers/{driver_id}/activate", response_model=ActiveDriverProfile)
def activate_driver(
    driver_id: str,
    service: PersonalizationService = Depends(get_service),
) -> ActiveDriverProfile:
    """
    Make a driver the ACTIVE driver and publish their live profile.

    This is the endpoint the whole car calls when someone sits down. The
    returned ActiveDriverProfile is what Verticals 2, 3, and 4 consume.
    """
    try:
        return service.activate_driver(driver_id)
    except DriverNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
