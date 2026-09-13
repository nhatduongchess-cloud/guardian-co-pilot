"""
memory_routes.py
================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: API  ·  the HTTP surface for §2.4 (Personalization API)

This router exposes the Driver-Memory + Reasoning half of Vertical 1, kept in
its own file so the original `routes.py` (driver identification) stays small.
main.py plugs both routers into the same app under /api/v1.

ENDPOINT MAP (spec §2.4)
------------------------
    POST   /telemetry                                 ingest a TelemetryEvent
    POST   /drivers/{id}/trips/{trip_id}/end          run the Learning Loop
    GET    /drivers/{id}/threshold?event_type=...     GetPersonalizedThreshold (V4)
    GET    /explanations/{event_id}                    GetExplanation (V2)
    POST   /explain                                    explain a live event (V2)
    GET    /drivers/{id}/memory                        GetMemorySummary (consent)
    DELETE /drivers/{id}/memory                        ResetDriverMemory (consent)
    PUT    /drivers/{id}/consent?enabled=...           opt in/out
    GET    /reasoning/status                           LLM backend health

TRANSPORT NOTE
--------------
The spec pictures these as gRPC methods. We serve them over REST for the
project; the SERVICE layer beneath is transport-agnostic, so a future gRPC
server (to match TV5's middleware) would reuse the exact same services.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import os
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status

from ingestion.feature_store import FeatureStore, JsonlFeatureStore
from memory.sqlite_store import SqliteMemoryStore
from memory.store import MemoryStore
from models.driver_memory import DriverMemory
from models.explanation import ExplanationResponse, PersonalizedThreshold
from models.telemetry import EventType, TelemetryEvent
from reasoning.reasoning_engine import ReasoningEngine
from services.explanation_service import EventNotFoundError, ExplanationService
from services.memory_service import DriverMemoryService, TripNotFoundError


# ===========================================================================
# SECTION 1 — DEPENDENCY PROVIDERS (singletons, overridable in tests)
# ===========================================================================
# These are cached so that ingested events, the SQLite memory, and the loaded
# Phi-3 model persist across requests within one running server. Tests override
# them via `app.dependency_overrides` to inject in-memory fakes.
# ---------------------------------------------------------------------------

_FEATURE_STORE_PATH = os.getenv(
    "GUARDIAN_FEATURE_STORE_PATH", "mock_data/telemetry_log.jsonl"
)
_MEMORY_DB_PATH = os.getenv("GUARDIAN_MEMORY_DB_PATH", "mock_data/driver_memory.db")


@lru_cache(maxsize=1)
def get_feature_store() -> FeatureStore:
    """The append-only telemetry log. Swap this ONE function to change stores."""
    return JsonlFeatureStore(_FEATURE_STORE_PATH)


@lru_cache(maxsize=1)
def get_memory_store() -> MemoryStore:
    """The Driver-Memory persistence (SQLite here)."""
    return SqliteMemoryStore(_MEMORY_DB_PATH)


@lru_cache(maxsize=1)
def get_reasoning_engine() -> ReasoningEngine:
    """The reasoning engine (lazy Phi-3 + deterministic fallback)."""
    return ReasoningEngine()


def get_memory_service(
    features: FeatureStore = Depends(get_feature_store),
    memory: MemoryStore = Depends(get_memory_store),
) -> DriverMemoryService:
    return DriverMemoryService(features, memory)


def get_explanation_service(
    features: FeatureStore = Depends(get_feature_store),
    memory: MemoryStore = Depends(get_memory_store),
    engine: ReasoningEngine = Depends(get_reasoning_engine),
) -> ExplanationService:
    return ExplanationService(features, memory, engine)


# ===========================================================================
# SECTION 2 — THE ROUTER
# ===========================================================================
router = APIRouter(prefix="/api/v1", tags=["driver-memory"])


# -- ingestion --------------------------------------------------------------
@router.post(
    "/telemetry",
    response_model=TelemetryEvent,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest one telemetry event from V3/V4",
)
def ingest_telemetry(
    event: TelemetryEvent,
    service: DriverMemoryService = Depends(get_memory_service),
) -> TelemetryEvent:
    """
    Append a safety event to the feature store. FastAPI has already validated
    the body into a TelemetryEvent, so `event` is well-formed. Returns the
    stored event, including the `event_id` you later pass to GetExplanation.
    """
    return service.ingest_event(event)


# -- the Learning Loop ------------------------------------------------------
@router.post(
    "/drivers/{driver_id}/trips/{trip_id}/end",
    response_model=DriverMemory,
    summary="End a trip and fold it into the driver's memory (Learning Loop)",
)
def end_trip(
    driver_id: str,
    trip_id: str,
    service: DriverMemoryService = Depends(get_memory_service),
) -> DriverMemory:
    """Run the batch learning update for a finished trip; return updated memory."""
    try:
        return service.end_trip(driver_id, trip_id)
    except TripNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


# -- advisory threshold (read by V4) ---------------------------------------
@router.get(
    "/drivers/{driver_id}/threshold",
    response_model=PersonalizedThreshold,
    summary="Advisory, safety-floored threshold for one event type (read by V4)",
)
def get_threshold(
    driver_id: str,
    event_type: EventType,
    service: DriverMemoryService = Depends(get_memory_service),
) -> PersonalizedThreshold:
    """Return the advisory threshold. It is a suggestion only (Constraint #1)."""
    return service.get_personalized_threshold(driver_id, event_type)


# -- explanation (read by V2) ----------------------------------------------
@router.get(
    "/explanations/{event_id}",
    response_model=ExplanationResponse,
    summary="Explain a previously ingested event, in Vietnamese",
)
def get_explanation(
    event_id: str,
    service: ExplanationService = Depends(get_explanation_service),
) -> ExplanationResponse:
    """GetExplanation(event_id). Always grounded; never empty (Constraint #3)."""
    try:
        return service.explain_event(event_id)
    except EventNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="Explain a live event object (no prior ingestion needed)",
)
def explain_live(
    event: TelemetryEvent,
    service: ExplanationService = Depends(get_explanation_service),
) -> ExplanationResponse:
    """Convenience for V2: explain an event the moment it is produced."""
    return service.explain_event_object(event)


# -- transparency / consent -------------------------------------------------
@router.get(
    "/drivers/{driver_id}/memory",
    response_model=DriverMemory,
    summary="Show the driver everything Guardian has learned (transparency)",
)
def get_memory(
    driver_id: str,
    service: DriverMemoryService = Depends(get_memory_service),
) -> DriverMemory:
    """GetMemorySummary. Creates a COLD_START record if the driver is new."""
    return service.get_memory(driver_id)


@router.delete(
    "/drivers/{driver_id}/memory",
    response_model=DriverMemory,
    summary="Wipe learned data (ResetDriverMemory)",
)
def reset_memory(
    driver_id: str,
    service: DriverMemoryService = Depends(get_memory_service),
) -> DriverMemory:
    """Erase learned baseline/history; keep consent + V4 safety floor."""
    return service.reset_memory(driver_id)


@router.put(
    "/drivers/{driver_id}/consent",
    response_model=DriverMemory,
    summary="Turn personalisation on/off for a driver",
)
def set_consent(
    driver_id: str,
    enabled: bool,
    service: DriverMemoryService = Depends(get_memory_service),
) -> DriverMemory:
    """When disabled, all thresholds fall back to global defaults."""
    return service.set_consent(driver_id, enabled)


# -- reasoning backend health ----------------------------------------------
@router.get("/reasoning/status", summary="Report which reasoning backend is active")
def reasoning_status(
    engine: ReasoningEngine = Depends(get_reasoning_engine),
) -> dict:
    """
    Tells the demo operator whether the real Phi-3 model is loaded or whether
    the deterministic fallback templates are in use.
    """
    available = engine.backend_available()
    return {
        "llm_backend_available": available,
        "mode": "phi3-onnx" if available else "deterministic-fallback",
    }
