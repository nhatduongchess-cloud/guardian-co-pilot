"""
test_client_memory.py
=====================

Tests the CROSS-VERTICAL client methods for the Driver-Memory / Reasoning API
(the §2.4 surface V4 and V2 depend on). Same technique as test_client.py: a
PersonalizationClient talks to the real FastAPI app in memory, backed by
in-memory stores and the deterministic fallback backend.
"""

import pytest
from fastapi.testclient import TestClient

from main import app
from api.memory_routes import (
    get_feature_store,
    get_memory_store,
    get_reasoning_engine,
)
from client.personalization_client import PersonalizationClient
from ingestion.feature_store import InMemoryFeatureStore
from memory.store import InMemoryMemoryStore
from models.telemetry import EventType, TelemetryEvent
from reasoning.phi3_runtime import DisabledLLMBackend
from reasoning.reasoning_engine import ReasoningEngine


@pytest.fixture
def client():
    features = InMemoryFeatureStore()
    memory = InMemoryMemoryStore()
    engine = ReasoningEngine(backend=DisabledLLMBackend())

    app.dependency_overrides[get_feature_store] = lambda: features
    app.dependency_overrides[get_memory_store] = lambda: memory
    app.dependency_overrides[get_reasoning_engine] = lambda: engine

    test_http = TestClient(app)
    yield PersonalizationClient(http_client=test_http)

    app.dependency_overrides.clear()


def _fatigue(trip, perclos, driver="driver_001"):
    return TelemetryEvent(
        trip_id=trip, driver_id=driver, source="V3_PERCEPTION",
        event_type="FATIGUE_ALERT", risk_embedding=[0.1, 0.2, 0.3],
        scalar_features={"perclos": perclos},
    )


# ===========================================================================
# CROSS-VERTICAL FLOW (V4 threshold + V2 explanation)
# ===========================================================================


def test_ingest_then_explain_via_client(client):
    stored = client.ingest_telemetry(_fatigue("trip_001", 0.42))
    assert stored.event_id

    resp = client.get_explanation(stored.event_id)
    assert resp.text_vi.strip()
    assert resp.used_fallback_template is True


def test_three_trips_and_threshold_via_client(client):
    for i, p in enumerate([0.30, 0.31, 0.30], start=1):
        trip = f"trip_{i:03d}"
        client.ingest_telemetry(_fatigue(trip, p))
        mem = client.end_trip("driver_001", trip)

    assert mem.state.value == "PERSONALIZED"
    assert mem.trip_count == 3

    thr = client.get_personalized_threshold("driver_001", EventType.FATIGUE_ALERT)
    assert thr.is_advisory is True
    assert thr.used_global_default is False
    assert thr.personalization_state.value == "PERSONALIZED"


def test_threshold_accepts_string_event_type(client):
    thr = client.get_personalized_threshold("driver_001", "FATIGUE_ALERT")
    assert thr.event_type == EventType.FATIGUE_ALERT


def test_explain_live_event_via_client(client):
    resp = client.explain_event(_fatigue("trip_001", 0.42))
    assert resp.text_vi.strip()


# ===========================================================================
# TRANSPARENCY / CONSENT via client
# ===========================================================================


def test_memory_consent_reset_via_client(client):
    for i, p in enumerate([0.30, 0.31, 0.30], start=1):
        trip = f"trip_{i:03d}"
        client.ingest_telemetry(_fatigue(trip, p))
        client.end_trip("driver_001", trip)

    mem = client.get_memory("driver_001")
    assert mem.trip_count == 3

    off = client.set_consent("driver_001", enabled=False)
    assert off.consent.personalization_enabled is False

    reset = client.reset_memory("driver_001")
    assert reset.state.value == "COLD_START"
    assert reset.consent.last_reset_at is not None


def test_reasoning_status_via_client(client):
    status = client.reasoning_status()
    assert status["llm_backend_available"] is False
    assert status["mode"] == "deterministic-fallback"
