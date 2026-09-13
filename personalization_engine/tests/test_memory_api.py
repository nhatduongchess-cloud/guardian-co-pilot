"""
test_memory_api.py
==================

Integration tests for VERTICAL 1 -> API -> memory_routes.py.

We drive the REAL FastAPI app end-to-end with Starlette's TestClient (in-process,
no network port), but inject IN-MEMORY stores and a DISABLED LLM backend via
`app.dependency_overrides`. That keeps the tests:
  * deterministic (fallback templates, never a real model),
  * isolated (nothing written to mock_data/),
  * fast (no disk, no network).

The headline test walks a driver through 3 trips and asserts the state machine
graduates COLD_START -> WARMING -> PERSONALIZED over HTTP — the exact demo the
reviewers will see.
"""

import pytest
from fastapi.testclient import TestClient

from main import app
from api.memory_routes import (
    get_feature_store,
    get_memory_store,
    get_reasoning_engine,
)
from ingestion.feature_store import InMemoryFeatureStore
from memory.store import InMemoryMemoryStore
from reasoning.phi3_runtime import DisabledLLMBackend
from reasoning.reasoning_engine import ReasoningEngine


# ---------------------------------------------------------------------------
# fixture: a TestClient wired to fresh in-memory infrastructure
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    features = InMemoryFeatureStore()
    memory = InMemoryMemoryStore()
    engine = ReasoningEngine(backend=DisabledLLMBackend())

    app.dependency_overrides[get_feature_store] = lambda: features
    app.dependency_overrides[get_memory_store] = lambda: memory
    app.dependency_overrides[get_reasoning_engine] = lambda: engine

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


def _fatigue_body(trip, perclos, driver="driver_001"):
    return {
        "trip_id": trip,
        "driver_id": driver,
        "source": "V3_PERCEPTION",
        "event_type": "FATIGUE_ALERT",
        "risk_embedding": [0.1, 0.2, 0.3],
        "scalar_features": {"perclos": perclos},
    }


def _ingest_and_end(client, trip, perclos):
    r = client.post("/api/v1/telemetry", json=_fatigue_body(trip, perclos))
    assert r.status_code == 201, r.text
    event_id = r.json()["event_id"]
    r2 = client.post(f"/api/v1/drivers/driver_001/trips/{trip}/end")
    assert r2.status_code == 200, r2.text
    return event_id, r2.json()


# ===========================================================================
# GROUP 1 — INGESTION + EXPLANATION
# ===========================================================================


def test_ingest_returns_event_with_id(client):
    r = client.post("/api/v1/telemetry", json=_fatigue_body("trip_001", 0.4))
    assert r.status_code == 201
    assert r.json()["event_id"]


def test_explanation_by_event_id(client):
    r = client.post("/api/v1/telemetry", json=_fatigue_body("trip_001", 0.42))
    event_id = r.json()["event_id"]

    r2 = client.get(f"/api/v1/explanations/{event_id}")
    assert r2.status_code == 200
    body = r2.json()
    assert body["text_vi"].strip()
    assert body["used_fallback_template"] is True   # no model in tests
    assert body["audio_ready"] is True


def test_explanation_unknown_event_404(client):
    assert client.get("/api/v1/explanations/nope").status_code == 404


def test_explain_live_event(client):
    r = client.post("/api/v1/explain", json=_fatigue_body("trip_001", 0.42))
    assert r.status_code == 200
    assert r.json()["text_vi"].strip()


# ===========================================================================
# GROUP 2 — THE HEADLINE: STATE MACHINE OVER HTTP
# ===========================================================================


def test_three_trips_graduate_cold_to_personalized(client):
    _, mem1 = _ingest_and_end(client, "trip_001", 0.30)
    assert mem1["state"] == "COLD_START"
    assert mem1["trip_count"] == 1

    _, mem2 = _ingest_and_end(client, "trip_002", 0.31)
    assert mem2["state"] == "WARMING"

    _, mem3 = _ingest_and_end(client, "trip_003", 0.30)
    assert mem3["state"] == "PERSONALIZED"
    assert mem3["trip_count"] == 3
    assert len(mem3["memory_snippets"]) == 3


def test_end_trip_with_no_events_404(client):
    assert client.post("/api/v1/drivers/driver_001/trips/ghost/end").status_code == 404


# ===========================================================================
# GROUP 3 — ADVISORY THRESHOLD (read by V4)
# ===========================================================================


def test_threshold_is_global_at_cold_start(client):
    r = client.get(
        "/api/v1/drivers/new_driver/threshold", params={"event_type": "FATIGUE_ALERT"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["used_global_default"] is True
    assert body["is_advisory"] is True
    assert body["personalization_state"] == "COLD_START"


def test_threshold_personalizes_after_three_trips(client):
    for i, p in enumerate([0.30, 0.31, 0.30], start=1):
        _ingest_and_end(client, f"trip_{i:03d}", p)

    r = client.get(
        "/api/v1/drivers/driver_001/threshold", params={"event_type": "FATIGUE_ALERT"}
    )
    body = r.json()
    assert body["personalization_state"] == "PERSONALIZED"
    assert body["used_global_default"] is False
    assert body["confidence"] > 0.0


def test_threshold_invalid_event_type_422(client):
    r = client.get(
        "/api/v1/drivers/driver_001/threshold", params={"event_type": "BANANA"}
    )
    assert r.status_code == 422


# ===========================================================================
# GROUP 4 — TRANSPARENCY & CONSENT
# ===========================================================================


def test_get_memory_creates_cold_start(client):
    r = client.get("/api/v1/drivers/fresh/memory")
    assert r.status_code == 200
    assert r.json()["state"] == "COLD_START"


def test_opt_out_forces_global_threshold(client):
    for i, p in enumerate([0.30, 0.31, 0.30], start=1):
        _ingest_and_end(client, f"trip_{i:03d}", p)

    off = client.put(
        "/api/v1/drivers/driver_001/consent", params={"enabled": False}
    )
    assert off.status_code == 200
    assert off.json()["consent"]["personalization_enabled"] is False

    r = client.get(
        "/api/v1/drivers/driver_001/threshold", params={"event_type": "FATIGUE_ALERT"}
    )
    assert r.json()["used_global_default"] is True


def test_reset_memory_wipes_learned_data(client):
    for i, p in enumerate([0.30, 0.31, 0.30], start=1):
        _ingest_and_end(client, f"trip_{i:03d}", p)

    r = client.delete("/api/v1/drivers/driver_001/memory")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "COLD_START"
    assert body["trip_count"] == 0
    assert body["consent"]["last_reset_at"] is not None


# ===========================================================================
# GROUP 5 — REASONING BACKEND HEALTH
# ===========================================================================


def test_reasoning_status_reports_fallback(client):
    r = client.get("/api/v1/reasoning/status")
    assert r.status_code == 200
    body = r.json()
    # No model is loaded in tests, so the engine reports the fallback mode.
    assert body["llm_backend_available"] is False
    assert body["mode"] == "deterministic-fallback"
