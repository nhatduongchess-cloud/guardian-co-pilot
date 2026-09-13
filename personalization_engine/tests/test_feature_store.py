"""
test_feature_store.py
=====================

Unit tests for VERTICAL 1 -> INGESTION -> feature_store.py.

We test BOTH implementations through the SAME assertions to prove they honour
the identical `FeatureStore` contract — that is what lets us swap JSONL for
Postgres later without touching a caller. The JSONL store is additionally
tested for on-disk persistence and append-only behaviour.
"""

import pytest

from ingestion.feature_store import (
    InMemoryFeatureStore,
    JsonlFeatureStore,
    FeatureStoreError,
)
from models.telemetry import TelemetryEvent


# ---------------------------------------------------------------------------
# fixtures: give each test both a fresh in-memory store and a fresh JSONL store
# ---------------------------------------------------------------------------
@pytest.fixture(params=["memory", "jsonl"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryFeatureStore()
    return JsonlFeatureStore(tmp_path / "log.jsonl")


def _event(trip, driver="driver_001", etype="FATIGUE_ALERT", **scalars):
    return TelemetryEvent(
        trip_id=trip, driver_id=driver, source="V3_PERCEPTION",
        event_type=etype, scalar_features=scalars or {},
    )


# ===========================================================================
# CONTRACT TESTS (run against both implementations)
# ===========================================================================


def test_append_and_get_by_event_id(store):
    evt = _event("trip_001", perclos=0.4)
    store.append(evt)
    fetched = store.get_event(evt.event_id)
    assert fetched is not None
    assert fetched.event_id == evt.event_id
    assert fetched.scalar_features.perclos == pytest.approx(0.4)


def test_get_missing_event_returns_none(store):
    assert store.get_event("does-not-exist") is None


def test_get_trip_events_filters_by_trip(store):
    store.append(_event("trip_001"))
    store.append(_event("trip_001"))
    store.append(_event("trip_002"))
    assert len(store.get_trip_events("trip_001")) == 2
    assert len(store.get_trip_events("trip_002")) == 1


def test_list_trip_ids_in_first_seen_order(store):
    store.append(_event("trip_A"))
    store.append(_event("trip_B"))
    store.append(_event("trip_A"))
    assert store.list_trip_ids("driver_001") == ["trip_A", "trip_B"]


def test_driver_isolation(store):
    store.append(_event("t1", driver="alice"))
    store.append(_event("t1", driver="bob"))
    assert len(store.get_driver_events("alice")) == 1
    assert store.list_trip_ids("bob") == ["t1"]


def test_append_is_ordered(store):
    store.append(_event("t1", perclos=0.1))
    store.append(_event("t1", perclos=0.2))
    perclos = [e.scalar_features.perclos for e in store.get_trip_events("t1")]
    assert perclos == [pytest.approx(0.1), pytest.approx(0.2)]


# ===========================================================================
# JSONL-SPECIFIC TESTS
# ===========================================================================


def test_jsonl_persists_across_instances(tmp_path):
    path = tmp_path / "log.jsonl"
    JsonlFeatureStore(path).append(_event("trip_001", perclos=0.5))
    # A brand-new store object reading the same file must see the event.
    reopened = JsonlFeatureStore(path)
    assert len(reopened.get_trip_events("trip_001")) == 1


def test_jsonl_missing_file_is_empty_not_error(tmp_path):
    store = JsonlFeatureStore(tmp_path / "not_created_yet.jsonl")
    assert store.get_driver_events("anyone") == []
    assert store.list_trip_ids("anyone") == []


def test_jsonl_corrupt_line_raises(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text("{not valid json}\n", encoding="utf-8")
    with pytest.raises(FeatureStoreError):
        JsonlFeatureStore(path).get_driver_events("driver_001")
