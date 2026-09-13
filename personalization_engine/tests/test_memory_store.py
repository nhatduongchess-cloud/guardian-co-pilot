"""
test_memory_store.py
====================

Unit tests for VERTICAL 1 -> MEMORY -> store.py + sqlite_store.py.

Both stores are tested through the SAME contract assertions (the point of the
`MemoryStore` interface), plus a SQLite-specific persistence test.
"""

import pytest

from memory.sqlite_store import SqliteMemoryStore
from memory.store import InMemoryMemoryStore
from models.driver_memory import DriverMemory, PersonalizationState


# ---------------------------------------------------------------------------
# fixture: run every contract test against both implementations
# ---------------------------------------------------------------------------
@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        return InMemoryMemoryStore()
    return SqliteMemoryStore(tmp_path / "memory.db")


# ===========================================================================
# CONTRACT TESTS
# ===========================================================================


def test_get_unknown_returns_none(store):
    assert store.get("nobody") is None


def test_get_or_create_makes_cold_start(store):
    mem = store.get_or_create("driver_001")
    assert mem.state == PersonalizationState.COLD_START
    assert mem.trip_count == 0
    assert store.exists("driver_001") is True


def test_save_then_get_roundtrips(store):
    mem = DriverMemory.new("driver_001")
    mem.trip_count = 3
    mem.state = PersonalizationState.PERSONALIZED
    mem.baseline.fatigue_ema = 0.42
    store.save(mem)

    loaded = store.get("driver_001")
    assert loaded.trip_count == 3
    assert loaded.state == PersonalizationState.PERSONALIZED
    assert loaded.baseline.fatigue_ema == pytest.approx(0.42)


def test_save_is_upsert(store):
    store.save(DriverMemory.new("driver_001"))
    mem = store.get("driver_001")
    mem.trip_count = 5
    store.save(mem)
    assert store.get("driver_001").trip_count == 5


def test_delete_removes_and_reports(store):
    store.get_or_create("driver_001")
    assert store.delete("driver_001") is True
    assert store.get("driver_001") is None
    assert store.delete("driver_001") is False  # already gone


def test_stored_object_is_isolated_from_caller(store):
    mem = store.get_or_create("driver_001")
    mem.trip_count = 99  # mutate the copy we got back
    # The store must NOT have been changed by mutating our local copy.
    assert store.get("driver_001").trip_count == 0


# ===========================================================================
# SQLITE-SPECIFIC
# ===========================================================================


def test_sqlite_persists_across_instances(tmp_path):
    path = tmp_path / "memory.db"
    s1 = SqliteMemoryStore(path)
    mem = s1.get_or_create("driver_001")
    mem.trip_count = 4
    s1.save(mem)
    s1.close()

    s2 = SqliteMemoryStore(path)
    assert s2.get("driver_001").trip_count == 4
    s2.close()
