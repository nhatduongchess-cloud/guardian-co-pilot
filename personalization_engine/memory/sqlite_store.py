"""
sqlite_store.py
===============

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: MEMORY  ·  §4 (tech-stack table)

The pragmatic `MemoryStore`: a single SQLite file, using
only Python's stdlib `sqlite3` (zero external dependencies, zero ops). It
honours the exact same `MemoryStore` interface as the in-memory store, so
swapping between them — or to Postgres + pgvector later for Guardian Fleet —
never touches a caller.

STORAGE SHAPE
-------------
We keep a few columns queryable (state, trip_count) for quick dashboards, and
store the full `DriverMemory` as a JSON blob in one column. At this scale that
is the pragmatic choice: no migrations when the model grows, and the object
round-trips through Pydantic so it is always validated on the way out.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from memory.store import MemoryStore, MemoryStoreError
from models.driver_memory import DriverMemory


# ===========================================================================
# THE SQLITE STORE
# ===========================================================================


class SqliteMemoryStore(MemoryStore):
    """SQLite-backed Driver Memory persistence."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS driver_memory (
            driver_id   TEXT PRIMARY KEY,
            state       TEXT NOT NULL,
            trip_count  INTEGER NOT NULL,
            updated_at  TEXT NOT NULL,
            data        TEXT NOT NULL          -- full DriverMemory as JSON
        )
    """

    def __init__(self, db_path: str | Path) -> None:
        """
        Args:
            db_path: file for the SQLite database, e.g. "mock_data/memory.db".
                     Use ":memory:" for an ephemeral in-process database.

        We hold ONE long-lived connection guarded by a lock, rather than a new
        connection per call. Two reasons:
          * `:memory:` databases are private to a single connection — a
            per-call connection would lose the schema between operations.
          * FastAPI runs sync endpoints in a threadpool, so `check_same_thread`
            is disabled and the lock serialises access. At the demo's scale
            (a handful of drivers) serialising DB access costs nothing.
        """
        self._db_path = str(db_path)
        # Make sure the parent folder exists (except for the special :memory:).
        if self._db_path != ":memory:":
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._ensure_schema()

    # -- internals ----------------------------------------------------------
    def _ensure_schema(self) -> None:
        try:
            with self._lock, self._conn:
                self._conn.execute(self._SCHEMA)
        except sqlite3.Error as exc:
            raise MemoryStoreError(f"Could not initialise SQLite schema: {exc}") from exc

    # -- interface ----------------------------------------------------------
    def get(self, driver_id: str) -> Optional[DriverMemory]:
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT data FROM driver_memory WHERE driver_id = ?",
                    (driver_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise MemoryStoreError(f"Read failed for '{driver_id}': {exc}") from exc

        if row is None:
            return None
        try:
            return DriverMemory.model_validate_json(row[0])
        except ValidationError as exc:
            raise MemoryStoreError(
                f"Stored memory for '{driver_id}' is corrupt: {exc}"
            ) from exc

    def save(self, memory: DriverMemory) -> DriverMemory:
        payload = memory.model_dump_json()
        try:
            with self._lock, self._conn:  # lock + transaction (auto-commit)
                self._conn.execute(
                    """
                    INSERT INTO driver_memory (driver_id, state, trip_count, updated_at, data)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(driver_id) DO UPDATE SET
                        state=excluded.state,
                        trip_count=excluded.trip_count,
                        updated_at=excluded.updated_at,
                        data=excluded.data
                    """,
                    (
                        memory.driver_id,
                        memory.state.value,
                        memory.trip_count,
                        memory.updated_at.isoformat(),
                        payload,
                    ),
                )
        except sqlite3.Error as exc:
            raise MemoryStoreError(
                f"Write failed for '{memory.driver_id}': {exc}"
            ) from exc
        return memory

    def delete(self, driver_id: str) -> bool:
        try:
            with self._lock, self._conn:
                cur = self._conn.execute(
                    "DELETE FROM driver_memory WHERE driver_id = ?", (driver_id,)
                )
                return cur.rowcount > 0
        except sqlite3.Error as exc:
            raise MemoryStoreError(f"Delete failed for '{driver_id}': {exc}") from exc

    def close(self) -> None:
        """Close the underlying connection (call on shutdown / in tests)."""
        with self._lock:
            self._conn.close()


# ===========================================================================
# SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    store = SqliteMemoryStore(":memory:")
    mem = store.get_or_create("driver_001")
    print("created state:", mem.state.value)

    mem.trip_count = 3
    mem.state = mem.state.__class__.PERSONALIZED
    store.save(mem)

    loaded = store.get("driver_001")
    print("reloaded:", loaded.state.value, "trips", loaded.trip_count)
    print("deleted:", store.delete("driver_001"))
    print("after delete:", store.get("driver_001"))
