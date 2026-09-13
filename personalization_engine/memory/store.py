"""
store.py
========

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: MEMORY  ·  §2.2 / §4 of the Architecture Spec

The `MemoryStore` interface is the seam the spec calls out explicitly: the
project uses SQLite, but Guardian Fleet (v3.0) will move to Postgres +
pgvector. Because every caller depends on THIS abstract class — never a
concrete store — that migration is a one-file swap.

This module ships the interface plus a dependency-free `InMemoryMemoryStore`
used by the unit tests (so state-machine tests never touch disk). The real
`SqliteMemoryStore` lives next door in sqlite_store.py.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Optional

from models.driver_memory import DriverMemory


# ===========================================================================
# SECTION 1 — ERRORS
# ===========================================================================


class MemoryStoreError(Exception):
    """Base error for any Driver Memory storage problem."""


# ===========================================================================
# SECTION 2 — THE INTERFACE
# ===========================================================================


class MemoryStore(ABC):
    """
    Persistence contract for Driver Memory.

    `get_or_create` is the method callers use most: personalisation should
    Just Work for a driver we have never seen (they start at COLD_START).
    """

    @abstractmethod
    def get(self, driver_id: str) -> Optional[DriverMemory]:
        """Return the stored memory, or None if this driver is unknown."""
        raise NotImplementedError

    @abstractmethod
    def save(self, memory: DriverMemory) -> DriverMemory:
        """Insert or update a driver's memory; return what was saved."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, driver_id: str) -> bool:
        """Hard-delete a driver's memory. Return True if something was removed."""
        raise NotImplementedError

    # -- convenience (shared, not abstract) ---------------------------------
    def get_or_create(self, driver_id: str) -> DriverMemory:
        """
        Return the existing memory, or a fresh COLD_START one (persisted) for a
        brand-new driver. This keeps every caller from re-implementing the
        "first time we see this driver" branch.
        """
        existing = self.get(driver_id)
        if existing is not None:
            return existing
        fresh = DriverMemory.new(driver_id)
        return self.save(fresh)

    def exists(self, driver_id: str) -> bool:
        return self.get(driver_id) is not None


# ===========================================================================
# SECTION 3 — IN-MEMORY IMPLEMENTATION (tests / ephemeral)
# ===========================================================================


class InMemoryMemoryStore(MemoryStore):
    """A dict-backed store. Fast, isolated, disk-free — ideal for unit tests."""

    def __init__(self) -> None:
        self._data: dict[str, DriverMemory] = {}

    def get(self, driver_id: str) -> Optional[DriverMemory]:
        mem = self._data.get(driver_id)
        # Return a deep copy so callers cannot mutate our stored object by
        # accident — the same guarantee the SQLite store gives (it re-parses).
        return mem.model_copy(deep=True) if mem is not None else None

    def save(self, memory: DriverMemory) -> DriverMemory:
        self._data[memory.driver_id] = memory.model_copy(deep=True)
        return memory

    def delete(self, driver_id: str) -> bool:
        return self._data.pop(driver_id, None) is not None


# ===========================================================================
# SECTION 4 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    store = InMemoryMemoryStore()
    mem = store.get_or_create("driver_001")
    print("new driver state:", mem.state.value)   # COLD_START
    print("exists:", store.exists("driver_001"))   # True
    print("deleted:", store.delete("driver_001"))  # True
