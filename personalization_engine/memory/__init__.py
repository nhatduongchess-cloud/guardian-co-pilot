"""
memory package — Driver Memory & Learning Loop (§2.2).

    store.py          -> MemoryStore interface (+ in-memory impl for tests)
    sqlite_store.py   -> zero-ops SQLite implementation shipped for the demo
    state_machine.py  -> COLD_START / WARMING / PERSONALIZED transitions,
                         EMA baseline update, z-score, safety-floor clamp
"""

from memory.store import MemoryStore, MemoryStoreError, InMemoryMemoryStore
from memory.sqlite_store import SqliteMemoryStore
from memory.state_machine import PersonalizationStateMachine

__all__ = [
    "MemoryStore",
    "MemoryStoreError",
    "InMemoryMemoryStore",
    "SqliteMemoryStore",
    "PersonalizationStateMachine",
]
