"""
feature_store.py
================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: INGESTION  ·  §2.1 of the Architecture Spec

The Feature Store is an APPEND-ONLY log of every `TelemetryEvent` we receive,
organised by trip. Two design ideas from the spec drive it:

  1. Separation of raw vs derived (feature-store pattern).
     Upstream teams write RAW events here once; the Learning Loop, threshold
     logic and reasoning engine all read the SAME layer. Nobody re-parses CAN
     or scene graphs per-vertical.

  2. Append-only + replayable.
     We never mutate a past event. That makes the demo reproducible: we can
     replay a whole trip from the log to show COLD_START -> PERSONALIZED.

TRANSPORT NOTE
--------------
The spec pictures V3/V4 pushing events over gRPC. We chose REST for the
project, so events arrive via the POST /telemetry endpoint and land here.
The `FeatureStore` interface below is transport-agnostic: swapping in a gRPC
subscriber later means writing a new caller, not touching this file.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Iterable, Optional

from pydantic import ValidationError

from models.telemetry import TelemetryEvent


# ===========================================================================
# SECTION 1 — ERRORS
# ===========================================================================


class FeatureStoreError(Exception):
    """Base error for any feature-store problem."""


# ===========================================================================
# SECTION 2 — THE INTERFACE (the contract, swappable for gRPC/Postgres later)
# ===========================================================================


class FeatureStore(ABC):
    """
    Abstract append-only event log. Concrete stores (JSONL now, Postgres for
    Guardian Fleet later) must honour these methods.
    """

    @abstractmethod
    def append(self, event: TelemetryEvent) -> TelemetryEvent:
        """Persist one event. Never overwrites an existing one."""
        raise NotImplementedError

    @abstractmethod
    def get_event(self, event_id: str) -> Optional[TelemetryEvent]:
        """Return one event by its id, or None (used by GetExplanation)."""
        raise NotImplementedError

    @abstractmethod
    def get_trip_events(self, trip_id: str) -> list[TelemetryEvent]:
        """Return every event of one trip, in arrival order."""
        raise NotImplementedError

    @abstractmethod
    def list_trip_ids(self, driver_id: str) -> list[str]:
        """Return the distinct trip ids seen for a driver, in first-seen order."""
        raise NotImplementedError

    @abstractmethod
    def get_driver_events(self, driver_id: str) -> list[TelemetryEvent]:
        """Return every event for one driver across all trips."""
        raise NotImplementedError


# ===========================================================================
# SECTION 3 — IN-MEMORY IMPLEMENTATION (tests + ephemeral demos)
# ===========================================================================


class InMemoryFeatureStore(FeatureStore):
    """A dependency-free store that keeps events in a list. Perfect for tests."""

    def __init__(self) -> None:
        self._events: list[TelemetryEvent] = []

    def append(self, event: TelemetryEvent) -> TelemetryEvent:
        self._events.append(event)
        return event

    def get_event(self, event_id: str) -> Optional[TelemetryEvent]:
        for e in self._events:
            if e.event_id == event_id:
                return e
        return None

    def get_trip_events(self, trip_id: str) -> list[TelemetryEvent]:
        return [e for e in self._events if e.trip_id == trip_id]

    def list_trip_ids(self, driver_id: str) -> list[str]:
        seen: list[str] = []
        for e in self._events:
            if e.driver_id == driver_id and e.trip_id not in seen:
                seen.append(e.trip_id)
        return seen

    def get_driver_events(self, driver_id: str) -> list[TelemetryEvent]:
        return [e for e in self._events if e.driver_id == driver_id]


# ===========================================================================
# SECTION 4 — JSONL IMPLEMENTATION (what we ship here)
# ===========================================================================


class JsonlFeatureStore(FeatureStore):
    """
    Append-only log backed by a single JSON Lines file (one event per line).

    Why JSONL, not a DB table?
      * Appending is a single `open(..., 'a')` write — no schema migrations.
      * It is human-readable and diffable — great for debugging on demo day.
      * A trip can be replayed just by reading lines back.

    At the demo's scale (hundreds of events) a full re-read per query is
    instant. The interface hides this, so a busy deployment can swap in the
    Postgres implementation without any caller change.
    """

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)

    # -- internal -----------------------------------------------------------
    def _iter_raw(self) -> Iterable[dict]:
        """Yield each stored event as a raw dict; empty if the file is absent."""
        if not self._file_path.exists():
            return
        try:
            with self._file_path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise FeatureStoreError(
                            f"Corrupt line {line_no} in {self._file_path}: {exc}"
                        ) from exc
        except OSError as exc:
            raise FeatureStoreError(
                f"Could not read feature store {self._file_path}: {exc}"
            ) from exc

    def _iter_events(self) -> Iterable[TelemetryEvent]:
        for raw in self._iter_raw():
            try:
                yield TelemetryEvent.model_validate(raw)
            except ValidationError as exc:
                raise FeatureStoreError(
                    f"Invalid stored event in {self._file_path}: {exc}"
                ) from exc

    # -- interface ----------------------------------------------------------
    def append(self, event: TelemetryEvent) -> TelemetryEvent:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.model_dump(mode="json"), ensure_ascii=False)
        try:
            with self._file_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            raise FeatureStoreError(
                f"Could not append to feature store {self._file_path}: {exc}"
            ) from exc
        return event

    def get_event(self, event_id: str) -> Optional[TelemetryEvent]:
        for e in self._iter_events():
            if e.event_id == event_id:
                return e
        return None

    def get_trip_events(self, trip_id: str) -> list[TelemetryEvent]:
        return [e for e in self._iter_events() if e.trip_id == trip_id]

    def list_trip_ids(self, driver_id: str) -> list[str]:
        seen: list[str] = []
        for e in self._iter_events():
            if e.driver_id == driver_id and e.trip_id not in seen:
                seen.append(e.trip_id)
        return seen

    def get_driver_events(self, driver_id: str) -> list[TelemetryEvent]:
        return [e for e in self._iter_events() if e.driver_id == driver_id]


# ===========================================================================
# SECTION 5 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    import tempfile

    tmp = Path(tempfile.gettempdir()) / "guardian_feature_store_demo.jsonl"
    if tmp.exists():
        tmp.unlink()

    store = JsonlFeatureStore(tmp)
    store.append(TelemetryEvent(
        trip_id="trip_001", driver_id="driver_001",
        source="V3_PERCEPTION", event_type="FATIGUE_ALERT",
        scalar_features={"perclos": 0.40},
    ))
    store.append(TelemetryEvent(
        trip_id="trip_001", driver_id="driver_001",
        source="V4_WORLD_MODEL", event_type="SAFETY_DECISION",
    ))
    print("trip ids:", store.list_trip_ids("driver_001"))
    print("events in trip_001:", len(store.get_trip_events("trip_001")))
    print(f"log at: {tmp}")
