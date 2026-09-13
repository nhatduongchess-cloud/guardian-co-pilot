"""
memory_service.py
=================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: SERVICE  ·  §2.2 / §2.4

The business brain of the Driver Memory + Learning Loop. It wires together the
three pieces the API needs but keeps them decoupled through interfaces:

    FeatureStore  (where events live)      ── injected
    MemoryStore   (where memory lives)     ── injected
    PersonalizationStateMachine (the math) ── pure, static

Like the original `PersonalizationService`, this class never speaks HTTP and
never touches a concrete database — it depends only on abstract stores, so the
same logic runs against SQLite in production and in-memory fakes in tests.

RESPONSIBILITIES (map to the spec's API, §2.4)
----------------------------------------------
    ingest_event            <- POST /telemetry              (feature ingestion)
    end_trip                <- POST .../trips/{id}/end       (Learning Loop)
    get_personalized_threshold <- read by V4 (advisory only, safety-floored)
    get_memory              <- GetMemorySummary (transparency)
    reset_memory            <- ResetDriverMemory (consent)
    set_consent             <- opt in/out of personalisation
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from datetime import datetime, timezone
from typing import Optional

from ingestion.feature_store import FeatureStore
from memory.state_machine import PersonalizationStateMachine
from memory.store import MemoryStore
from models.driver_memory import DriverMemory
from models.explanation import PersonalizedThreshold
from models.telemetry import EventType, TelemetryEvent


# ===========================================================================
# SECTION 1 — ERRORS
# ===========================================================================


class MemoryServiceError(Exception):
    """Base error for the Driver Memory service."""


class TripNotFoundError(MemoryServiceError):
    """Raised when ending a trip that has no recorded events."""


# ===========================================================================
# SECTION 2 — THE SERVICE
# ===========================================================================


class DriverMemoryService:
    """Orchestrates ingestion, the Learning Loop, and advisory thresholds."""

    def __init__(
        self,
        feature_store: FeatureStore,
        memory_store: MemoryStore,
    ) -> None:
        self._features = feature_store
        self._memory = memory_store
        self._sm = PersonalizationStateMachine

    # -- ingestion ----------------------------------------------------------
    def ingest_event(self, event: TelemetryEvent) -> TelemetryEvent:
        """
        Persist one telemetry event and make sure the driver has a memory
        record (created at COLD_START the first time we see them). The learning
        update itself is deferred to `end_trip` — we never adapt mid-drive
        (spec §2.2: batch, not real-time).
        """
        self._features.append(event)
        self._memory.get_or_create(event.driver_id)
        return event

    # -- the Learning Loop --------------------------------------------------
    def end_trip(self, driver_id: str, trip_id: str) -> DriverMemory:
        """
        Fold a finished trip into the driver's memory and advance the state
        machine. Returns the updated memory (so a demo can show the transition).

        Raises TripNotFoundError if no events were recorded for that trip —
        we refuse to "learn" from nothing.
        """
        trip_events = [
            e for e in self._features.get_trip_events(trip_id)
            if e.driver_id == driver_id
        ]
        if not trip_events:
            raise TripNotFoundError(
                f"No events recorded for trip '{trip_id}' of driver '{driver_id}'."
            )

        memory = self._memory.get_or_create(driver_id)
        updated = self._sm.apply_completed_trip(memory, trip_events, trip_id)
        return self._memory.save(updated)

    # -- advisory threshold (read by V4) ------------------------------------
    def get_personalized_threshold(
        self, driver_id: str, event_type: EventType
    ) -> PersonalizedThreshold:
        """
        Return the advisory (safety-floored) threshold for one driver + event
        type. Constraint #1: this is a suggestion; V4 decides whether to use it.
        """
        memory = self._memory.get_or_create(driver_id)
        threshold, confidence, used_global, floored = self._sm.suggest_threshold(
            memory, event_type
        )
        return PersonalizedThreshold(
            driver_id=driver_id,
            event_type=event_type,
            threshold=threshold,
            confidence=confidence,
            personalization_state=memory.state,
            used_global_default=used_global,
            safety_floor_applied=floored,
            is_advisory=True,
        )

    # -- transparency / consent (driver-facing) -----------------------------
    def get_memory(self, driver_id: str) -> DriverMemory:
        """Return the driver's full memory (GetMemorySummary). Creates on demand."""
        return self._memory.get_or_create(driver_id)

    def reset_memory(self, driver_id: str) -> DriverMemory:
        """
        Wipe everything Guardian LEARNED about a driver (ResetDriverMemory),
        while preserving what the driver/V4 explicitly OWN: their consent choice
        and the V4-set safety floor. Stamps `last_reset_at` for auditability.
        """
        old = self._memory.get_or_create(driver_id)
        fresh = DriverMemory.new(driver_id)
        # Carry over the things that are NOT "learned data".
        fresh.safety_floor = old.safety_floor
        fresh.consent.personalization_enabled = old.consent.personalization_enabled
        fresh.consent.last_reset_at = datetime.now(timezone.utc)
        return self._memory.save(fresh)

    def set_consent(self, driver_id: str, enabled: bool) -> DriverMemory:
        """Turn personalisation on/off for a driver (they can always opt out)."""
        memory = self._memory.get_or_create(driver_id)
        memory.consent.personalization_enabled = enabled
        memory.updated_at = datetime.now(timezone.utc)
        return self._memory.save(memory)


# ===========================================================================
# SECTION 3 — SELF-TEST (a 3-trip walk-through, no HTTP, no disk)
# ===========================================================================
if __name__ == "__main__":
    from ingestion.feature_store import InMemoryFeatureStore
    from memory.store import InMemoryMemoryStore

    svc = DriverMemoryService(InMemoryFeatureStore(), InMemoryMemoryStore())

    for i in range(1, 4):
        trip = f"trip_{i:03d}"
        svc.ingest_event(TelemetryEvent(
            trip_id=trip, driver_id="driver_001",
            source="V3_PERCEPTION", event_type="FATIGUE_ALERT",
            risk_embedding=[0.1 * i, 0.2, 0.3],
            scalar_features={"perclos": 0.30 + 0.03 * i},
        ))
        mem = svc.end_trip("driver_001", trip)
        thr = svc.get_personalized_threshold("driver_001", EventType.FATIGUE_ALERT)
        print(
            f"trip {i}: state={mem.state.value:12s} "
            f"thr={thr.threshold:.3f} conf={thr.confidence:.2f} "
            f"global={thr.used_global_default}"
        )

    print("memory snippets:", len(svc.get_memory('driver_001').memory_snippets))
    reset = svc.reset_memory("driver_001")
    print("after reset -> state:", reset.state.value, "| reset_at:", reset.consent.last_reset_at)
