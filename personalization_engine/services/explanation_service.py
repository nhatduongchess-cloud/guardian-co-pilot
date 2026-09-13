"""
explanation_service.py
======================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: SERVICE  ·  §2.3 / §2.4

Implements GetExplanation(event_id): given an event that already happened,
produce the short Vietnamese explanation the cockpit (V2) will display/speak.

It is a thin coordinator on purpose:
    feature_store.get_event  -> the grounded facts
    memory_store.get         -> the driver's history + state
    context_builder          -> assemble the ExplanationContext
    reasoning_engine.explain -> phrase it (LLM or fallback), always grounded

All the safety-critical decisions were already made upstream (by V4 and by the
state machine). This service only RETRIEVES and EXPLAINS — it never decides
anything about the vehicle (Constraint #1).
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from ingestion.feature_store import FeatureStore
from memory.store import MemoryStore
from models.explanation import ExplanationResponse
from models.telemetry import TelemetryEvent
from reasoning.context_builder import build_context
from reasoning.reasoning_engine import ReasoningEngine


# ===========================================================================
# SECTION 1 — ERRORS
# ===========================================================================


class ExplanationServiceError(Exception):
    """Base error for the explanation service."""


class EventNotFoundError(ExplanationServiceError):
    """Raised when GetExplanation is called with an unknown event_id."""


# ===========================================================================
# SECTION 2 — THE SERVICE
# ===========================================================================


class ExplanationService:
    """Coordinates retrieval + reasoning to explain one recorded event."""

    def __init__(
        self,
        feature_store: FeatureStore,
        memory_store: MemoryStore,
        engine: ReasoningEngine,
    ) -> None:
        self._features = feature_store
        self._memory = memory_store
        self._engine = engine

    def explain_event(self, event_id: str) -> ExplanationResponse:
        """
        Explain a previously ingested event by its id.

        Raises EventNotFoundError if the id is unknown — the caller (the API)
        turns that into an HTTP 404.
        """
        event = self._features.get_event(event_id)
        if event is None:
            raise EventNotFoundError(f"No event found with id '{event_id}'.")
        return self._explain(event)

    def explain_event_object(self, event: TelemetryEvent) -> ExplanationResponse:
        """
        Explain an event object directly (without a prior store lookup). Handy
        for a live pipeline that wants an explanation the instant an event is
        produced, and for tests.
        """
        return self._explain(event)

    # -- internal -----------------------------------------------------------
    def _explain(self, event: TelemetryEvent) -> ExplanationResponse:
        # get_or_create so a brand-new driver still gets a COLD_START context.
        memory = self._memory.get_or_create(event.driver_id)
        context = build_context(event, memory)
        return self._engine.explain(context)

    def backend_available(self) -> bool:
        """Expose LLM availability for the /health endpoint."""
        return self._engine.backend_available()


# ===========================================================================
# SECTION 3 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    from ingestion.feature_store import InMemoryFeatureStore
    from memory.store import InMemoryMemoryStore
    from reasoning.phi3_runtime import DisabledLLMBackend

    features = InMemoryFeatureStore()
    memory = InMemoryMemoryStore()
    svc = ExplanationService(features, memory, ReasoningEngine(DisabledLLMBackend()))

    evt = TelemetryEvent(
        trip_id="trip_001", driver_id="driver_001",
        source="V4_WORLD_MODEL", event_type="OBJECT_RISK",
        scalar_features={"ttc_seconds": 1.1, "object_class": "xe máy"},
        safety_kernel_decision={"action": "BRAKE_ASSIST", "threshold_used": 2.0},
    )
    features.append(evt)

    resp = svc.explain_event(evt.event_id)
    print("used_fallback:", resp.used_fallback_template)
    print("text_vi      :", resp.text_vi)
