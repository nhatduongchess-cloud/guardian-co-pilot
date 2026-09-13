"""
test_context_builder.py
=======================

Unit tests for VERTICAL 1 -> REASONING -> context_builder.py.

The context builder is where GROUNDING happens: raw numbers become factual
Vietnamese, the kernel decision becomes plain words, and the most-relevant
memory snippet is retrieved by cosine similarity. Because every downstream
explanation is built from this object, we pin its behaviour directly here
(the API tests only exercise the FATIGUE path in fallback mode).
"""

import pytest

from models.driver_memory import DriverMemory, MemorySnippet, PersonalizationState
from models.telemetry import SafetyKernelDecision, TelemetryEvent
from reasoning.context_builder import build_context, _describe_situation, _describe_decision


def _event(etype, decision=None, **scalars):
    return TelemetryEvent(
        trip_id="trip_001", driver_id="driver_001",
        source="V4_WORLD_MODEL", event_type=etype,
        scalar_features=scalars,
        safety_kernel_decision=decision,
    )


# ===========================================================================
# GROUP 1 — SITUATION TEXT (numbers -> grounded Vietnamese)
# ===========================================================================


def test_fatigue_situation_high_zscore():
    mem = DriverMemory.new("driver_001")
    mem.baseline.fatigue_ema = 0.30
    mem.baseline.fatigue_var = 0.0025  # std 0.05 -> 0.45 is +3 sigma
    text = _describe_situation(_event("FATIGUE_ALERT", perclos=0.45), mem)
    assert "0.45" in text
    assert "cao hơn" in text


def test_fatigue_situation_low_zscore():
    mem = DriverMemory.new("driver_001")
    mem.baseline.fatigue_ema = 0.50
    mem.baseline.fatigue_var = 0.0025
    text = _describe_situation(_event("FATIGUE_ALERT", perclos=0.30), mem)
    assert "thấp hơn" in text


def test_object_situation_includes_ttc_and_class():
    mem = DriverMemory.new("driver_001")
    text = _describe_situation(
        _event("OBJECT_RISK", ttc_seconds=1.1, object_class="xe máy"), mem
    )
    assert "1.1" in text
    assert "xe máy" in text


def test_surface_situation_includes_friction():
    mem = DriverMemory.new("driver_001")
    text = _describe_situation(_event("SURFACE_RISK", friction_estimate=0.28), mem)
    assert "0.28" in text


@pytest.mark.parametrize("etype", ["LANE_RISK", "SAFETY_DECISION"])
def test_other_event_types_have_nonempty_situation(etype):
    assert _describe_situation(_event(etype), DriverMemory.new("d")).strip()


# ===========================================================================
# GROUP 2 — DECISION TEXT (deterministic kernel action -> words)
# ===========================================================================


@pytest.mark.parametrize("action,marker", [
    ("BRAKE_ASSIST", "phanh"),
    ("WARN", "cảnh báo"),
    ("NONE", "theo dõi"),
])
def test_decision_text(action, marker):
    evt = _event("SAFETY_DECISION", decision=SafetyKernelDecision(action=action))
    assert marker in _describe_decision(evt)


def test_decision_text_when_missing():
    assert _describe_decision(_event("FATIGUE_ALERT")).strip()


# ===========================================================================
# GROUP 3 — RETRIEVAL (cosine similarity over snippets)
# ===========================================================================


def _mem_with_snippets():
    mem = DriverMemory.new("driver_001")
    mem.state = PersonalizationState.PERSONALIZED
    mem.memory_snippets = [
        MemorySnippet(trip_id="t1", summary_vi="Chuyến t1: buổi sáng.", embedding=[1.0, 0.0, 0.0]),
        MemorySnippet(trip_id="t2", summary_vi="Chuyến t2: buổi tối.", embedding=[0.0, 1.0, 0.0]),
    ]
    return mem


def test_retrieval_picks_most_similar_snippet():
    mem = _mem_with_snippets()
    # A query aligned with t2's embedding must retrieve t2's summary.
    evt = _event("FATIGUE_ALERT", perclos=0.4)
    evt.risk_embedding = [0.0, 1.0, 0.0]
    ctx = build_context(evt, mem)
    assert "t2" in ctx.driver_memory_snippet


def test_retrieval_no_embedding_uses_latest():
    mem = _mem_with_snippets()
    evt = _event("FATIGUE_ALERT", perclos=0.4)  # no risk_embedding
    ctx = build_context(evt, mem)
    # With no comparable query vector, the newest snippet is used.
    assert "t2" in ctx.driver_memory_snippet


def test_retrieval_empty_memory_returns_blank():
    mem = DriverMemory.new("driver_001")
    ctx = build_context(_event("FATIGUE_ALERT", perclos=0.4), mem)
    assert ctx.driver_memory_snippet == ""


def test_build_context_passes_through_ids_and_state():
    mem = _mem_with_snippets()
    evt = _event("FATIGUE_ALERT", perclos=0.4)
    ctx = build_context(evt, mem, include_history=False)
    assert ctx.event_id == evt.event_id
    assert ctx.personalization_state == PersonalizationState.PERSONALIZED
    assert ctx.driver_memory_snippet == ""  # history disabled
