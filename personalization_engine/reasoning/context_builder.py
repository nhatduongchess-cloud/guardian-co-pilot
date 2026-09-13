"""
context_builder.py
==================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: REASONING  ·  §2.3 / Constraint #3

This is where "grounding" actually happens. It takes a raw `TelemetryEvent`
plus the driver's `DriverMemory` and assembles a fully-decided
`ExplanationContext`: humanised situation text, the decision in plain words,
and the single most-relevant memory snippet (retrieved by cosine similarity —
the light RAG from spec §1/§2.3).

CRITICAL SEPARATION OF CONCERNS
-------------------------------
Everything factual and safety-bearing is decided HERE, deterministically. By
the time the prompt reaches Phi-3, all the numbers, the decision, and the
history are fixed strings. The model can only change the wording, never the
facts. That is the architectural guarantee behind "no hallucinated safety
information" (spec §0.3, §8).
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from typing import Optional

import numpy as np

from memory.state_machine import PersonalizationStateMachine
from models.driver_memory import DriverMemory
from models.explanation import ExplanationContext
from models.telemetry import EventType, KernelAction, TelemetryEvent


# ===========================================================================
# SECTION 1 — SITUATION TEXT (numbers -> grounded Vietnamese)
# ===========================================================================
def _describe_situation(event: TelemetryEvent, memory: DriverMemory) -> str:
    """
    Turn the event's scalar readings into a short factual Vietnamese clause.
    Uses the driver's own baseline (via z-score) so we can honestly say
    "cao hơn bình thường của bạn" only when the data supports it.
    """
    s = event.scalar_features

    if event.event_type == EventType.FATIGUE_ALERT and s.perclos is not None:
        z = PersonalizationStateMachine.fatigue_zscore(memory, s.perclos)
        if z >= 1.0:
            qualifier = "cao hơn mức bình thường của bạn"
        elif z <= -1.0:
            qualifier = "thấp hơn mức bình thường của bạn"
        else:
            qualifier = "ở mức đáng lưu ý"
        return f"PERCLOS {s.perclos:.2f} ({qualifier})."

    if event.event_type == EventType.OBJECT_RISK and s.ttc_seconds is not None:
        obj = f" với {s.object_class}" if s.object_class else ""
        return f"Thời gian tới va chạm chỉ còn {s.ttc_seconds:.1f} giây{obj}."

    if event.event_type == EventType.OBJECT_RISK and s.object_class:
        # TTC may be unavailable (e.g. redacted evaluation data); still ground
        # the sentence in the real detected object class.
        return f"Guardian phát hiện {s.object_class} ở phía trước có nguy cơ va chạm."

    if event.event_type == EventType.SURFACE_RISK and s.friction_estimate is not None:
        return f"Độ bám đường ước lượng thấp ({s.friction_estimate:.2f})."

    if event.event_type == EventType.LANE_RISK:
        return "Xe có dấu hiệu lệch khỏi làn đường."

    if event.event_type == EventType.SAFETY_DECISION:
        return "Guardian đã đánh giá một tình huống an toàn."

    return "Guardian ghi nhận một tín hiệu an toàn."


# ===========================================================================
# SECTION 2 — DECISION TEXT (the deterministic kernel action, in words)
# ===========================================================================
def _describe_decision(event: TelemetryEvent) -> str:
    """Render what the V4 Safety Kernel did. Advisory framing, never a command."""
    decision = event.safety_kernel_decision
    if decision is None:
        return "Guardian đang tiếp tục theo dõi tình huống."
    if decision.action == KernelAction.BRAKE_ASSIST:
        return "Guardian đã hỗ trợ phanh để giữ khoảng cách an toàn."
    if decision.action == KernelAction.WARN:
        return "Guardian đã phát cảnh báo để nhắc bạn."
    return "Guardian ghi nhận và tiếp tục theo dõi."


# ===========================================================================
# SECTION 3 — RETRIEVAL (cosine similarity over stored snippets)
# ===========================================================================
def _retrieve_snippet(event: TelemetryEvent, memory: DriverMemory) -> str:
    """
    Return the summary of the memory snippet most similar to this event's risk
    embedding. Falls back to the most recent snippet when there is no usable
    embedding. Returns "" when the driver has no memory yet.

    In-process numpy cosine over a handful of vectors — no vector DB needed at
    this scale (spec §1 "bớt"). The `MemoryStore` interface still lets us swap
    in pgvector for Guardian Fleet later.
    """
    snippets = memory.memory_snippets
    if not snippets:
        return ""

    query = event.risk_embedding
    usable = [s for s in snippets if s.embedding and query and len(s.embedding) == len(query)]

    if not usable:
        # No comparable embeddings -> return the newest snippet's summary.
        return snippets[-1].summary_vi

    q = np.asarray(query, dtype=float)
    q_norm = np.linalg.norm(q)
    if q_norm == 0.0:
        return snippets[-1].summary_vi

    best_summary = usable[-1].summary_vi
    best_score = -1.0
    # Iterate so that on a tie the MOST RECENT snippet wins (>=), which reads
    # more naturally in an explanation than always citing the oldest trip.
    for snip in usable:
        v = np.asarray(snip.embedding, dtype=float)
        v_norm = np.linalg.norm(v)
        if v_norm == 0.0:
            continue
        score = float(np.dot(q, v) / (q_norm * v_norm))
        if score >= best_score:
            best_score = score
            best_summary = snip.summary_vi
    return best_summary


# ===========================================================================
# SECTION 4 — THE PUBLIC BUILDER
# ===========================================================================
def build_context(
    event: TelemetryEvent,
    memory: DriverMemory,
    include_history: bool = True,
) -> ExplanationContext:
    """
    Assemble a grounded ExplanationContext from one event + the driver memory.

    `include_history=False` skips retrieval — handy for COLD_START, where there
    is nothing personal to reference yet.
    """
    snippet = _retrieve_snippet(event, memory) if include_history else ""
    return ExplanationContext(
        event_id=event.event_id,
        event_type=event.event_type,
        situation_summary=_describe_situation(event, memory),
        decision_taken=_describe_decision(event),
        driver_memory_snippet=snippet,
        personalization_state=memory.state,
        target_language="vi",
    )


# ===========================================================================
# SECTION 5 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    from models.telemetry import SafetyKernelDecision

    mem = DriverMemory.new("driver_001")
    mem.baseline.fatigue_ema = 0.30
    mem.baseline.fatigue_var = 0.0025  # std = 0.05
    evt = TelemetryEvent(
        trip_id="trip_010", driver_id="driver_001",
        source="V4_WORLD_MODEL", event_type="FATIGUE_ALERT",
        scalar_features={"perclos": 0.45},
        safety_kernel_decision=SafetyKernelDecision(action="WARN", threshold_used=0.7),
    )
    ctx = build_context(evt, mem)
    print(ctx.model_dump_json(indent=2))
