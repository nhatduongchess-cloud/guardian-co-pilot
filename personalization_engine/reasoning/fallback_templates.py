"""
fallback_templates.py
=====================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: REASONING  ·  §2.3 / Constraint #3

The DETERMINISTIC safety net of the reasoning engine. When Phi-3 is not
available, is too slow, or returns something we do not trust, we produce a
short Vietnamese explanation from a fixed template filled with REAL facts from
the `ExplanationContext`.

Two reasons this file matters more than the LLM itself:

  1. It can never hallucinate. Every word is either fixed Vietnamese or a
     value copied verbatim from the grounded context. A wrong safety
     explanation is worse than a plain one (spec §0.3), so the safe default
     must be boring and correct.

  2. It guarantees an answer. `GetExplanation` must NEVER return empty and
     must NEVER let the LLM invent — so the engine always has this to fall
     back to (Constraint #3).

The templates are intentionally simple and readable. They are the reference
"ground truth" the LLM is later asked only to REPHRASE more naturally.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from models.driver_memory import PersonalizationState
from models.explanation import ExplanationContext
from models.telemetry import EventType


# ===========================================================================
# SECTION 1 — PER-EVENT-TYPE OPENERS
# ===========================================================================
# The first clause names WHAT happened, in plain Vietnamese, per event type.
# ---------------------------------------------------------------------------
_OPENERS: dict[EventType, str] = {
    EventType.FATIGUE_ALERT: "Guardian nhận thấy dấu hiệu mệt mỏi",
    EventType.OBJECT_RISK: "Guardian phát hiện nguy cơ va chạm với vật cản",
    EventType.LANE_RISK: "Guardian nhận thấy xe đang lệch làn",
    EventType.SURFACE_RISK: "Guardian nhận thấy mặt đường trơn trượt",
    EventType.SAFETY_DECISION: "Guardian đã xử lý một tình huống an toàn",
}

# A closing clause that reflects the personalisation state — this is what
# lets the demo show the explanation TONE change as the driver graduates.
_STATE_SUFFIX: dict[PersonalizationState, str] = {
    PersonalizationState.COLD_START: (
        "Guardian đang dùng ngưỡng an toàn mặc định vì chưa đủ dữ liệu về bạn."
    ),
    PersonalizationState.WARMING: (
        "Guardian đang học thói quen lái của bạn để điều chỉnh cho phù hợp hơn."
    ),
    PersonalizationState.PERSONALIZED: (
        "Guardian đã điều chỉnh theo thói quen lái quen thuộc của bạn."
    ),
}


# ===========================================================================
# SECTION 2 — THE PUBLIC BUILDER
# ===========================================================================


def render_fallback(context: ExplanationContext) -> str:
    """
    Build a grounded, EXACTLY <=2 sentence Vietnamese explanation (schema §3.4).

    Structure (2 sentences):
        S1: <opener>: <situation>; <decision>.
        S2: [<history> — ]<state suffix>

    We join clauses with ';' and '—' rather than '.' so the whole thing stays
    within two sentences. Every dynamic piece comes straight from `context`,
    so the output is a faithful restatement of facts, never an invention.
    """
    opener = _OPENERS.get(context.event_type, "Guardian ghi nhận một sự kiện an toàn")

    # Sentence 1: what happened + what the deterministic kernel did, joined
    # with a semicolon so it reads as ONE sentence.
    situation = _strip_terminal(context.situation_summary)
    decision = _strip_terminal(context.decision_taken)
    sentence1 = f"{opener}: {situation}; {decision}."

    # Sentence 2: the state-aware tone, optionally led by a short history clause
    # joined with an em dash (keeps it a single sentence). State suffixes
    # already end in a period.
    suffix = _STATE_SUFFIX[context.personalization_state]
    snippet = _strip_terminal(context.driver_memory_snippet.strip())
    sentence2 = f"{snippet} — {suffix}" if snippet else suffix

    return f"{sentence1} {sentence2}".strip()


# ===========================================================================
# SECTION 3 — HELPERS
# ===========================================================================
def _strip_terminal(text: str) -> str:
    """Remove a trailing sentence-final period so clauses can be joined inline."""
    text = text.strip()
    return text[:-1] if text.endswith(".") else text


# ===========================================================================
# SECTION 4 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    for state in PersonalizationState:
        ctx = ExplanationContext(
            event_id="evt_1",
            event_type=EventType.FATIGUE_ALERT,
            situation_summary="PERCLOS 0.42, cao hơn mức bình thường của bạn.",
            decision_taken="Guardian đã phát cảnh báo nhắc bạn nghỉ ngơi.",
            driver_memory_snippet="Chuyến trước bạn cũng có dấu hiệu mệt vào buổi tối.",
            personalization_state=state,
        )
        print(f"[{state.value}] {render_fallback(ctx)}\n")
