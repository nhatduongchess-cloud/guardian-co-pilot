"""
explanation.py
==============

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: MODELS  ·  Schemas §3.3, §3.4 (+ the threshold contract for V4)

Two contracts live here:

  1. ExplanationContext  (INPUT to the reasoning engine, §3.3)
     A fully assembled, STRUCTURED bundle of facts. The reasoning engine is
     only allowed to phrase what is inside this object — never to invent new
     facts. This is how we keep explanations "grounded" (Constraint #3).

  2. ExplanationResponse (OUTPUT of the reasoning engine, §3.4)
     A short Vietnamese sentence + metadata (confidence, whether a fallback
     template was used, a trace id).

Plus `PersonalizedThreshold`: the advisory number V4 reads. It is explicitly
advisory (Constraint #1) — V4 decides whether to use it and always re-clamps
to its own safety floor.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from models.driver_memory import PersonalizationState
from models.telemetry import EventType


# ===========================================================================
# SECTION 1 — EXPLANATION CONTEXT (input to Phi-3, §3.3)
# ===========================================================================


class ExplanationContext(BaseModel):
    """
    Everything — and ONLY everything — the reasoning engine is allowed to say.

    The `context_builder` fills this from real data: the event that happened,
    the deterministic decision V4 took, one retrieved memory snippet, and the
    driver's current personalisation state. The LLM's whole job is to turn
    these facts into one natural Vietnamese sentence, not to add facts.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(..., description="The event being explained.")
    event_type: EventType = Field(..., description="Kind of safety event.")

    # A compact, already-humanised description of the situation, e.g.
    # "PERCLOS 0.42 (mức mệt cao), TTC 1.1s". Built deterministically so the
    # LLM never has to interpret raw numbers.
    situation_summary: str = Field(
        ...,
        description="Short, factual description of what was measured.",
    )

    decision_taken: str = Field(
        ...,
        description="What the V4 Safety Kernel did, in plain words.",
    )

    driver_memory_snippet: str = Field(
        default="",
        description="1-2 sentences of relevant history, from retrieval.",
    )

    personalization_state: PersonalizationState = Field(
        ...,
        description="COLD_START / WARMING / PERSONALIZED — changes the tone.",
    )

    target_language: str = Field(
        default="vi",
        description="Output language. The demo is Vietnamese.",
    )


# ===========================================================================
# SECTION 2 — EXPLANATION RESPONSE (output, §3.4)
# ===========================================================================


class ExplanationResponse(BaseModel):
    """The short, driver-facing explanation the cockpit (V2) will show/speak."""

    model_config = ConfigDict(extra="forbid")

    text_vi: str = Field(..., description="The explanation, <=2 sentences, Vietnamese.")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="How confident the engine is in this wording.",
    )
    used_fallback_template: bool = Field(
        default=False,
        description="True if a static template was used instead of the LLM.",
    )
    audio_ready: bool = Field(
        default=True,
        description="True if the text is safe to feed to TTS (always, once produced).",
    )
    trace_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Correlation id for logging/debugging one explanation.",
    )


# ===========================================================================
# SECTION 3 — PERSONALIZED THRESHOLD (advisory number read by V4)
# ===========================================================================


class PersonalizedThreshold(BaseModel):
    """
    The ADVISORY threshold for one driver + event type.

    IMPORTANT (Constraint #1): this is a suggestion. V4 is free to ignore it,
    and V1 has already clamped it to the safety floor before returning it.
    `is_advisory` is hard-coded True so no downstream reader can mistake it
    for a command.
    """

    model_config = ConfigDict(extra="forbid")

    driver_id: str
    event_type: EventType
    threshold: float = Field(..., description="Suggested (safety-floored) threshold.")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the personalisation (grows with trips).",
    )
    personalization_state: PersonalizationState = Field(
        ...,
        description="Which lifecycle stage produced this number.",
    )
    used_global_default: bool = Field(
        default=True,
        description="True while COLD_START / opted-out (no personalisation yet).",
    )
    safety_floor_applied: bool = Field(
        default=False,
        description="True if the floor clamped the suggestion.",
    )
    is_advisory: bool = Field(
        default=True,
        description="Always True. V1 advises; V4 decides.",
    )


# ===========================================================================
# SECTION 4 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    ctx = ExplanationContext(
        event_id="evt_1",
        event_type="FATIGUE_ALERT",
        situation_summary="PERCLOS 0.42 (cao hon binh thuong).",
        decision_taken="Phat canh bao nghi ngoi.",
        driver_memory_snippet="Chuyen truoc ban cung co dau hieu met vao buoi toi.",
        personalization_state="PERSONALIZED",
    )
    print(ctx.model_dump_json(indent=2))
