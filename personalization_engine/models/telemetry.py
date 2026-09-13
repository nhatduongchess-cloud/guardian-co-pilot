"""
telemetry.py
============

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: MODELS  ·  Schema §3.1 of the Architecture Spec

This file defines the INPUT contract of the Personalization Engine: the
`TelemetryEvent`. Every safety-relevant thing that happens while driving —
a fatigue alert from V3, an object-risk from perception, a decision taken by
the V4 Safety Kernel — arrives here as one `TelemetryEvent`.

WHY A STRICT SCHEMA MATTERS
---------------------------
V1 consumes data produced by three other teams (V3 perception, V4 world
model / safety kernel, TV5 middleware). If everyone agrees on THIS shape
early (Week 1 of the plan), the teams can build in parallel without guessing
each other's JSON. A wrong field name found on demo day is a disaster; a
wrong field name rejected by Pydantic at ingestion time is a 5-second fix.

ADVISORY-ONLY BOUNDARY (Constraint #1)
--------------------------------------
A `TelemetryEvent` can CARRY the Safety Kernel's decision (so V1 can explain
it), but V1 never PRODUCES a driving command. The `safety_kernel_decision`
field is read-only context for us: it records what V4 already, deterministically,
decided. V1 only ever advises and explains.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


# ===========================================================================
# SECTION 1 — ENUMS (fixed vocabularies shared across the vertical)
# ===========================================================================


class EventSource(str, Enum):
    """Which upstream vertical emitted this event."""

    V3_PERCEPTION = "V3_PERCEPTION"      # scene graph / risk embedding
    V4_WORLD_MODEL = "V4_WORLD_MODEL"    # fused world model + safety kernel


class EventType(str, Enum):
    """
    The KIND of safety event. This is the primary key we personalise on:
    a driver has a different baseline for fatigue than for object risk, so
    thresholds and explanations are computed per `EventType`.
    """

    FATIGUE_ALERT = "FATIGUE_ALERT"      # driver drowsiness (PERCLOS / EAR)
    OBJECT_RISK = "OBJECT_RISK"          # collision risk with an object
    LANE_RISK = "LANE_RISK"              # lane departure / drift
    SURFACE_RISK = "SURFACE_RISK"        # low friction road surface
    SAFETY_DECISION = "SAFETY_DECISION"  # V4 kernel took an action


class KernelAction(str, Enum):
    """The action the V4 Safety Kernel took. DETERMINISTIC, never AI."""

    NONE = "NONE"                # observed, no intervention
    WARN = "WARN"               # advisory warning to the driver
    BRAKE_ASSIST = "BRAKE_ASSIST"  # assisted braking engaged


# ===========================================================================
# SECTION 2 — NESTED VALUE OBJECTS
# ===========================================================================


class ScalarFeatures(BaseModel):
    """
    The numeric readings attached to an event. All fields are OPTIONAL because
    a fatigue event carries `perclos`/`ear` while an object event carries
    `ttc_seconds`/`object_class` — no single event fills them all.

    `extra="allow"` deliberately lets upstream teams add new scalar readings
    (e.g. `steering_entropy`) without breaking ingestion. The feature store
    keeps whatever it is sent; the parts V1 does not understand are simply
    ignored by the reasoning/threshold logic.
    """

    model_config = ConfigDict(extra="allow")

    perclos: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="% of time eyes are closed (fatigue signal, 0..1).",
    )
    ear: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Eye Aspect Ratio (lower = eyes more closed).",
    )
    ttc_seconds: Optional[float] = Field(
        default=None, ge=0.0,
        description="Time-To-Collision in seconds (lower = more dangerous).",
    )
    object_class: Optional[str] = Field(
        default=None,
        description="Detected object type, e.g. 'pedestrian', 'car'.",
    )
    friction_estimate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Estimated road friction (lower = more slippery).",
    )


class SafetyKernelDecision(BaseModel):
    """
    A read-only record of what V4's Safety Kernel decided for this event.
    V1 uses it ONLY to explain the decision to the driver — never to change it.
    """

    model_config = ConfigDict(extra="forbid")

    action: KernelAction = Field(
        default=KernelAction.NONE,
        description="What the deterministic kernel did.",
    )
    threshold_used: float = Field(
        default=0.0,
        description="The numeric threshold the kernel compared against.",
    )
    is_deterministic: bool = Field(
        default=True,
        description="Always True — the kernel is rule-based, never AI.",
    )


# ===========================================================================
# SECTION 3 — THE TELEMETRY EVENT (the input contract, §3.1)
# ===========================================================================


class TelemetryEvent(BaseModel):
    """
    One safety-relevant event on a trip. This is the atomic record that the
    Feature Store appends and that everything downstream (state machine,
    threshold logic, reasoning engine) reads.
    """

    model_config = ConfigDict(extra="forbid")

    # `event_id` is added by V1 on ingestion (the spec's GetExplanation takes
    # an event_id, so every stored event must have one). If the caller does
    # not supply it we mint a stable UUID.
    event_id: str = Field(
        default_factory=lambda: uuid4().hex,
        description="Unique id for this event (used by GetExplanation).",
    )

    trip_id: str = Field(..., min_length=1, description="Which trip this belongs to.")
    driver_id: str = Field(..., min_length=1, description="Who was driving.")

    timestamp_ms: int = Field(
        default_factory=lambda: int(datetime.now(timezone.utc).timestamp() * 1000),
        ge=0,
        description="Event time in Unix milliseconds.",
    )

    source: EventSource = Field(..., description="Which vertical emitted this.")
    event_type: EventType = Field(..., description="The kind of safety event.")

    # 128-d risk embedding from V3 (§2.1). Optional so a pure V4 event (which
    # may not carry an embedding) still validates. Length is NOT hard-pinned
    # to 128 to stay tolerant of dimension changes during this project; the
    # retrieval code treats it as an opaque vector.
    risk_embedding: list[float] = Field(
        default_factory=list,
        description="Risk-context embedding from V3 perception (e.g. 128-d).",
    )

    scalar_features: ScalarFeatures = Field(
        default_factory=ScalarFeatures,
        description="Numeric readings for this event.",
    )

    safety_kernel_decision: Optional[SafetyKernelDecision] = Field(
        default=None,
        description="What V4 decided (present on SAFETY_DECISION events).",
    )

    def primary_scalar(self) -> Optional[float]:
        """
        Return the single most relevant scalar for this event type — the value
        the baseline/z-score personalisation is computed on. Kept here (on the
        data object) so every layer agrees which number 'matters' per type.
        """
        s = self.scalar_features
        if self.event_type == EventType.FATIGUE_ALERT:
            return s.perclos
        if self.event_type == EventType.OBJECT_RISK:
            return s.ttc_seconds
        if self.event_type == EventType.SURFACE_RISK:
            return s.friction_estimate
        # LANE_RISK / SAFETY_DECISION have no single canonical scalar here.
        return None


# ===========================================================================
# SECTION 4 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    evt = TelemetryEvent(
        trip_id="trip_001",
        driver_id="driver_001",
        source="V3_PERCEPTION",
        event_type="FATIGUE_ALERT",
        risk_embedding=[0.1, 0.2, 0.3],
        scalar_features={"perclos": 0.42, "ear": 0.18},
    )
    print(evt.model_dump_json(indent=2))
    print("primary scalar (perclos):", evt.primary_scalar())
