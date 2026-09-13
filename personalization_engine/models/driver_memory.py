"""
driver_memory.py
================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: MODELS  ·  Schema §3.2 of the Architecture Spec

This is the "Driver Memory": the per-driver profile the Learning Loop builds
up over trips. It is DELIBERATELY separate from `DriverProfile`
(driver_profile.py):

    DriverProfile  -> WHO the driver is + their chosen comfort/UI preferences
                      (seat, climate, theme). Static, edited by a human.
    DriverMemory   -> WHAT Guardian has LEARNED about this driver's behaviour
                      (fatigue baseline, alert responsiveness, driving style).
                      Grown automatically, one trip at a time.

Keeping them apart matters: comfort preferences must never silently move a
safety threshold, and learned behaviour must never overwrite what the human
explicitly set. They meet only at the API surface.

THE THREE CONSTRAINTS THIS FILE ENCODES
---------------------------------------
* Constraint #2 — personalisation activates only after >=3 trips. Modelled as
  an explicit `PersonalizationState` machine, not a magic `if trip_count > 3`.
* Safety floor — V4 owns a minimum threshold V1 may NEVER go below. Stored
  here as `safety_floor`, read-only from V1's point of view.
* Consent / transparency — the driver can inspect and wipe this memory.
  Modelled as a first-class `Consent` block, not an afterthought.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ===========================================================================
# SECTION 1 — THE PERSONALIZATION STATE MACHINE (Constraint #2)
# ===========================================================================


class PersonalizationState(str, Enum):
    """
    The lifecycle of personalisation for one driver.

        COLD_START   trips 1-2  -> use GLOBAL defaults only. Log, do not adapt.
        WARMING      trip 3     -> baseline forming; blend global + personal.
        PERSONALIZED trips >=3   -> use personal baseline (still safety-floored).

    Encoding this as a state (instead of scattering `trip_count` checks) means
    the demo can literally show a driver graduating COLD_START -> PERSONALIZED,
    and every layer reads one authoritative field to decide how to behave.
    """

    COLD_START = "COLD_START"
    WARMING = "WARMING"
    PERSONALIZED = "PERSONALIZED"


# ===========================================================================
# SECTION 2 — NESTED VALUE OBJECTS
# ===========================================================================


class Baseline(BaseModel):
    """
    The rolling, explainable statistics the Learning Loop maintains. We use an
    EMA (exponential moving average) + variance instead of a trained model:
    fast, transparent, and good enough at the 3-10 trip scale of the demo
    (spec §1 "bớt": rolling baseline over ML model).
    """

    model_config = ConfigDict(extra="forbid")

    fatigue_ema: float = Field(
        default=0.0, ge=0.0,
        description="Exponential moving average of the fatigue signal (PERCLOS).",
    )
    fatigue_var: float = Field(
        default=0.0, ge=0.0,
        description="Running variance of the fatigue signal (for z-scores).",
    )
    alert_response_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Fraction of alerts the driver reacted to (0..1).",
    )
    false_alarm_feedback_count: int = Field(
        default=0, ge=0,
        description="How many times the driver flagged an alert as a false alarm.",
    )
    driving_style_vector: list[float] = Field(
        default_factory=list,
        description="Averaged risk-context embedding — this driver's 'style'.",
    )


class SafetyFloor(BaseModel):
    """
    The absolute minimum threshold V4 permits. V1 reads it and CLAMPS every
    advisory number to it; V1 code must never lower a value below the floor.
    """

    model_config = ConfigDict(extra="forbid")

    min_ttc_threshold: float = Field(
        default=1.2, ge=0.0,
        description="Minimum Time-To-Collision threshold in seconds.",
    )
    max_fatigue_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Max PERCLOS before an alert MUST fire (upper safety bound).",
    )
    note: str = Field(
        default="V4 sets and owns this value; V1 never lowers below it.",
        description="Reminder of who owns this data.",
    )


class MemorySnippet(BaseModel):
    """
    One short, human-readable memory of a past trip, plus its embedding so the
    reasoning engine can RETRIEVE the most relevant snippet for grounding
    (the light RAG in spec §1 / §2.3).
    """

    model_config = ConfigDict(extra="forbid")

    trip_id: str = Field(..., min_length=1)
    summary_vi: str = Field(..., description="1-2 sentence Vietnamese summary.")
    embedding: list[float] = Field(
        default_factory=list,
        description="Context embedding for cosine-similarity retrieval.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class Consent(BaseModel):
    """Transparency & control block — the driver owns their memory."""

    model_config = ConfigDict(extra="forbid")

    personalization_enabled: bool = Field(
        default=True,
        description="If False, V1 always uses global defaults (opt-out).",
    )
    last_reset_at: Optional[datetime] = Field(
        default=None,
        description="When the driver last wiped their memory (if ever).",
    )


# ===========================================================================
# SECTION 3 — THE DRIVER MEMORY (§3.2)
# ===========================================================================


class DriverMemory(BaseModel):
    """
    The full learned profile for one driver. This is what the MemoryStore
    persists and what GetMemorySummary exposes to the driver for transparency.
    """

    model_config = ConfigDict(extra="forbid")

    driver_id: str = Field(..., min_length=1)
    state: PersonalizationState = Field(
        default=PersonalizationState.COLD_START,
        description="Current personalisation lifecycle stage.",
    )
    trip_count: int = Field(
        default=0, ge=0,
        description="Number of COMPLETED trips learned from.",
    )
    baseline: Baseline = Field(default_factory=Baseline)
    safety_floor: SafetyFloor = Field(default_factory=SafetyFloor)
    memory_snippets: list[MemorySnippet] = Field(default_factory=list)
    consent: Consent = Field(default_factory=Consent)

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this memory was last modified.",
    )

    @classmethod
    def new(cls, driver_id: str) -> "DriverMemory":
        """Create a fresh, COLD_START memory for a brand-new driver."""
        return cls(driver_id=driver_id)


# ===========================================================================
# SECTION 4 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    mem = DriverMemory.new("driver_001")
    print(mem.model_dump_json(indent=2))
    print("state:", mem.state.value, "| trips:", mem.trip_count)
