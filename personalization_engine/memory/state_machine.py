"""
state_machine.py
================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: MEMORY  ·  §2.2 of the Architecture Spec  ·  Constraint #2

This is the deterministic core of personalisation. It contains NO AI and NO
I/O — just pure, testable functions over a `DriverMemory`. That is why the
plan (§6) says to write its tests FIRST: everything else trusts this math.

WHAT IT OWNS
------------
1. The state machine COLD_START -> WARMING -> PERSONALIZED (Constraint #2:
   personalisation only after >=3 trips).
2. The Learning Loop's batch update: after a trip ends, fold that trip's
   events into the driver's rolling baseline (EMA + variance).
3. Threshold SUGGESTION math: blend a global default with the personal
   baseline by a weight that grows with trips, then CLAMP to the V4 safety
   floor. The result is advisory only (Constraint #1).

WHY EMA + Z-SCORE INSTEAD OF A TRAINED MODEL (spec §1 "bớt")
-----------------------------------------------------------
At 3-10 trips there is not enough data to train anything meaningful, and an
opaque model cannot be explained to a driver. An exponential moving average
with a running variance is fast, needs no training, and — crucially — lets us
say "your PERCLOS is 2.1 standard deviations above your normal", which is a
sentence a human trusts.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import math
from datetime import datetime, timezone

from models.driver_memory import DriverMemory, MemorySnippet, PersonalizationState
from models.telemetry import EventType, TelemetryEvent


# ===========================================================================
# SECTION 1 — TUNING CONSTANTS (one obvious place to tweak behaviour)
# ===========================================================================

# How many completed trips before we trust a personal baseline. This is the
# "3 trips" business rule from the slides, named once.
_TRIPS_FOR_PERSONALIZATION = 3

# EMA smoothing factor. Higher = adapt faster to recent trips, lower = smoother.
_EMA_ALPHA = 0.35

# When suggesting a fatigue threshold we allow warning slightly LATER for a
# driver whose normal PERCLOS runs high: threshold = ema + margin * std.
_FATIGUE_MARGIN_STD = 1.5

# Global defaults, used during COLD_START and as the base of every blend.
# Keyed by event type. Fatigue/friction are "warn when ABOVE", TTC is
# "warn when BELOW" — the safety-floor clamp in §5 respects that direction.
GLOBAL_DEFAULTS: dict[EventType, float] = {
    EventType.FATIGUE_ALERT: 0.70,   # PERCLOS: warn when eyes-closed fraction > 0.70
    EventType.OBJECT_RISK: 2.0,      # TTC seconds: warn when time-to-collision < 2.0
    EventType.SURFACE_RISK: 0.35,    # friction: warn when estimate < 0.35
    EventType.LANE_RISK: 0.50,       # generic lane-risk score
}


# ===========================================================================
# SECTION 2 — THE STATE MACHINE
# ===========================================================================


class PersonalizationStateMachine:
    """
    A stateless helper: give it a `DriverMemory`, get back the next memory or a
    derived number. It never stores anything itself (the MemoryStore does that).
    """

    # -- state derivation ---------------------------------------------------
    @staticmethod
    def state_for(trip_count: int) -> PersonalizationState:
        """
        Map completed-trip count to a lifecycle state.

            0-1 completed  -> COLD_START   (we are inside trip 1 or 2)
            2   completed  -> WARMING      (we are inside trip 3, blending)
            >=3 completed  -> PERSONALIZED

        Reading this table against the slides: personalisation is fully on only
        after the 3rd completed trip — exactly Constraint #2.
        """
        if trip_count >= _TRIPS_FOR_PERSONALIZATION:
            return PersonalizationState.PERSONALIZED
        if trip_count == _TRIPS_FOR_PERSONALIZATION - 1:  # == 2
            return PersonalizationState.WARMING
        return PersonalizationState.COLD_START

    @staticmethod
    def blend_weight(state: PersonalizationState) -> float:
        """
        How much to trust the PERSONAL baseline vs the global default, in [0,1].
        0 = pure global (safe when we know nothing), 1 = pure personal.
        """
        return {
            PersonalizationState.COLD_START: 0.0,
            PersonalizationState.WARMING: 0.5,
            PersonalizationState.PERSONALIZED: 0.85,
        }[state]

    @staticmethod
    def confidence_for(trip_count: int) -> float:
        """
        Confidence in the personalisation, in [0,1], growing with experience.
        Saturates a little above the personalisation threshold so the number
        keeps climbing for a few trips after PERSONALIZED begins.
        """
        return max(0.0, min(1.0, trip_count / (_TRIPS_FOR_PERSONALIZATION + 2)))

    # -- z-score (explainability helper) ------------------------------------
    @staticmethod
    def fatigue_zscore(memory: DriverMemory, value: float) -> float:
        """
        How many standard deviations `value` is above this driver's normal
        fatigue. Returns 0.0 when we have no spread yet (avoids divide-by-zero).
        Used by the reasoning engine to say "cao hon binh thuong".
        """
        std = math.sqrt(memory.baseline.fatigue_var)
        if std <= 1e-9:
            return 0.0
        return (value - memory.baseline.fatigue_ema) / std

    # -- the Learning Loop batch update -------------------------------------
    @classmethod
    def apply_completed_trip(
        cls,
        memory: DriverMemory,
        trip_events: list[TelemetryEvent],
        trip_id: str,
    ) -> DriverMemory:
        """
        Fold ONE finished trip into the driver's memory and return the updated
        copy. This is the batch update the spec insists runs AFTER a trip, not
        during driving (§2.2) — so it can never add latency to the Fast Path.

        Steps:
          1. Update the fatigue EMA + variance from this trip's PERCLOS samples.
          2. Update the driving-style vector (running mean of risk embeddings).
          3. Update alert-response rate if the events carry response signals.
          4. Increment trip_count and recompute the state.
          5. Append a short Vietnamese memory snippet for later retrieval.

        We work on a deep copy so the caller's object is never mutated.
        """
        mem = memory.model_copy(deep=True)

        # -- 1. fatigue EMA + variance --------------------------------------
        perclos_samples = [
            e.scalar_features.perclos
            for e in trip_events
            if e.event_type == EventType.FATIGUE_ALERT
            and e.scalar_features.perclos is not None
        ]
        for x in perclos_samples:
            cls._update_ema_var(mem, x)

        # -- 2. driving-style vector (mean of risk embeddings) --------------
        embeddings = [e.risk_embedding for e in trip_events if e.risk_embedding]
        if embeddings:
            cls._update_style_vector(mem, embeddings)

        # -- 3. alert-response rate (only if the data is there) -------------
        responses = [
            float(bool(e.scalar_features.model_extra.get("responded")))
            for e in trip_events
            if e.scalar_features.model_extra
            and "responded" in e.scalar_features.model_extra
        ]
        if responses:
            trip_rate = sum(responses) / len(responses)
            prev = mem.baseline.alert_response_rate
            mem.baseline.alert_response_rate = (
                (1 - _EMA_ALPHA) * prev + _EMA_ALPHA * trip_rate
                if mem.trip_count > 0 else trip_rate
            )

        # -- 4. advance the state machine -----------------------------------
        mem.trip_count += 1
        mem.state = cls.state_for(mem.trip_count)

        # -- 5. append a retrievable memory snippet -------------------------
        mem.memory_snippets.append(cls._build_snippet(mem, trip_events, trip_id))

        mem.updated_at = datetime.now(timezone.utc)
        return mem

    # -- threshold suggestion (advisory) ------------------------------------
    @classmethod
    def suggest_threshold(
        cls, memory: DriverMemory, event_type: EventType
    ) -> tuple[float, float, bool, bool]:
        """
        Compute the advisory threshold for one driver + event type.

        Returns a tuple:
            (threshold, confidence, used_global_default, safety_floor_applied)

        Personalisation is fully worked out for FATIGUE_ALERT (we hold that
        baseline). Other event types honestly fall back to the global default,
        still safety-floored — we do not invent a personal number from data we
        do not have.
        """
        state = memory.state
        confidence = cls.confidence_for(memory.trip_count)
        global_default = GLOBAL_DEFAULTS.get(event_type, 1.0)

        # Consent opt-out or COLD_START -> pure global default.
        personal_ok = (
            memory.consent.personalization_enabled
            and state != PersonalizationState.COLD_START
            and event_type == EventType.FATIGUE_ALERT
        )
        if not personal_ok:
            floored, applied = cls._apply_safety_floor(
                event_type, global_default, memory
            )
            used_global = True
            # Confidence is meaningless when opted out.
            conf = 0.0 if not memory.consent.personalization_enabled else confidence
            return floored, conf, used_global, applied

        # Personal fatigue suggestion: warn a little later for a driver whose
        # normal PERCLOS runs high, but stay grounded in their own statistics.
        std = math.sqrt(memory.baseline.fatigue_var)
        personal = memory.baseline.fatigue_ema + _FATIGUE_MARGIN_STD * std

        w = cls.blend_weight(state)
        blended = (1 - w) * global_default + w * personal

        floored, applied = cls._apply_safety_floor(event_type, blended, memory)
        used_global = math.isclose(w, 0.0)
        return floored, confidence, used_global, applied

    # =======================================================================
    # PRIVATE HELPERS
    # =======================================================================
    @staticmethod
    def _update_ema_var(mem: DriverMemory, x: float) -> None:
        """
        Incremental EMA + EMA-of-variance update (a standard, explainable form).

            delta = x - ema
            ema  += alpha * delta
            var   = (1 - alpha) * (var + alpha * delta^2)

        On the very first sample ever (trip_count 0 and var 0 and ema 0) we
        seed the EMA to x so it does not crawl up from zero.
        """
        b = mem.baseline
        first_sample = mem.trip_count == 0 and b.fatigue_ema == 0.0 and b.fatigue_var == 0.0
        if first_sample:
            b.fatigue_ema = x
            b.fatigue_var = 0.0
            return
        delta = x - b.fatigue_ema
        b.fatigue_ema += _EMA_ALPHA * delta
        b.fatigue_var = (1 - _EMA_ALPHA) * (b.fatigue_var + _EMA_ALPHA * delta * delta)

    @staticmethod
    def _update_style_vector(mem: DriverMemory, embeddings: list[list[float]]) -> None:
        """Blend the mean of this trip's embeddings into the style vector (EMA)."""
        dim = len(embeddings[0])
        # Guard against ragged embeddings from upstream by ignoring odd lengths.
        clean = [e for e in embeddings if len(e) == dim]
        trip_mean = [sum(vals) / len(clean) for vals in zip(*clean)]

        current = mem.baseline.driving_style_vector
        if not current or len(current) != dim:
            mem.baseline.driving_style_vector = trip_mean
            return
        mem.baseline.driving_style_vector = [
            (1 - _EMA_ALPHA) * c + _EMA_ALPHA * t for c, t in zip(current, trip_mean)
        ]

    @classmethod
    def _build_snippet(
        cls, mem: DriverMemory, trip_events: list[TelemetryEvent], trip_id: str
    ) -> MemorySnippet:
        """
        Produce a short, grounded Vietnamese summary of the trip for retrieval.
        This is TEMPLATED (not LLM-generated) so a stored memory can never be a
        hallucination — the reasoning engine only retrieves facts, never fiction.
        """
        n_fatigue = sum(
            1 for e in trip_events if e.event_type == EventType.FATIGUE_ALERT
        )
        n_object = sum(
            1 for e in trip_events if e.event_type == EventType.OBJECT_RISK
        )
        parts = [f"Chuyến {trip_id}:"]
        if n_fatigue:
            parts.append(
                f"có {n_fatigue} cảnh báo mệt (PERCLOS trung bình ~{mem.baseline.fatigue_ema:.2f})."
            )
        if n_object:
            parts.append(f"có {n_object} tình huống rủi ro vật cản.")
        if not n_fatigue and not n_object:
            parts.append("không có sự kiện an toàn đáng chú ý.")
        summary = " ".join(parts)

        return MemorySnippet(
            trip_id=trip_id,
            summary_vi=summary,
            embedding=list(mem.baseline.driving_style_vector),
        )

    @staticmethod
    def _apply_safety_floor(
        event_type: EventType, value: float, memory: DriverMemory
    ) -> tuple[float, bool]:
        """
        Clamp a suggested threshold to V4's safety floor. Direction matters:

          * FATIGUE_ALERT / SURFACE_RISK: higher threshold = warn LATER = less
            safe, so we CAP it from above (max_fatigue_threshold for fatigue).
          * OBJECT_RISK (TTC): lower threshold = warn LATER = less safe, so we
            FLOOR it from below (min_ttc_threshold).

        Returns (clamped_value, was_clamped).
        """
        floor = memory.safety_floor
        if event_type == EventType.FATIGUE_ALERT:
            if value > floor.max_fatigue_threshold:
                return floor.max_fatigue_threshold, True
            return value, False
        if event_type == EventType.OBJECT_RISK:
            if value < floor.min_ttc_threshold:
                return floor.min_ttc_threshold, True
            return value, False
        # No explicit floor defined for LANE/SURFACE in the schema.
        return value, False


# ===========================================================================
# SECTION 3 — SELF-TEST (a mini "3 trips" walk-through)
# ===========================================================================
if __name__ == "__main__":
    sm = PersonalizationStateMachine
    mem = DriverMemory.new("driver_001")
    print("start:", mem.state.value, "trips", mem.trip_count)

    for i in range(1, 4):
        events = [
            TelemetryEvent(
                trip_id=f"trip_{i:03d}", driver_id="driver_001",
                source="V3_PERCEPTION", event_type="FATIGUE_ALERT",
                risk_embedding=[0.1 * i, 0.2, 0.3],
                scalar_features={"perclos": 0.30 + 0.05 * i},
            )
        ]
        mem = sm.apply_completed_trip(mem, events, f"trip_{i:03d}")
        thr, conf, glob, floored = sm.suggest_threshold(mem, EventType.FATIGUE_ALERT)
        print(
            f"after trip {i}: state={mem.state.value:12s} "
            f"ema={mem.baseline.fatigue_ema:.3f} thr={thr:.3f} "
            f"conf={conf:.2f} global={glob} floored={floored}"
        )
