"""
simulate_trips.py
=================

VERTICAL 1 — PERSONALIZATION ENGINE
Demo scenario (spec §6, week 3): a scripted walk through 3+ simulated trips that
shows the driver graduating COLD_START -> WARMING -> PERSONALIZED, with the
advisory threshold and the Vietnamese explanation changing at each stage.

It runs STANDALONE — no server, no model download, no network — using in-memory
stores and the deterministic fallback reasoning path. That is deliberate: the
demo must never depend on live driving or a 2 GB model being present (spec §8,
"chưa đủ dữ liệu 3 chuyến thật lúc demo" mitigation).

RUN (from personalization_engine/):
    python -m mock_data.simulate_trips
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import sys

# Force UTF-8 stdout so the Vietnamese output never trips the Windows console
# codepage (cp1252) — matters only when output is piped/redirected; harmless
# otherwise. Guarded so it never breaks on odd stream types.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ingestion.feature_store import InMemoryFeatureStore
from memory.store import InMemoryMemoryStore
from models.telemetry import EventType, TelemetryEvent
from reasoning.phi3_runtime import DisabledLLMBackend, Phi3Runtime
from reasoning.reasoning_engine import ReasoningEngine
from services.explanation_service import ExplanationService
from services.memory_service import DriverMemoryService

DRIVER = "driver_001"

# A scripted set of trips for one driver. This driver runs a LOW normal PERCLOS
# (calm, alert), so once personalised Guardian should warn them EARLIER than the
# global default — the whole point of personalisation.
_TRIPS = {
    "trip_001": [("FATIGUE_ALERT", {"perclos": 0.28}), ("FATIGUE_ALERT", {"perclos": 0.31})],
    "trip_002": [("FATIGUE_ALERT", {"perclos": 0.30}), ("OBJECT_RISK", {"ttc_seconds": 1.4, "object_class": "xe máy"})],
    "trip_003": [("FATIGUE_ALERT", {"perclos": 0.33}), ("FATIGUE_ALERT", {"perclos": 0.29})],
    "trip_004": [("FATIGUE_ALERT", {"perclos": 0.52})],  # a spike, now personalised
}


def _make_event(trip_id, etype, scalars):
    return TelemetryEvent(
        trip_id=trip_id, driver_id=DRIVER,
        source="V3_PERCEPTION", event_type=etype,
        risk_embedding=[0.2, 0.1, 0.3],
        scalar_features=scalars,
        safety_kernel_decision={"action": "WARN", "threshold_used": 0.70}
        if etype == "FATIGUE_ALERT"
        else {"action": "BRAKE_ASSIST", "threshold_used": 2.0},
    )


def main() -> None:
    features = InMemoryFeatureStore()
    memory = InMemoryMemoryStore()

    # Use the real Phi-3 runtime if a model happens to be installed; otherwise
    # the deterministic fallback. Either way the demo runs.
    phi3 = Phi3Runtime()
    backend = phi3 if phi3.is_available() else DisabledLLMBackend()
    engine = ReasoningEngine(backend=backend)

    mem_svc = DriverMemoryService(features, memory)
    exp_svc = ExplanationService(features, memory, engine)

    mode = "Phi-3 ONNX" if engine.backend_available() else "fallback template (deterministic)"
    print("=" * 78)
    print(f"GUARDIAN · Vertical 1 — Personalization demo   [reasoning: {mode}]")
    print("=" * 78)

    for trip_id, events in _TRIPS.items():
        print(f"\n──────── {trip_id} ────────")
        last_event_id = None
        for etype, scalars in events:
            evt = _make_event(trip_id, etype, scalars)
            mem_svc.ingest_event(evt)
            last_event_id = evt.event_id
            reading = ", ".join(f"{k}={v}" for k, v in scalars.items())
            print(f"  · sự kiện {etype:<14} ({reading})")

        # End the trip -> Learning Loop runs, state may advance.
        mem = mem_svc.end_trip(DRIVER, trip_id)
        thr = mem_svc.get_personalized_threshold(DRIVER, EventType.FATIGUE_ALERT)

        print(f"  → Trạng thái cá nhân hoá : {mem.state.value}  (đã học {mem.trip_count} chuyến)")
        print(f"  → Ngưỡng mệt (advisory)  : {thr.threshold:.3f}  "
              f"(global_default={thr.used_global_default}, "
              f"confidence={thr.confidence:.2f}, floored={thr.safety_floor_applied})")

        # Explain the last event of the trip, in Vietnamese.
        resp = exp_svc.explain_event(last_event_id)
        tag = "fallback" if resp.used_fallback_template else "Phi-3"
        print(f"  → Giải thích [{tag}] : {resp.text_vi}")

    # -- transparency / consent demo ---------------------------------------
    print("\n──────── Minh bạch & quyền tài xế ────────")
    summary = mem_svc.get_memory(DRIVER)
    print(f"  · Bộ nhớ đang lưu {len(summary.memory_snippets)} tóm tắt chuyến, "
          f"PERCLOS nền ~{summary.baseline.fatigue_ema:.2f}.")

    mem_svc.set_consent(DRIVER, enabled=False)
    thr_off = mem_svc.get_personalized_threshold(DRIVER, EventType.FATIGUE_ALERT)
    print(f"  · Tài xế TẮT cá nhân hoá → ngưỡng quay về mặc định "
          f"({thr_off.threshold:.3f}, global={thr_off.used_global_default}).")

    mem_svc.set_consent(DRIVER, enabled=True)
    reset = mem_svc.reset_memory(DRIVER)
    print(f"  · Tài xế XOÁ bộ nhớ → trạng thái {reset.state.value}, "
          f"đã học {reset.trip_count} chuyến (reset_at={reset.consent.last_reset_at:%Y-%m-%d %H:%M}).")

    print("\nHoàn tất. V1 chỉ TƯ VẤN — mọi lệnh an toàn vẫn do V4 (deterministic) quyết định.")


if __name__ == "__main__":
    main()
