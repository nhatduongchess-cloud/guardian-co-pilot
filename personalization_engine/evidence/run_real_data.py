"""
run_real_data.py
================

VERTICAL 1 — EVIDENCE · run V1 over the REAL evaluation trips.

This is the "chạy lại trên data thật" run. It feeds Vertical 1 with signals
that genuinely exist in the redacted evaluation set:

  * FATIGUE   — drowsiness probability from OUR Challenge-2 model looking at the
                REAL driver footage (see evidence/perceive_driver.py). Not the
                redacted ground-truth; our own perception output.
  * OBJECT    — the REAL scenario events from each trip's `events_log`
                (motorcycle cut-in, lead brake, pedestrian jaywalk, ...).
  * BRAKE     — harsh-deceleration episodes derived from the REAL per-frame ego
                kinematics.

The 10 recorded sessions are fed sequentially as ONE monitored driver's history,
so the state machine can graduate COLD_START → PERSONALIZED on real data and the
fatigue threshold personalises against the driver's real, model-observed
drowsiness — always clamped by the V4 safety floor.

WHAT IS REDACTED (and therefore NOT used): per-frame driver_state, min_ttc and
risk scores — exactly the targets teams must predict. We reconstruct the driver
signal with our own model and take object risk as discrete events, honestly.

RUN (from personalization_engine/):
    python -m evidence.perceive_driver      # once: writes driver_state_realdata.json
    python -m evidence.run_real_data        # this file
    python -m evidence.run_real_data --json # writes evidence/report_realdata.json
"""

import argparse
import gzip
import json
import os
import statistics
import sys
import time
import zipfile
from pathlib import Path

# Force UTF-8 stdout so Vietnamese output survives a piped/redirected Windows
# console (cp1252). Harmless when writing straight to a terminal window.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ingestion.feature_store import InMemoryFeatureStore
from memory.state_machine import PersonalizationStateMachine as SM
from memory.store import InMemoryMemoryStore
from models.telemetry import EventType, TelemetryEvent
from reasoning.phi3_runtime import DisabledLLMBackend, Phi3Runtime
from reasoning.reasoning_engine import ReasoningEngine
from services.explanation_service import ExplanationService
from services.memory_service import DriverMemoryService

_DEFAULT_ZIP = os.getenv(
    "GUARDIAN_DATASET_ZIP",
    str(Path.home() / "Downloads" / "Hackathon_Dataset_Redacted.zip"),
)
_PERCEPTION = Path(__file__).with_name("driver_state_realdata.json")
_OUT = Path(__file__).with_name("report_realdata.json")
_DRIVER = "driver_carla"

# events_log type -> (V1 event_type, object label in Vietnamese, kernel action)
_EVENT_MAP = {
    "motorcycle_cut_in": ("OBJECT_RISK", "xe máy tạt đầu", "WARN"),
    "lead_brake": ("OBJECT_RISK", "xe phía trước phanh gấp", "BRAKE_ASSIST"),
    "pedestrian_jaywalk": ("OBJECT_RISK", "người đi bộ băng qua đường", "BRAKE_ASSIST"),
    "stopped_vehicle_ahead": ("OBJECT_RISK", "xe dừng phía trước", "WARN"),
}


def _load_trip_json(zf: zipfile.ZipFile, tid: str) -> dict:
    raw = zf.read(f"Hackathon_Dataset_Redacted/{tid}/{tid}.json.gz")
    return json.loads(gzip.decompress(raw).decode("utf-8"))


def _harsh_brakes(frames: list, thresh: float = -8.0, min_gap_s: float = 2.0) -> list:
    """Return timestamps of distinct harsh-deceleration episodes (real kinematics)."""
    episodes, last_t = [], -1e9
    for fr in frames:
        acc = fr.get("ego", {}).get("longitudinal_accel", 0.0)
        t = fr.get("timestamp", 0.0)
        if acc <= thresh and (t - last_t) >= min_gap_s:
            episodes.append((t, acc))
            last_t = t
    return episodes


def _build_events(tid: str, doc: dict, perception: dict) -> list[TelemetryEvent]:
    """Turn one real trip into a stream of grounded TelemetryEvents."""
    frames = doc["frames"]
    meta = doc.get("metadata", {})
    dur = meta.get("duration_sec", 90) or 90
    base_ms = 0
    events: list[TelemetryEvent] = []

    # -- FATIGUE from our perception model's real drowsiness series ----------
    series = perception["trips"][tid]["drowsiness_series"]
    # ego speed/targets for a real risk embedding
    speeds = [fr.get("ego", {}).get("speed_kmh", 0.0) for fr in frames]
    max_speed = max(speeds) or 1.0
    for i, drow in enumerate(series):
        ts = int((i / max(1, len(series) - 1)) * dur * 1000)
        events.append(TelemetryEvent(
            trip_id=tid, driver_id=_DRIVER, timestamp_ms=ts,
            source="V3_PERCEPTION", event_type="FATIGUE_ALERT",
            risk_embedding=[drow, min(1.0, max_speed / 120.0), 0.0],
            scalar_features={"perclos": drow},
        ))

    # -- OBJECT risk from the REAL events_log -------------------------------
    for ev in doc.get("events_log", []):
        mapped = _EVENT_MAP.get(ev["type"])
        if not mapped:
            continue
        etype, label, action = mapped
        events.append(TelemetryEvent(
            trip_id=tid, driver_id=_DRIVER, timestamp_ms=int(ev["t"] * 1000),
            source="V4_WORLD_MODEL", event_type=etype,
            risk_embedding=[0.0, min(1.0, max_speed / 120.0), 1.0],
            scalar_features={"object_class": label},
            safety_kernel_decision={"action": action, "threshold_used": 2.0},
        ))

    # -- BRAKE episodes from REAL ego kinematics ----------------------------
    for t, acc in _harsh_brakes(frames):
        events.append(TelemetryEvent(
            trip_id=tid, driver_id=_DRIVER, timestamp_ms=int(t * 1000),
            source="V4_WORLD_MODEL", event_type="SAFETY_DECISION",
            risk_embedding=[0.0, min(1.0, max_speed / 120.0), 1.0],
            scalar_features={},
            safety_kernel_decision={"action": "BRAKE_ASSIST", "threshold_used": 2.0},
        ))

    events.sort(key=lambda e: e.timestamp_ms)
    return events


def run(as_json: bool = False) -> dict:
    zip_path = Path(_DEFAULT_ZIP)
    if not zip_path.exists():
        raise SystemExit(f"Dataset zip not found: {zip_path} (set GUARDIAN_DATASET_ZIP).")
    if not _PERCEPTION.exists():
        raise SystemExit("Run `python -m evidence.perceive_driver` first "
                         "(missing driver_state_realdata.json).")

    perception = json.loads(_PERCEPTION.read_text(encoding="utf-8"))
    zf = zipfile.ZipFile(zip_path)
    trip_ids = sorted(perception["trips"].keys())

    features = InMemoryFeatureStore()
    memory = InMemoryMemoryStore()
    phi3 = Phi3Runtime()
    backend = phi3 if phi3.is_available() else DisabledLLMBackend()
    engine = ReasoningEngine(backend=backend)
    mem_svc = DriverMemoryService(features, memory)
    exp_svc = ExplanationService(features, memory, engine)

    report = {
        "source": zip_path.name,
        "perception_model": perception.get("model"),
        "reasoning_backend": "phi3-onnx" if engine.backend_available() else "deterministic-fallback",
        "driver": _DRIVER,
        "trips": [],
        "totals": {"events": 0, "by_type": {}},
    }
    lat_ms: list[float] = []

    for tid in trip_ids:
        doc = _load_trip_json(zf, tid)
        meta = doc.get("metadata", {})
        p = perception["trips"][tid]

        events = _build_events(tid, doc, perception)
        for e in events:
            mem_svc.ingest_event(e)
            report["totals"]["by_type"][e.event_type.value] = \
                report["totals"]["by_type"].get(e.event_type.value, 0) + 1
        report["totals"]["events"] += len(events)

        mem = mem_svc.end_trip(_DRIVER, tid)
        thr = mem_svc.get_personalized_threshold(_DRIVER, EventType.FATIGUE_ALERT)

        # Pick a representative event to explain: the real object event if any,
        # else the drowsiest fatigue event of the trip.
        object_events = [e for e in events if e.event_type == EventType.OBJECT_RISK]
        if object_events:
            headline = object_events[0]
        else:
            headline = max((e for e in events if e.event_type == EventType.FATIGUE_ALERT),
                           key=lambda e: e.scalar_features.perclos or 0.0)
        t0 = time.perf_counter()
        expl = exp_svc.explain_event(headline.event_id)
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

        report["trips"].append({
            "trip_id": tid,
            "scenario": meta.get("description", "").replace("DEBUG 30s: ", ""),
            "driver_profile": meta.get("driver_profile"),
            "detected_states": p["class_distribution"],
            "drowsiness_mean": p["drowsiness_mean"],
            "drowsiness_max": p["drowsiness_max"],
            "real_events": [e["type"] for e in doc.get("events_log", [])],
            "n_events_ingested": len(events),
            # V1 outputs:
            "state": mem.state.value,
            "trip_count": mem.trip_count,
            "fatigue_threshold": round(thr.threshold, 3),
            "confidence": round(thr.confidence, 2),
            "used_global_default": thr.used_global_default,
            "safety_floor_hit": thr.safety_floor_applied,
            "headline_event": headline.event_type.value,
            "explanation_vi": expl.text_vi,
            "used_fallback": expl.used_fallback_template,
        })

    report["latency_explain_ms"] = {
        "p50": round(statistics.median(lat_ms), 3),
        "max": round(max(lat_ms), 3),
        "n": len(lat_ms),
    }
    # grounding: no explanation may contain a number absent from its facts —
    # checked implicitly by the engine gate; here we just confirm none is empty.
    report["explanations_nonempty"] = all(t["explanation_vi"].strip() for t in report["trips"])

    if as_json:
        _OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {_OUT}")
    else:
        _print(report)
    return report


def _print(r: dict) -> None:
    print("=" * 92)
    print(f"VERTICAL 1 on REAL evaluation data  ·  {r['source']}  ·  "
          f"perception={r['perception_model']}  ·  reasoning={r['reasoning_backend']}")
    print("=" * 92)
    for t in r["trips"]:
        cap = " ⚠FLOOR" if t["safety_floor_hit"] else ""
        print(f"\n{t['trip_id']} — {t['scenario'][:58]}")
        print(f"   perception: {t['detected_states']}  drowsiness≈{t['drowsiness_mean']:.2f}"
              f"   real events: {t['real_events']}")
        print(f"   → V1: {t['state']:12s} (trip {t['trip_count']})  "
              f"ngưỡng mệt={t['fatigue_threshold']:.3f} conf={t['confidence']:.2f} "
              f"global={t['used_global_default']}{cap}  [{t['n_events_ingested']} events]")
        print(f"   → giải thích: {t['explanation_vi']}")
    tot = r["totals"]
    print("\n" + "-" * 92)
    print(f"Tổng: {tot['events']} sự kiện thật từ {len(r['trips'])} chuyến  ·  {tot['by_type']}")
    print(f"Giải thích p50={r['latency_explain_ms']['p50']} ms  ·  "
          f"tất cả không rỗng={r['explanations_nonempty']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    run(as_json=args.json)
