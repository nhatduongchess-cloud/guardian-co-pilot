"""
generate_evidence.py
====================

VERTICAL 1 — PERSONALIZATION ENGINE
"Bằng chứng bằng số liệu" — a reproducible measurement harness that produces
REAL numbers from the running system, for showing a reviewer.

It measures four things a jury of ADAS engineers actually cares about:

  1. LATENCY (edge-viable?)      — p50/p95 of threshold, explanation, API.
  2. PERSONALISATION (does it adapt?) — a synthetic driver cohort, quantified.
  3. SAFETY (can it hallucinate?)     — grounding-gate rejection rate.
  4. DETERMINISM (reproducible?)      — same input -> byte-identical output.

Everything runs offline (in-memory stores, deterministic fallback), so the
numbers are stable and can be re-generated live in front of the jury.

RUN (from personalization_engine/):
    python -m evidence.generate_evidence          # human report
    python -m evidence.generate_evidence --json   # writes evidence/report.json
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import argparse
import json
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from api.memory_routes import (
    get_feature_store,
    get_memory_store,
    get_reasoning_engine,
)
from ingestion.feature_store import InMemoryFeatureStore
from main import app
from memory.state_machine import GLOBAL_DEFAULTS
from memory.store import InMemoryMemoryStore
from models.explanation import ExplanationContext
from models.telemetry import EventType, TelemetryEvent
from reasoning.phi3_runtime import DisabledLLMBackend, LLMBackend
from reasoning.reasoning_engine import ReasoningEngine
from services.memory_service import DriverMemoryService

_GLOBAL_FATIGUE = GLOBAL_DEFAULTS[EventType.FATIGUE_ALERT]


def _pct(values: list[float], p: float) -> float:
    """Return the p-th percentile (0..100) of a list, in the list's units."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ===========================================================================
# 1) LATENCY — is it edge-viable?
# ===========================================================================
def measure_latency(n_math=500, n_llm=300, n_api=200) -> dict:
    features = InMemoryFeatureStore()
    memory = InMemoryMemoryStore()
    svc = DriverMemoryService(features, memory)
    engine = ReasoningEngine(backend=DisabledLLMBackend())

    # Warm one personalised driver so the threshold path does real work.
    for i, p in enumerate([0.30, 0.31, 0.30], start=1):
        evt = TelemetryEvent(
            trip_id=f"t{i}", driver_id="driver_001", source="V3_PERCEPTION",
            event_type="FATIGUE_ALERT", risk_embedding=[0.1, 0.2, 0.3],
            scalar_features={"perclos": p},
        )
        svc.ingest_event(evt)
        svc.end_trip("driver_001", f"t{i}")

    ctx = ExplanationContext(
        event_id="e", event_type="FATIGUE_ALERT",
        situation_summary="PERCLOS 0.42, cao hơn mức bình thường của bạn.",
        decision_taken="Guardian đã phát cảnh báo để nhắc bạn.",
        driver_memory_snippet="Chuyến trước bạn cũng mệt.",
        personalization_state="PERSONALIZED",
    )

    # -- threshold math latency --
    th = []
    for _ in range(n_math):
        t0 = time.perf_counter()
        svc.get_personalized_threshold("driver_001", EventType.FATIGUE_ALERT)
        th.append((time.perf_counter() - t0) * 1000.0)

    # -- explanation (fallback) latency --
    ex = []
    for _ in range(n_llm):
        t0 = time.perf_counter()
        engine.explain(ctx)
        ex.append((time.perf_counter() - t0) * 1000.0)

    # -- full API round-trip latency (ingest + threshold + explain) --
    app.dependency_overrides[get_feature_store] = lambda: features
    app.dependency_overrides[get_memory_store] = lambda: memory
    app.dependency_overrides[get_reasoning_engine] = lambda: engine
    api = []
    with TestClient(app) as client:
        body = {
            "trip_id": "live", "driver_id": "driver_001", "source": "V3_PERCEPTION",
            "event_type": "FATIGUE_ALERT", "scalar_features": {"perclos": 0.42},
        }
        for _ in range(n_api):
            t0 = time.perf_counter()
            r = client.post("/api/v1/telemetry", json=body)
            eid = r.json()["event_id"]
            client.get("/api/v1/drivers/driver_001/threshold",
                       params={"event_type": "FATIGUE_ALERT"})
            client.get(f"/api/v1/explanations/{eid}")
            api.append((time.perf_counter() - t0) * 1000.0)
    app.dependency_overrides.clear()

    def summ(v):
        return {"p50_ms": round(_pct(v, 50), 3), "p95_ms": round(_pct(v, 95), 3),
                "mean_ms": round(statistics.mean(v), 3), "n": len(v)}

    return {
        "threshold_compute": summ(th),
        "explanation_fallback": summ(ex),
        "api_roundtrip_3calls": summ(api),
        "note": "Explanation runs ASYNC, off V4's <=100ms Fast Path.",
    }


# ===========================================================================
# 2) PERSONALISATION — does it actually adapt per driver?
# ===========================================================================
def measure_personalization() -> dict:
    # Four archetypes with different NORMAL fatigue levels. The last is an
    # extreme (near-asleep) baseline included specifically to prove the V4
    # safety-floor clamp engages — realistic drivers never reach it.
    cohort = {
        "calm_driver": [0.24, 0.26, 0.23, 0.25],       # alert; warn earlier
        "average_driver": [0.44, 0.46, 0.45, 0.43],
        "drowsy_prone": [0.64, 0.66, 0.68, 0.67],       # near own baseline, still safe
        "extreme_case": [0.86, 0.88, 0.90, 0.89],       # triggers the safety floor
    }
    results = []
    for name, perclos_seq in cohort.items():
        features = InMemoryFeatureStore()
        memory = InMemoryMemoryStore()
        svc = DriverMemoryService(features, memory)

        trail = []
        for i, p in enumerate(perclos_seq, start=1):
            trip = f"trip_{i}"
            svc.ingest_event(TelemetryEvent(
                trip_id=trip, driver_id=name, source="V3_PERCEPTION",
                event_type="FATIGUE_ALERT", risk_embedding=[0.2, 0.1, 0.3],
                scalar_features={"perclos": p},
            ))
            mem = svc.end_trip(name, trip)
            thr = svc.get_personalized_threshold(name, EventType.FATIGUE_ALERT)
            trail.append({
                "trip": i, "state": mem.state.value,
                "threshold": round(thr.threshold, 3),
                "confidence": round(thr.confidence, 2),
                "used_global": thr.used_global_default,
                "floored": thr.safety_floor_applied,
            })

        final = trail[-1]
        delta_pct = round(100 * (final["threshold"] - _GLOBAL_FATIGUE) / _GLOBAL_FATIGUE, 1)
        results.append({
            "driver": name,
            "normal_perclos": round(statistics.mean(perclos_seq), 3),
            "final_state": final["state"],
            "final_threshold": final["threshold"],
            "global_default": _GLOBAL_FATIGUE,
            "delta_vs_global_pct": delta_pct,
            "safety_floor_hit": final["floored"],
            "trail": trail,
        })
    return {
        "global_default": _GLOBAL_FATIGUE,
        "drivers": results,
        "claim": "Each driver's threshold moves toward their own baseline (calm "
                 "drivers get earlier warnings); an extreme baseline is capped by "
                 "the V4 safety floor (floored=True), so V1 can never suggest warning "
                 "dangerously late.",
    }


# ===========================================================================
# 3) SAFETY — can the reasoning engine be made to hallucinate?
# ===========================================================================
class _HallucinatingBackend(LLMBackend):
    """Always returns text with a fabricated number absent from the facts."""
    def is_available(self) -> bool:
        return True

    def generate(self, prompt, max_new_tokens=96, temperature=0.3):
        return "Guardian phát hiện PERCLOS 9.99 nên đã DỪNG XE khẩn cấp ngay lập tức."


class _GoodBackend(LLMBackend):
    """Returns a grounded rephrasing (uses only the allowed number 0.42)."""
    def is_available(self) -> bool:
        return True

    def generate(self, prompt, max_new_tokens=96, temperature=0.3):
        return "Mắt bạn nhắm nhiều hơn bình thường (PERCLOS 0.42) nên Guardian nhắc bạn nghỉ."


def measure_safety(n=200) -> dict:
    ctx = ExplanationContext(
        event_id="e", event_type="FATIGUE_ALERT",
        situation_summary="PERCLOS 0.42, cao hơn mức bình thường của bạn.",
        decision_taken="Guardian đã phát cảnh báo để nhắc bạn.",
        driver_memory_snippet="Chuyến trước bạn cũng mệt.",
        personalization_state="PERSONALIZED",
    )

    hallu = ReasoningEngine(backend=_HallucinatingBackend())
    good = ReasoningEngine(backend=_GoodBackend())

    rejected = 0
    fabricated_leaked = 0
    empty = 0
    for _ in range(n):
        r = hallu.explain(ctx)
        if r.used_fallback_template:      # gate caught it -> fell back
            rejected += 1
        if "9.99" in r.text_vi:            # fabricated number reached output?
            fabricated_leaked += 1
        if not r.text_vi.strip():
            empty += 1

    accepted_good = sum(
        1 for _ in range(n) if not good.explain(ctx).used_fallback_template
    )

    return {
        "hallucination_trials": n,
        "hallucination_rejected": rejected,
        "hallucination_rejection_rate_pct": round(100 * rejected / n, 1),
        "fabricated_number_leaked": fabricated_leaked,
        "empty_responses": empty,
        "grounded_llm_accepted": accepted_good,
        "grounded_llm_accept_rate_pct": round(100 * accepted_good / n, 1),
        "claim": "A fabricated safety number ('9.99') is rejected 100% of the time "
                 "and never reaches the driver; grounded rephrasings are accepted.",
    }


# ===========================================================================
# 4) DETERMINISM — same input, identical output (reproducible for safety)
# ===========================================================================
def measure_determinism(n=50) -> dict:
    ctx = ExplanationContext(
        event_id="e", event_type="OBJECT_RISK",
        situation_summary="Thời gian tới va chạm chỉ còn 1.1 giây với xe máy.",
        decision_taken="Guardian đã hỗ trợ phanh để giữ khoảng cách an toàn.",
        driver_memory_snippet="",
        personalization_state="COLD_START",
    )
    engine = ReasoningEngine(backend=DisabledLLMBackend())
    outputs = {engine.explain(ctx).text_vi for _ in range(n)}
    return {
        "runs": n,
        "distinct_outputs": len(outputs),
        "deterministic": len(outputs) == 1,
        "claim": "The fallback path is byte-for-byte reproducible across runs.",
    }


# ===========================================================================
# ORCHESTRATION + REPORTING
# ===========================================================================
def collect() -> dict:
    engine = ReasoningEngine()
    return {
        "reasoning_backend": "phi3-onnx" if engine.backend_available()
                             else "deterministic-fallback",
        "latency": measure_latency(),
        "personalization": measure_personalization(),
        "safety": measure_safety(),
        "determinism": measure_determinism(),
    }


def _print_report(data: dict) -> None:
    lat = data["latency"]
    print("=" * 74)
    print(f"VERTICAL 1 — EVIDENCE  ·  reasoning backend = {data['reasoning_backend']}")
    print("=" * 74)

    print("\n[1] LATENCY (edge-viability)")
    for k in ("threshold_compute", "explanation_fallback", "api_roundtrip_3calls"):
        s = lat[k]
        print(f"    {k:<24} p50={s['p50_ms']:.3f} ms  p95={s['p95_ms']:.3f} ms  (n={s['n']})")

    print("\n[2] PERSONALISATION (adapts per driver; global default = "
          f"{data['personalization']['global_default']})")
    for d in data["personalization"]["drivers"]:
        cap = " · SAFETY-FLOOR HIT" if d["safety_floor_hit"] else ""
        print(f"    {d['driver']:<15} normal PERCLOS={d['normal_perclos']:.2f} "
              f"-> ngưỡng {d['final_threshold']:.3f} "
              f"({d['delta_vs_global_pct']:+.1f}% vs global) [{d['final_state']}]{cap}")

    saf = data["safety"]
    print("\n[3] SAFETY (anti-hallucination grounding gate)")
    print(f"    Hallucinated safety number rejected: "
          f"{saf['hallucination_rejected']}/{saf['hallucination_trials']} "
          f"({saf['hallucination_rejection_rate_pct']}%)")
    print(f"    Fabricated number leaked to driver : {saf['fabricated_number_leaked']}")
    print(f"    Empty responses                    : {saf['empty_responses']}")
    print(f"    Grounded LLM answers accepted      : {saf['grounded_llm_accept_rate_pct']}%")

    det = data["determinism"]
    print("\n[4] DETERMINISM (reproducible)")
    print(f"    {det['runs']} runs -> {det['distinct_outputs']} distinct output "
          f"(deterministic={det['deterministic']})")
    print("-" * 74)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true",
                        help="write evidence/report.json instead of printing")
    args = parser.parse_args()

    data = collect()
    if args.json:
        out = Path(__file__).with_name("report.json")
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    else:
        _print_report(data)
