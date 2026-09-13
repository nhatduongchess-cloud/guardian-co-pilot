"""
pipeline.py
===========

GUARDIAN CO-PILOT — END-TO-END DECISION CHAIN ON REAL DATA

Wires the whole architecture together and runs it over a dataset trip:

    Perception (C1) ─┐
                     ├─► World Model ─► Prediction ─► Planning ─► Safety Kernel
    Driver DMS (C2) ─┘                                                  │
                                                                        ▼
                                                            Verified Command

Everything before this file was validated on hand-made scenarios. Here the
chain meets real CARLA trips, and we answer the question the whole Guardian
thesis rests on:

    Does making the intervention threshold depend on the DRIVER actually warn
    earlier than a classic fixed-TTC AEB — without inventing false alarms?

THE COMPARISON
--------------
    Fixed AEB   brake when TTC < 1.2 s. One threshold for everyone. This is
                what most production AEB does and what the proposal calls out
                as the problem.
    Guardian    the full chain: driver state -> reaction time -> safety margin
                -> cost-based plan -> kernel verification.

Both are measured against the SAME ground truth on the SAME frames, using
event-based accounting (a contiguous hazard is one event, not 40 frames), which
matches how the CarSky AEB test bench scores braking.
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from guardian.challenge1.perception import CLASS_GAP_M, load_trip_perception
from guardian.challenge1.ttc import (
    TtcConfig,
    build_tracks,
    closing_speeds,
    compute_trip_ttc,
    prepare_detections,
)
from guardian.challenge2.classifier import build_temporal_features
from guardian.challenge2.features import load_trip_features
from guardian.challenge2.rules import RuleEngine
from guardian.decision.planner import PlanningEngine
from guardian.explain import explain
from guardian.decision.safety_kernel import SafetyKernel, VerifiedCommand
from guardian.world_model.state import (
    ContextState,
    DriverState,
    SceneState,
    TrackedObject,
    VehicleState,
    WorldModel,
)

# --- dataset toolkit ---------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STARTERKIT = _PROJECT_ROOT / "starterkit"
if str(_STARTERKIT) not in sys.path:
    sys.path.insert(0, str(_STARTERKIT))
from team_kit.dataset_loader import TripDataset  # noqa: E402


DEFAULT_DATA_ROOT = Path(os.getenv("GUARDIAN_DATA_ROOT", r"C:/guardian_data"))
PRACTICE_TRIP_IDS = tuple(f"T0{i}-Sample" for i in range(1, 7))

#: Fixed-threshold AEB baseline: the "one threshold for everyone" design.
FIXED_AEB_TTC_S = 1.2

#: A ground-truth hazard is a frame whose true TTC is below this.
GT_HAZARD_TTC_S = 3.0


# ===========================================================================
# SECTION 1 — BUILDING THE WORLD-MODEL STREAM
# ===========================================================================


def _context_from_metadata(metadata: dict) -> ContextState:
    """Translate CARLA weather metadata into Guardian's context state."""
    weather = metadata.get("weather", {}) or {}
    wetness = float(weather.get("wetness", 0.0))
    precipitation = float(weather.get("precipitation", 0.0))
    sun_altitude = float(weather.get("sun_altitude_angle", 45.0))

    # Surface: either standing water or active rain makes the road wet.
    surface = "wet" if (wetness > 10.0 or precipitation > 10.0) else "dry"

    return ContextState(
        speed_limit_kmh=float(metadata.get("speed_limit_kmh", 50.0)),
        surface=surface,
        weather="rain" if precipitation > 10.0 else "clear",
        is_night=sun_altitude < 0.0,
    )


def build_world_stream(
    trip_id: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    config: Optional[TtcConfig] = None,
) -> list[WorldModel]:
    """
    Assemble one WorldModel per frame from the cached perception + driver features.

    This is the only place that knows how the dataset is laid out. On CarSky the
    same WorldModel would be filled from KUKSA/CAN signals instead, and every
    stage downstream would be unchanged — which is the point of having it.
    """
    config = config or TtcConfig()
    trip = TripDataset(Path(data_root) / trip_id)
    frames = list(trip.iter_frames())
    n_frames = len(frames)
    ego_speeds_mps = np.array([f.speed_kmh for f in frames], dtype=float) / 3.6

    # --- Challenge 1: obstacles, with per-frame closing speed ---------------
    perception = load_trip_perception(trip_id)
    detections = prepare_detections(perception, config)
    objects_by_frame: dict[int, list[TrackedObject]] = defaultdict(list)

    if not detections.empty:
        for track in build_tracks(detections, config):
            speeds = closing_speeds(track, config)
            gap = CLASS_GAP_M.get(track.target_class, 0.0)
            for i, frame_id in enumerate(track.frames):
                if frame_id >= n_frames:
                    continue
                speed = speeds[i]
                if not math.isfinite(speed):
                    # Track too young: assume a stationary obstacle, so the
                    # closing speed is our own. Correct and conservative.
                    speed = ego_speeds_mps[frame_id]

                distance = track.distances[i]
                effective = distance - gap
                ttc = (effective / speed
                       if speed > config.min_closing_speed and effective > 0
                       else math.inf)

                objects_by_frame[frame_id].append(TrackedObject(
                    object_id=track.track_id,
                    object_class=track.target_class,
                    distance_m=distance,
                    lateral_m=track.laterals[i],
                    closing_speed_mps=speed,
                    ttc_s=ttc,
                ))

    # The authoritative per-frame TTC is the smoothed series we submit for
    # Challenge 1 — the decision chain must reason about the same numbers the
    # benchmark scores.
    ttc_series = compute_trip_ttc(
        trip_id, perception=perception, config=config,
        ego_speeds=ego_speeds_mps, n_frames=n_frames,
    )

    # --- Challenge 2: driver state -----------------------------------------
    temporal = build_temporal_features(load_trip_features(trip_id))
    engine = RuleEngine()
    states = engine.predict(temporal)

    # --- assemble -----------------------------------------------------------
    context = _context_from_metadata(trip.metadata)
    stream: list[WorldModel] = []
    for i, frame in enumerate(frames):
        row = temporal.iloc[i]
        driver = DriverState(
            state=str(states[i]),
            perclos=float(row["perclos_w201"]),
            eye_closure=float(row["eb_mean_w201"]),
            longest_closure_s=float(row["closed_run_sec"]),
            phone_rate=float(row["phone_rate_w61"]),
            explanation=engine.explain(temporal, i),
        )
        stream.append(WorldModel(
            frame_id=int(frame.frame_id),
            timestamp=float(frame.timestamp),
            scene=SceneState(
                objects=objects_by_frame.get(int(frame.frame_id), []),
                min_ttc_s=float(ttc_series[i]),
            ),
            driver=driver,
            vehicle=VehicleState(
                speed_kmh=float(frame.speed_kmh),
                longitudinal_accel=float(frame.longitudinal_accel),
                lateral_accel=float(frame.lateral_accel),
            ),
            context=context,
        ))
    return stream


# ===========================================================================
# SECTION 2 — RUNNING THE DECISION CHAIN
# ===========================================================================


def run_chain(stream: list[WorldModel]) -> list[VerifiedCommand]:
    """Plan and verify every frame, carrying kernel state across the trip."""
    planner = PlanningEngine()
    kernel = SafetyKernel()
    kernel.reset()

    commands: list[VerifiedCommand] = []
    previous: Optional[WorldModel] = None
    for world in stream:
        candidate = planner.plan(world, previous)
        commands.append(kernel.verify(candidate, world))
        previous = world
    return commands


def fixed_aeb(stream: list[WorldModel], threshold_s: float = FIXED_AEB_TTC_S) -> np.ndarray:
    """
    The classic baseline: brake hard when TTC drops below a fixed threshold.

    No driver awareness, no cost trade-off — exactly the "one threshold for
    everyone" behaviour Guardian argues against.
    """
    return np.array(
        [100.0 if w.scene.min_ttc_s < threshold_s else 0.0 for w in stream],
        dtype=float,
    )


# ===========================================================================
# SECTION 3 — EVENT-BASED COMPARISON
# ===========================================================================


def _events(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs in a boolean mask, as (start, end) frame indices."""
    events: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(mask) - 1))
    return events


@dataclass
class TripComparison:
    """Head-to-head result for one trip."""

    trip_id: str
    n_frames: int
    gt_events: int = 0
    guardian_covered: int = 0
    fixed_covered: int = 0
    guardian_lead_s: list[float] = field(default_factory=list)
    fixed_lead_s: list[float] = field(default_factory=list)
    # Interventions outside a ground-truth hazard window, split by whether an
    # object was really there. Lumping these together hides the difference
    # between "warned early about a real car" and "braked at nothing".
    guardian_early_events: int = 0
    guardian_phantom_events: int = 0
    fixed_early_events: int = 0
    fixed_phantom_events: int = 0
    driver_states: dict[str, int] = field(default_factory=dict)

    @property
    def guardian_false_events(self) -> int:
        return self.guardian_early_events + self.guardian_phantom_events

    @property
    def fixed_false_events(self) -> int:
        return self.fixed_early_events + self.fixed_phantom_events

    @property
    def mean_guardian_lead(self) -> float:
        return float(np.mean(self.guardian_lead_s)) if self.guardian_lead_s else 0.0

    @property
    def mean_fixed_lead(self) -> float:
        return float(np.mean(self.fixed_lead_s)) if self.fixed_lead_s else 0.0


def compare_trip(
    trip_id: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> TripComparison:
    """
    Run both systems over one trip and account for them event by event.

    For each ground-truth hazard event we ask: did the system intervene at any
    point during it, and how early relative to the moment the hazard began?
    Interventions with no ground-truth hazard anywhere nearby are false alarms.
    """
    trip = TripDataset(Path(data_root) / trip_id)
    frames = list(trip.iter_frames())
    gt_ttc = np.array([f.min_ttc for f in frames], dtype=float)
    timestamps = np.array([f.timestamp for f in frames], dtype=float)

    stream = build_world_stream(trip_id, data_root)
    guardian_brake = np.array([c.brake_pct for c in run_chain(stream)], dtype=float)
    fixed_brake = fixed_aeb(stream)

    result = TripComparison(trip_id=trip_id, n_frames=len(frames))
    unique, counts = np.unique([w.driver.state for w in stream], return_counts=True)
    result.driver_states = dict(zip(unique.tolist(), counts.tolist()))

    gt_hazard = np.isfinite(gt_ttc) & (gt_ttc < GT_HAZARD_TTC_S)
    gt_events = _events(gt_hazard)
    result.gt_events = len(gt_events)

    for start, end in gt_events:
        # Systems may legitimately react slightly before the hazard window
        # opens — that is the whole point of an early warning — so we look back
        # 2 s as well.
        look_back = max(0, start - 40)
        window = slice(look_back, end + 1)
        hazard_time = timestamps[start]

        for brake, covered_key, lead_list in (
            (guardian_brake, "guardian", result.guardian_lead_s),
            (fixed_brake, "fixed", result.fixed_lead_s),
        ):
            active = np.flatnonzero(brake[window] > 0.0)
            if active.size == 0:
                continue
            first = look_back + int(active[0])
            # Positive lead = intervened BEFORE the hazard window opened.
            lead_list.append(float(hazard_time - timestamps[first]))
            if covered_key == "guardian":
                result.guardian_covered += 1
            else:
                result.fixed_covered += 1

    # Interventions outside any ground-truth hazard window, classified by
    # whether the ground truth had ANY object on a collision course at the
    # time. "Braked early on a real car" and "braked at empty road" are
    # different engineering problems and must not share a counter.
    hazard_dilated = np.convolve(gt_hazard.astype(int), np.ones(81), mode="same") > 0
    for brake, early_attr, phantom_attr in (
        (guardian_brake, "guardian_early_events", "guardian_phantom_events"),
        (fixed_brake, "fixed_early_events", "fixed_phantom_events"),
    ):
        early = phantom = 0
        for start, end in _events(brake > 0.0):
            if hazard_dilated[start:end + 1].any():
                continue  # already counted as coverage
            if np.isfinite(gt_ttc[start:end + 1]).any():
                early += 1      # a real object, just further away
            else:
                phantom += 1    # nothing was there
        setattr(result, early_attr, early)
        setattr(result, phantom_attr, phantom)

    return result


def compare_all(
    trip_ids: Optional[Iterable[str]] = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    verbose: bool = True,
) -> list[TripComparison]:
    """Run the comparison across trips and print the summary table."""
    trip_ids = list(trip_ids) if trip_ids else list(PRACTICE_TRIP_IDS)
    results = [compare_trip(t, data_root) for t in trip_ids]

    if not verbose:
        return results

    print("=" * 94)
    print("GUARDIAN (driver-adaptive) vs FIXED-THRESHOLD AEB  (TTC < "
          f"{FIXED_AEB_TTC_S}s)")
    print("=" * 94)
    print(f"{'trip':13s}{'GT ev':>6s}{'covered G/F':>13s}"
          f"{'lead G':>9s}{'lead F':>9s}{'early G/F':>11s}{'phantom G/F':>13s}"
          f"   driver")
    print("-" * 94)

    for r in results:
        dominant = max(r.driver_states.items(), key=lambda kv: kv[1])[0] if r.driver_states else "-"
        print(f"{r.trip_id:13s}{r.gt_events:>6d}"
              f"{f'{r.guardian_covered}/{r.fixed_covered}':>13s}"
              f"{r.mean_guardian_lead:>8.2f}s{r.mean_fixed_lead:>8.2f}s"
              f"{f'{r.guardian_early_events}/{r.fixed_early_events}':>11s}"
              f"{f'{r.guardian_phantom_events}/{r.fixed_phantom_events}':>13s}"
              f"   {dominant}")

    total_gt = sum(r.gt_events for r in results)
    g_cov = sum(r.guardian_covered for r in results)
    f_cov = sum(r.fixed_covered for r in results)
    g_leads = [x for r in results for x in r.guardian_lead_s]
    f_leads = [x for r in results for x in r.fixed_lead_s]
    g_early = sum(r.guardian_early_events for r in results)
    g_phantom = sum(r.guardian_phantom_events for r in results)
    f_early = sum(r.fixed_early_events for r in results)
    f_phantom = sum(r.fixed_phantom_events for r in results)

    print("-" * 94)
    print(f"{'TOTAL':13s}{total_gt:>6d}{f'{g_cov}/{f_cov}':>13s}"
          f"{np.mean(g_leads) if g_leads else 0:>8.2f}s"
          f"{np.mean(f_leads) if f_leads else 0:>8.2f}s"
          f"{f'{g_early}/{f_early}':>11s}{f'{g_phantom}/{f_phantom}':>13s}")
    print("=" * 94)

    if total_gt:
        print(f"\nHazard coverage : Guardian {g_cov}/{total_gt} "
              f"({100 * g_cov / total_gt:.0f}%)  vs  fixed AEB {f_cov}/{total_gt} "
              f"({100 * f_cov / total_gt:.0f}%)")
    if g_leads and f_leads:
        gain = np.mean(g_leads) - np.mean(f_leads)
        print(f"Warning lead    : Guardian is {gain:+.2f}s earlier on average "
              f"(~{gain * 13.9:.0f} m at 50 km/h)")
    print(f"Early on a real object : Guardian {g_early}  vs  fixed AEB {f_early}   "
          f"(a real car, further than the {GT_HAZARD_TTC_S}s hazard window)")
    print(f"Phantom braking        : Guardian {g_phantom}  vs  fixed AEB {f_phantom}   "
          f"(nothing was there - this is the cost to fix)")
    return results


# ===========================================================================
# SECTION 4 — CLI
# ===========================================================================
if __name__ == "__main__":
    import argparse

    # The driver-facing explanations are Vietnamese by default, and a Windows
    # console defaults to cp1252, which cannot encode "ổ". Without this the
    # trace dies on the first tone mark instead of printing the explanation.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # already UTF-8, or not a real TTY
            pass

    parser = argparse.ArgumentParser(
        description="Run Guardian's decision chain on real trips."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--trips", nargs="*", default=None)
    parser.add_argument("--trace", metavar="TRIP",
                        help="Print a frame-by-frame audit trail for one trip.")
    parser.add_argument("--trace-frames", type=int, default=6,
                        help="How many intervention frames to show with --trace.")
    parser.add_argument("--lang", default="vi", choices=["vi", "en"],
                        help="Language for the driver-facing explanation.")
    args = parser.parse_args()

    if args.trace:
        stream = build_world_stream(args.trace, Path(args.data_root))
        commands = run_chain(stream)
        shown = 0
        print(f"Audit trail for {args.trace} (intervention frames only)\n")
        for world, command in zip(stream, commands):
            if not command.is_intervention:
                continue
            print(f"--- frame {world.frame_id}  t={world.timestamp:.2f}s ---")
            print(world.summary())
            print(command.audit_trail())
            # The audit trail is for an engineer; this is what the driver hears.
            explanation = explain(world, command, lang=args.lang)
            print(f"  SAYS: {explanation.headline}")
            for factor in explanation.factors:
                print(f"        - {factor.text}")
            print(f"  (explanation in {explanation.latency_ms:.2f} ms)")
            print()
            shown += 1
            if shown >= args.trace_frames:
                break
        if shown == 0:
            print("No interventions in this trip.")
    else:
        compare_all(args.trips, Path(args.data_root))
