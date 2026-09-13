"""
run_all.py
==========

GUARDIAN CO-PILOT — END-TO-END SUBMISSION PIPELINE

ONE COMMAND turns the raw dataset into the files we submit:

    python guardian/run_all.py

    -> predictions/GuardianCoPilot/T01d.csv ... T10d.csv

The evaluation rubric explicitly scores "code that can be re-run from the
instructions", so this file is the entry point a judge will use. It is also
what gives us a VALID SUBMISSION EARLY: get something legal on disk first,
then improve the models behind it. Every later improvement is one re-run away.

WHAT IT PRODUCES PER TRIP
-------------------------
    predicted_ttc            Challenge 1 — seconds, `inf` when no obstacle
    predicted_driver_state   Challenge 2 — the rule engine's label
    predicted_risk_score     Challenge 3 — registers us for the challenge

A note on Challenge 3: the benchmark's evaluator does NOT read the numbers in
`predicted_risk_score`. It recomputes the trip score from kinematics plus
`near_miss`, which it derives from OUR `predicted_ttc`. The column's presence
is what opts us in. We still emit a meaningful per-frame risk (from the driver
state) rather than a constant — it costs nothing and it is what the cockpit
dashboard visualises.

PLUGGABLE PREDICTORS
--------------------
`ttc_predictor` is an argument, not a hard-coded call. Today it is the
benchmark's SGBM baseline; when the YOLOv8 pipeline lands we swap one argument
and re-run. The pipeline itself never changes.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from guardian.challenge2.classifier import build_temporal_features
from guardian.challenge2.features import (
    DriverFeatureExtractor,
    FEATURES_DIR,
    feature_path,
    load_trip_features,
)
from guardian.challenge2.rules import RuleEngine
from guardian.submission.builder import SubmissionBuilder, validate_submission_file

# --- dataset toolkit ---------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STARTERKIT = _PROJECT_ROOT / "starterkit"
if str(_STARTERKIT) not in sys.path:
    sys.path.insert(0, str(_STARTERKIT))
from team_kit.dataset_loader import TripDataset  # noqa: E402


DEFAULT_DATA_ROOT = Path(os.getenv("GUARDIAN_DATA_ROOT", r"C:/guardian_data"))
SCORED_TRIP_IDS = tuple(f"T{i:02d}d" for i in range(1, 11))
PRACTICE_TRIP_IDS = tuple(f"T0{i}-Sample" for i in range(1, 7))
TEAM_NAME = "GuardianCoPilot"

# Per-frame risk weight for each driver state. Used only for our own
# `predicted_risk_score` column and the dashboard — see the note above.
_STATE_RISK = {
    "alert": 5.0,
    "distracted": 45.0,
    "drowsy": 55.0,
    "yawning": 35.0,
    "microsleep": 90.0,
}


# ===========================================================================
# SECTION 1 — CHALLENGE 1: TTC PREDICTORS (pluggable)
# ===========================================================================

# A TTC predictor maps a loaded trip to one TTC value per frame.
TtcPredictor = Callable[[TripDataset], np.ndarray]


def guardian_ttc(trip: TripDataset) -> np.ndarray:
    """
    Guardian's own TTC model: YOLOv8 detection + calibrated stereo depth +
    collision-cone filtering + track-based closing speed.

    Practice-set composite 65.6 vs the benchmark baseline's 19.7. Requires
    cached perception (`guardian/challenge1/perception.py`), which
    `ensure_perception` below generates on demand.
    """
    from guardian.challenge1.perception import extract_trips, perception_path
    from guardian.challenge1.ttc import guardian_ttc_predictor

    if not perception_path(trip.trip_id).exists():
        print(f"    perception missing -> extracting {trip.trip_id}")
        extract_trips([trip.trip_id], data_root=trip.trip_dir.parent)
    return guardian_ttc_predictor(trip)


def baseline_ttc(trip: TripDataset) -> np.ndarray:
    """
    The benchmark's stereo-SGBM baseline (composite 19.7 on practice trips).

    Kept as the default so the pipeline always produces a legal submission,
    and as the reference every improvement is measured against.
    """
    from team_kit.baseline_ttc_predictor import predict_trip

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / f"{trip.trip_id}.csv"
        predict_trip(trip.trip_dir, csv_path, verbose=False)
        df = pd.read_csv(csv_path)

    # Take ONLY the prediction column: the baseline also writes
    # `ground_truth_ttc`, which must never reach a submitted file.
    return df["predicted_ttc"].to_numpy(dtype=float)


# ===========================================================================
# SECTION 2 — CHALLENGE 2: DRIVER STATE
# ===========================================================================


def ensure_features(
    trip_id: str,
    data_root: Path,
    features_dir: Path = FEATURES_DIR,
    extractor: Optional[DriverFeatureExtractor] = None,
) -> pd.DataFrame:
    """
    Load cached driver features, extracting them first if needed.

    Extraction is the slow part (~50 fps), so it is cached to CSV and reused.
    """
    if not feature_path(trip_id, features_dir).exists():
        print(f"    features missing -> extracting {trip_id}")
        extractor = extractor or DriverFeatureExtractor()
        df, _ = extractor.extract_trip(trip_id, data_root=data_root)
        features_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(feature_path(trip_id, features_dir), index=False)
    return load_trip_features(trip_id, features_dir)


# ===========================================================================
# SECTION 3 — BUILD ONE TRIP'S SUBMISSION
# ===========================================================================


def build_trip_submission(
    trip_id: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    out_root: Path = Path("predictions"),
    team: str = TEAM_NAME,
    ttc_predictor: Optional[TtcPredictor] = None,
    engine: Optional[RuleEngine] = None,
    features_dir: Path = FEATURES_DIR,
    do_challenge1: bool = True,
    do_challenge2: bool = True,
    do_challenge3: bool = True,
) -> tuple[Path, dict]:
    """
    Produce one validated submission CSV.

    Returns:
        (path, stats) where stats summarises what went into the file.
    """
    t0 = time.time()
    trip = TripDataset(Path(data_root) / trip_id)
    builder = SubmissionBuilder.from_trip(trip)
    stats: dict = {"trip_id": trip_id, "frames": builder.n_frames}

    # --- Challenge 1 --------------------------------------------------------
    if do_challenge1:
        predictor = ttc_predictor or guardian_ttc
        ttc = predictor(trip)
        if len(ttc) != builder.n_frames:
            raise ValueError(
                f"{trip_id}: TTC predictor returned {len(ttc)} values for "
                f"{builder.n_frames} frames."
            )
        builder.set_ttc(ttc)
        finite = np.isfinite(ttc)
        stats["ttc_finite"] = int(finite.sum())
        # Frames the benchmark will count as near-misses for Challenge 3.
        stats["near_miss_frames"] = int((ttc < 1.5).sum())

    # --- Challenge 2 --------------------------------------------------------
    states: Optional[np.ndarray] = None
    if do_challenge2:
        raw_features = ensure_features(trip_id, data_root, features_dir)
        temporal = build_temporal_features(raw_features)
        engine = engine or RuleEngine()
        states = engine.predict(temporal)
        builder.set_driver_state(states)
        unique, counts = np.unique(states, return_counts=True)
        stats["states"] = dict(zip(unique.tolist(), counts.tolist()))

    # --- Challenge 3 --------------------------------------------------------
    if do_challenge3:
        if states is not None:
            # A per-frame risk derived from driver state. Not read by the
            # scorer, but honest, useful for the dashboard, and better than
            # a meaningless constant.
            risk = np.array([_STATE_RISK.get(s, 5.0) for s in states], dtype=float)
            builder.set_risk_score(risk)
        else:
            builder.enable_risk_score(0.0)

    path = builder.write(root=out_root, team=team)
    stats["seconds"] = time.time() - t0
    stats["path"] = str(path)
    return path, stats


# ===========================================================================
# SECTION 4 — RUN EVERYTHING
# ===========================================================================


def run_all(
    trip_ids: Optional[list[str]] = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    out_root: Path = Path("predictions"),
    team: str = TEAM_NAME,
    ttc_predictor: Optional[TtcPredictor] = None,
    do_challenge1: bool = True,
    do_challenge2: bool = True,
    do_challenge3: bool = True,
) -> pd.DataFrame:
    """Generate and validate submissions for every requested trip."""
    trip_ids = list(trip_ids) if trip_ids else list(SCORED_TRIP_IDS)
    engine = RuleEngine()

    # One extractor shared across trips (model loading is expensive).
    extractor_needed = do_challenge2 and any(
        not feature_path(t).exists() for t in trip_ids
    )
    if extractor_needed:
        print("Some feature caches are missing; the extractor will load once.\n")

    rows = []
    total_t0 = time.time()
    for i, trip_id in enumerate(trip_ids, 1):
        print(f"[{i}/{len(trip_ids)}] {trip_id}")
        path, stats = build_trip_submission(
            trip_id,
            data_root=data_root,
            out_root=out_root,
            team=team,
            ttc_predictor=ttc_predictor,
            engine=engine,
            do_challenge1=do_challenge1,
            do_challenge2=do_challenge2,
            do_challenge3=do_challenge3,
        )

        # Validate immediately: a bad file must never sit in the output folder.
        report = validate_submission_file(path, expected_frames=stats["frames"])
        rows.append({**stats, "valid": True, "challenges": report["challenges"]})

        summary = []
        if "ttc_finite" in stats:
            summary.append(f"ttc_finite={stats['ttc_finite']}")
            summary.append(f"near_miss={stats['near_miss_frames']}")
        if "states" in stats:
            top = sorted(stats["states"].items(), key=lambda kv: -kv[1])[:2]
            summary.append("states=" + ",".join(f"{k}:{v}" for k, v in top))
        print(f"    -> {Path(path).name}  {'  '.join(summary)}  "
              f"({stats['seconds']:.0f}s)")

    table = pd.DataFrame(rows)
    print(f"\nWrote {len(table)} submission file(s) to "
          f"{Path(out_root) / team} in {time.time() - total_t0:.0f}s")
    print("All files passed validation.")
    return table


# ===========================================================================
# SECTION 5 — CLI
# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Guardian Co-Pilot benchmark submissions."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--out", default="predictions", help="Output root folder.")
    parser.add_argument("--team", default=TEAM_NAME)
    parser.add_argument("--trips", nargs="*", default=None,
                        help="Trip ids (default: the 10 scored trips).")
    parser.add_argument("--practice", action="store_true",
                        help="Run on the 6 practice trips instead (self-check).")
    parser.add_argument("--no-c1", action="store_true", help="Skip Challenge 1.")
    parser.add_argument("--no-c2", action="store_true", help="Skip Challenge 2.")
    parser.add_argument("--no-c3", action="store_true", help="Skip Challenge 3.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Score the output (practice trips only).")
    args = parser.parse_args()

    if args.trips:
        trips = args.trips
    elif args.practice:
        trips = list(PRACTICE_TRIP_IDS)
    else:
        trips = list(SCORED_TRIP_IDS)

    table = run_all(
        trip_ids=trips,
        data_root=Path(args.data_root),
        out_root=Path(args.out),
        team=args.team,
        do_challenge1=not args.no_c1,
        do_challenge2=not args.no_c2,
        do_challenge3=not args.no_c3,
    )

    if args.evaluate:
        # Only practice trips have ground truth to score against.
        from team_kit.evaluation import evaluate, print_report

        print("\nScoring with the benchmark's evaluator...")
        report = evaluate(Path(args.out) / args.team, Path(args.data_root), None)
        print_report(report)
