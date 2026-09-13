"""
harness.py
==========

GUARDIAN CO-PILOT — EVALUATION LAYER
The measuring stick for the whole project.

WHY THIS MODULE EXISTS
----------------------
"Improving the model" is meaningless without a number. This harness runs ANY
predictor across the practice trips (the 6 that ship with ground truth),
scores it with the reference `team_kit/evaluation.py`, and prints one
comparable benchmark table. Every model we build later plugs in here to answer
a single question: "is this better than the baseline?"

DESIGN — the predictor is a plug
--------------------------------
The harness does not know or care what a predictor is. It only needs a
callable:

    predict(trip: TripDataset) -> SubmissionBuilder

Today we plug in the benchmark's SGBM baseline; next week a YOLOv8 pipeline.
The harness code never changes. (Same Dependency-Injection idea as the rest of
Guardian, applied to ML.)

IMPORTANT — scoring is always done by the BENCHMARK's evaluator
---------------------------------------------------------------
We call `team_kit.evaluation.evaluate(...)`, which loads ground truth from the
trusted trip directory (never from our CSV). So our benchmark number is the
same one the benchmark would compute locally.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

import pandas as pd

from guardian.submission.builder import SubmissionBuilder


# ===========================================================================
# SECTION 1 — LOCATE THE TEAM KIT AND THE DATA
# ===========================================================================
# The dataset toolkit (team_kit/) is the benchmark's package; the dataset lives
# outside the repo (it must NOT be committed). We resolve both here so the
# rest of the module can just `import team_kit...`.
# ---------------------------------------------------------------------------

# Project root = two levels up from this file (guardian/evaluation/harness.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make `team_kit` importable even when running from the repo root, by adding
# the sibling `starterkit/` folder to sys.path if needed.
_STARTERKIT = _PROJECT_ROOT / "starterkit"
if str(_STARTERKIT) not in sys.path:
    sys.path.insert(0, str(_STARTERKIT))

try:
    from team_kit.dataset_loader import TripDataset  # noqa: E402
    from team_kit.evaluation import evaluate  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment problem, be loud
    raise ImportError(
        f"Could not import the dataset toolkit from {_STARTERKIT}. "
        f"Make sure starterkit/ exists next to guardian/. ({exc})"
    ) from exc


# Where the extracted trips live. Overridable via env var so no path is
# hard-coded into logic. Default matches where we extracted the dataset.
DATA_ROOT = Path(os.getenv("GUARDIAN_DATA_ROOT", r"C:/guardian_data"))

# The 6 practice trips have full ground truth -> the only trips we can score
# locally. Scored trips (T0Xd) are graded by the benchmark.
PRACTICE_TRIP_IDS: tuple[str, ...] = (
    "T01-Sample",
    "T02-Sample",
    "T03-Sample",
    "T04-Sample",
    "T05-Sample",
    "T06-Sample",
)


# A predictor is any callable turning a loaded trip into a filled builder.
class Predictor(Protocol):
    def __call__(self, trip: TripDataset) -> SubmissionBuilder: ...


# ===========================================================================
# SECTION 2 — BENCHMARK RESULT
# ===========================================================================


@dataclass
class BenchmarkResult:
    """Structured result of one benchmark run, for printing or comparison."""

    data_root: Path
    trip_ids: list[str]
    # overall composites (None = that challenge was not attempted)
    c1_composite: Optional[float] = None
    c2_composite: Optional[float] = None
    c3_composite: Optional[float] = None
    # per-trip rows for a table: list of dicts
    per_trip: list[dict] = field(default_factory=list)


# ===========================================================================
# SECTION 3 — THE HARNESS
# ===========================================================================


def run_benchmark(
    predictor: Predictor,
    trip_ids: Optional[list[str]] = None,
    data_root: Optional[Path] = None,
    team: str = "benchmark",
    work_dir: Optional[Path] = None,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Run `predictor` over the given trips and score with the benchmark's evaluator.

    Args:
        predictor: callable(trip) -> SubmissionBuilder (already filled).
        trip_ids:  which trips to score (default: the 6 practice trips).
        data_root: folder containing the trip directories (default: DATA_ROOT).
        team:      subfolder name for the temporary prediction CSVs.
        work_dir:  where to write CSVs (default: a temp dir, auto-cleaned).
        verbose:   print the table when done.

    Returns:
        BenchmarkResult with overall + per-trip numbers.
    """
    data_root = Path(data_root) if data_root else DATA_ROOT
    trip_ids = list(trip_ids) if trip_ids else list(PRACTICE_TRIP_IDS)

    if not data_root.is_dir():
        raise FileNotFoundError(
            f"Data root not found: {data_root}. Set GUARDIAN_DATA_ROOT or pass "
            f"data_root=..."
        )

    # Use a temp dir for the CSVs unless the caller wants to keep them.
    tmp_ctx = (
        tempfile.TemporaryDirectory() if work_dir is None else None
    )
    pred_root = Path(work_dir) if work_dir else Path(tmp_ctx.name)
    pred_dir = pred_root / team
    pred_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1) Run the predictor on each trip and write a CSV.
        for trip_id in trip_ids:
            trip_path = data_root / trip_id
            trip = TripDataset(trip_path)
            builder = predictor(trip)
            builder.write(root=pred_root, team=team)

        # 2) Score everything with the benchmark's evaluator (GT from data_root).
        report = evaluate(pred_dir, data_root, None)

        # 3) Reshape into our BenchmarkResult.
        result = _report_to_result(report, data_root, trip_ids)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    if verbose:
        print_benchmark(result)
    return result


def _report_to_result(report, data_root: Path, trip_ids: list[str]) -> BenchmarkResult:
    """Convert the benchmark's EvaluationReport into a flat BenchmarkResult."""
    # Index per-trip metrics by trip id for easy joining.
    c1_by_trip = {t.trip_id: t for t in report.per_trip}
    c2_by_trip = {t.trip_id: t for t in report.per_trip_challenge2}
    c3_by_trip = {t.trip_id: t for t in report.per_trip_challenge3}

    rows: list[dict] = []
    for tid in trip_ids:
        row: dict = {"trip_id": tid}
        if tid in c1_by_trip:
            m = c1_by_trip[tid]
            row["c1_mae_crit"] = m.mae_critical
            row["c1_f1"] = m.f1
            row["c1_composite"] = m.composite_score
        if tid in c2_by_trip:
            row["c2_acc"] = c2_by_trip[tid].accuracy
            row["c2_macro_f1"] = c2_by_trip[tid].macro_f1
            row["c2_composite"] = c2_by_trip[tid].composite_score
        if tid in c3_by_trip:
            row["c3_composite"] = c3_by_trip[tid].composite_score
        rows.append(row)

    return BenchmarkResult(
        data_root=data_root,
        trip_ids=trip_ids,
        c1_composite=report.overall_composite_score if report.per_trip else None,
        c2_composite=report.overall_challenge2_composite,
        c3_composite=report.overall_challenge3_composite,
        per_trip=rows,
    )


def print_benchmark(result: BenchmarkResult) -> None:
    """Pretty-print a benchmark result as a table + overall summary."""
    df = pd.DataFrame(result.per_trip).set_index("trip_id")
    # Round for readability without losing the story.
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print("\n" + "=" * 78)
        print("BENCHMARK - per trip")
        print("=" * 78)
        print(df.to_string())
        print("-" * 78)
        parts = []
        if result.c1_composite is not None:
            parts.append(f"C1 (TTC)          = {result.c1_composite:6.2f} / 100")
        if result.c2_composite is not None:
            parts.append(f"C2 (driver state) = {result.c2_composite:6.2f} / 100")
        if result.c3_composite is not None:
            parts.append(f"C3 (fleet score)  = {result.c3_composite:6.2f} / 100")
        print("OVERALL:  " + "   ".join(parts))
        print("=" * 78)


# ===========================================================================
# SECTION 4 — A REFERENCE PREDICTOR: the benchmark's SGBM baseline
# ===========================================================================
# Wrapping the baseline as a Predictor gives us the reference number every
# improvement is measured against. It also exercises the full path:
# baseline CSV (with its ground_truth_ttc column) -> strip -> SubmissionBuilder.
# ---------------------------------------------------------------------------


def baseline_predictor(trip: TripDataset) -> SubmissionBuilder:
    """Run the benchmark's SGBM baseline and return it as a clean builder."""
    from team_kit.baseline_ttc_predictor import predict_trip

    with tempfile.TemporaryDirectory() as tmp:
        raw_csv = Path(tmp) / f"{trip.trip_id}.csv"
        predict_trip(trip.trip_dir, raw_csv, verbose=False)
        df = pd.read_csv(raw_csv)

    # The baseline writes 'ground_truth_ttc' for local reference; we take only
    # the prediction column. 'inf' comes back as the float inf via read_csv.
    builder = SubmissionBuilder.from_trip(trip)
    builder.set_ttc(df["predicted_ttc"].to_numpy(dtype=float))
    return builder


# ===========================================================================
# SECTION 5 — CLI / SELF-TEST: benchmark the baseline on all practice trips
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark a predictor.")
    parser.add_argument(
        "--data-root", default=str(DATA_ROOT), help="Folder containing trips."
    )
    parser.add_argument(
        "--trips", nargs="*", default=None,
        help="Trip ids to score (default: the 6 practice trips).",
    )
    args = parser.parse_args()

    print(f"Benchmarking the benchmark's SGBM baseline on practice trips")
    print(f"Data root: {args.data_root}")

    run_benchmark(
        predictor=baseline_predictor,
        trip_ids=args.trips,
        data_root=Path(args.data_root),
    )
