"""
detect.py
=========

THE DROWSINESS CLAIM, MEASURED.

The repository claims a physiological rule engine detects driver impairment.
This module tests that claim against the labelled footage and reports the
numbers a safety reviewer would ask for, rather than the one that reads best:

    1. RECALL ON IMPAIRED FRAMES - of the frames where the driver really was
       drowsy, yawning or micro-sleeping, how many did we flag?
    2. FALSE-ALARM RATE ON CLEAR FRAMES - of the frames where the driver was
       fine, how many did we flag anyway? A system that cries wolf gets
       switched off, so this number matters as much as the first.
    3. TRANSITION LAG - when the driver's state changes, how long until the
       output follows? This is a real cost of the 2 s majority-vote smoothing,
       and it should be stated, not hidden.
    4. WARM-UP - PERCLOS is measured over a 10 s window, so the first 10 s of
       any trip is computed from a partial window. Scores are reported with
       and without it.

WHY `distracted` IS NOT COUNTED AS IMPAIRMENT
---------------------------------------------
Eyes-off-road and falling-asleep are different failures with different
responses. Pooling them would let a strong `distracted` score hide a weak
drowsiness score - which is precisely the claim under test. `IMPAIRED_STATES`
is drowsy / yawning / microsleep only.

WHAT THIS IS NOT
----------------
The labelled trips carry ONE driver each, and their labels are constant over
long stretches. Six drivers is six independent samples: enough to show the
thresholds fire on real physiology, nowhere near enough to claim a population
result. `guardian/opendata/` exists to test the same thresholds on strangers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from guardian.challenge2.classifier import (
    build_temporal_features,
    compute_metrics,
    load_trip_labels,
)
from guardian.challenge2.features import FEATURES_DIR, load_trip_features
from guardian.challenge2.labels import IMPAIRED_STATES
from guardian.challenge2.rules import (
    DEFAULT_DATA_ROOT,
    PRACTICE_TRIP_IDS,
    RuleEngine,
)

#: The capture rate of the labelled trips. Everything time-based derives from it.
DEFAULT_FPS = 20.0

#: PERCLOS is averaged over 10 s (201 frames at 20 FPS). Until that many frames
#: have been seen the window is partial, so early output is measured separately.
WARMUP_SECONDS = 10.0


def _is_impaired(states: Sequence[str]) -> np.ndarray:
    return np.array([s in IMPAIRED_STATES for s in states], dtype=bool)


def _segments(labels: Sequence[str]) -> list[tuple[str, int, int]]:
    """Run-length encode: [(state, start_index, length), ...]."""
    out: list[tuple[str, int, int]] = []
    index = 0
    for state, group in itertools.groupby(labels):
        length = len(list(group))
        out.append((str(state), index, length))
        index += length
    return out


@dataclass(frozen=True)
class Transition:
    """One ground-truth state change, and how long the output took to follow."""

    at_s: float
    from_state: str
    to_state: str
    #: None when the output never reached the new state before the trip ended.
    followed_after_s: Optional[float]

    def describe(self) -> str:
        if self.followed_after_s is None:
            return (f"{self.from_state} -> {self.to_state} at {self.at_s:.1f}s: "
                    f"never followed")
        return (f"{self.from_state} -> {self.to_state} at {self.at_s:.1f}s: "
                f"followed after {self.followed_after_s:.2f}s")


@dataclass(frozen=True)
class TripResult:
    """Everything measured for one trip."""

    trip_id: str
    fps: float
    truth: tuple[str, ...]
    predicted: tuple[str, ...]
    accuracy: float
    macro_f1: float
    composite: float
    impaired_frames: int
    impaired_caught: int
    clear_frames: int
    false_alarms: int
    false_alarms_after_warmup: int
    clear_frames_after_warmup: int
    transitions: tuple[Transition, ...] = field(default=())

    @property
    def n_frames(self) -> int:
        return len(self.truth)

    @property
    def recall(self) -> Optional[float]:
        """Share of truly impaired frames that were flagged. None if no impairment."""
        if self.impaired_frames == 0:
            return None
        return self.impaired_caught / self.impaired_frames

    @property
    def false_alarm_rate(self) -> Optional[float]:
        """Share of clear frames wrongly flagged as impaired. None if never clear."""
        if self.clear_frames == 0:
            return None
        return self.false_alarms / self.clear_frames

    @property
    def false_alarm_rate_after_warmup(self) -> Optional[float]:
        if self.clear_frames_after_warmup == 0:
            return None
        return self.false_alarms_after_warmup / self.clear_frames_after_warmup

    def to_dict(self) -> dict:
        return {
            "trip_id": self.trip_id,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "composite": self.composite,
            "impaired_frames": self.impaired_frames,
            "impaired_caught": self.impaired_caught,
            "recall": self.recall,
            "clear_frames": self.clear_frames,
            "false_alarms": self.false_alarms,
            "false_alarm_rate": self.false_alarm_rate,
            "false_alarm_rate_after_warmup": self.false_alarm_rate_after_warmup,
            "truth_segments": [
                {"state": s, "start_s": i / self.fps, "duration_s": n / self.fps}
                for s, i, n in _segments(self.truth)
            ],
            "predicted_segments": [
                {"state": s, "start_s": i / self.fps, "duration_s": n / self.fps}
                for s, i, n in _segments(self.predicted)
            ],
            "transitions": [
                {
                    "at_s": t.at_s,
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "followed_after_s": t.followed_after_s,
                }
                for t in self.transitions
            ],
        }


@dataclass(frozen=True)
class Summary:
    """Pooled result across trips - frame-weighted, not an average of averages."""

    trips: tuple[TripResult, ...]
    impaired_frames: int
    impaired_caught: int
    clear_frames: int
    false_alarms: int
    clear_frames_after_warmup: int
    false_alarms_after_warmup: int
    mean_composite: float

    @property
    def recall(self) -> Optional[float]:
        if self.impaired_frames == 0:
            return None
        return self.impaired_caught / self.impaired_frames

    @property
    def false_alarm_rate(self) -> Optional[float]:
        if self.clear_frames == 0:
            return None
        return self.false_alarms / self.clear_frames

    @property
    def false_alarm_rate_after_warmup(self) -> Optional[float]:
        if self.clear_frames_after_warmup == 0:
            return None
        return self.false_alarms_after_warmup / self.clear_frames_after_warmup

    def lags(self) -> list[float]:
        return [
            t.followed_after_s
            for trip in self.trips
            for t in trip.transitions
            if t.followed_after_s is not None
        ]

    def to_dict(self) -> dict:
        lags = self.lags()
        return {
            "trips": [t.to_dict() for t in self.trips],
            "pooled": {
                "impaired_frames": self.impaired_frames,
                "impaired_caught": self.impaired_caught,
                "recall": self.recall,
                "clear_frames": self.clear_frames,
                "false_alarms": self.false_alarms,
                "false_alarm_rate": self.false_alarm_rate,
                "false_alarm_rate_after_warmup": self.false_alarm_rate_after_warmup,
                "mean_composite": self.mean_composite,
                "transition_lag_s": {
                    "n": len(lags),
                    "median": float(np.median(lags)) if lags else None,
                    "max": float(np.max(lags)) if lags else None,
                },
            },
        }


def evaluate_trip(
    trip_id: str,
    engine: Optional[RuleEngine] = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    features_dir: Path = FEATURES_DIR,
    fps: float = DEFAULT_FPS,
) -> TripResult:
    """Run the shipped engine over one labelled trip and measure it."""
    engine = engine or RuleEngine()

    temporal = build_temporal_features(load_trip_features(trip_id, features_dir))
    predicted = np.asarray(engine.predict(temporal), dtype=object)
    truth = load_trip_labels(trip_id, data_root).to_numpy()

    # Feature extraction can end a frame or two short of the manifest; compare
    # only frames that exist on both sides rather than padding either.
    n = min(len(truth), len(predicted))
    truth, predicted = truth[:n], predicted[:n]

    metrics = compute_metrics(truth, predicted)
    truly = _is_impaired(truth)
    flagged = _is_impaired(predicted)

    warmup = int(round(WARMUP_SECONDS * fps))
    late = np.zeros(n, dtype=bool)
    late[min(warmup, n):] = True
    clear_late = (~truly) & late

    return TripResult(
        trip_id=trip_id,
        fps=fps,
        truth=tuple(str(s) for s in truth),
        predicted=tuple(str(s) for s in predicted),
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        composite=metrics["composite"],
        impaired_frames=int(truly.sum()),
        impaired_caught=int((truly & flagged).sum()),
        clear_frames=int((~truly).sum()),
        false_alarms=int((~truly & flagged).sum()),
        clear_frames_after_warmup=int(clear_late.sum()),
        false_alarms_after_warmup=int((clear_late & flagged).sum()),
        transitions=tuple(_transitions(truth, predicted, fps)),
    )


def _transitions(
    truth: Sequence[str], predicted: Sequence[str], fps: float
) -> list[Transition]:
    """
    For every ground-truth state change, how long until the output agreed?

    Measured to the first frame carrying the new state, not to the start of a
    sustained run: the question is when the information became available.
    """
    out: list[Transition] = []
    segments = _segments(truth)
    for (prev_state, _, _), (state, start, _) in zip(segments, segments[1:]):
        after = [i for i in range(start, len(predicted)) if predicted[i] == state]
        followed = (after[0] - start) / fps if after else None
        out.append(
            Transition(
                at_s=start / fps,
                from_state=prev_state,
                to_state=state,
                followed_after_s=followed,
            )
        )
    return out


def evaluate_trips(
    trip_ids: Optional[Iterable[str]] = None,
    engine: Optional[RuleEngine] = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    features_dir: Path = FEATURES_DIR,
    fps: float = DEFAULT_FPS,
) -> list[TripResult]:
    engine = engine or RuleEngine()
    ids = list(trip_ids) if trip_ids else list(PRACTICE_TRIP_IDS)
    return [
        evaluate_trip(t, engine, data_root=data_root, features_dir=features_dir, fps=fps)
        for t in ids
    ]


def summarise(results: Sequence[TripResult]) -> Summary:
    """
    Pool per-frame counts across trips.

    Pooling counts rather than averaging per-trip rates on purpose: a trip with
    600 impaired frames should not carry the same weight as one with 20.
    """
    if not results:
        raise ValueError("No trips to summarise.")
    return Summary(
        trips=tuple(results),
        impaired_frames=sum(r.impaired_frames for r in results),
        impaired_caught=sum(r.impaired_caught for r in results),
        clear_frames=sum(r.clear_frames for r in results),
        false_alarms=sum(r.false_alarms for r in results),
        clear_frames_after_warmup=sum(r.clear_frames_after_warmup for r in results),
        false_alarms_after_warmup=sum(r.false_alarms_after_warmup for r in results),
        mean_composite=float(np.mean([r.composite for r in results])),
    )


def format_report(summary: Summary) -> str:
    """The console report. Recall and false alarms side by side, always."""
    lines: list[str] = []
    add = lines.append
    add("=" * 78)
    add("CAN GUARDIAN SEE A SLEEPY DRIVER?")
    add("=" * 78)
    add(f"{'trip':14s}{'impaired':>10s}{'caught':>9s}{'recall':>9s}"
        f"{'clear':>8s}{'false':>8s}{'rate':>8s}{'composite':>11s}")
    add("-" * 78)
    for r in summary.trips:
        recall = "  -  " if r.recall is None else f"{r.recall:.3f}"
        far = "  -  " if r.false_alarm_rate is None else f"{r.false_alarm_rate:.3f}"
        add(f"{r.trip_id:14s}{r.impaired_frames:>10d}{r.impaired_caught:>9d}"
            f"{recall:>9s}{r.clear_frames:>8d}{r.false_alarms:>8d}{far:>8s}"
            f"{r.composite:>11.1f}")
    add("-" * 78)
    recall = "n/a" if summary.recall is None else f"{summary.recall:.1%}"
    far = "n/a" if summary.false_alarm_rate is None else f"{summary.false_alarm_rate:.1%}"
    add(f"Impaired frames caught : {summary.impaired_caught}/{summary.impaired_frames}"
        f"  ({recall})")
    add(f"False alarms when clear: {summary.false_alarms}/{summary.clear_frames}"
        f"  ({far})")
    if summary.false_alarm_rate_after_warmup is not None:
        add(f"  ... excluding the first {WARMUP_SECONDS:.0f}s, while the PERCLOS "
            f"window fills: {summary.false_alarms_after_warmup}/"
            f"{summary.clear_frames_after_warmup}"
            f"  ({summary.false_alarm_rate_after_warmup:.1%})")
    lags = summary.lags()
    if lags:
        add(f"Lag following a real state change: median {np.median(lags):.2f}s, "
            f"worst {np.max(lags):.2f}s  (2.05s majority vote is the floor)")
    add(f"Mean composite (multi-class): {summary.mean_composite:.1f}/100")
    add("=" * 78)
    add("Six drivers is six independent samples. These numbers show the")
    add("thresholds fire on real physiology; they are not a population claim.")
    return "\n".join(lines)
