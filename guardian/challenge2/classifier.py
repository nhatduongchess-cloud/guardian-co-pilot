"""
classifier.py
=============

GUARDIAN CO-PILOT — CHALLENGE 2 (Driver Intelligence)
Temporal feature engineering + driver-state classifier.

THE CORE IDEA: drowsiness is a PROCESS, not a picture
-----------------------------------------------------
A single frame cannot tell a normal blink from a microsleep — both show closed
eyes. What separates them is TIME: how long the eyes stay shut, how often they
close, how variable the pattern is. Measured on T02-Sample (all `drowsy`):

    eye_blink   mean 0.178   median 0.077   max 0.826   std 0.218

The median is low but the spikes are huge: the driver's eyes open and close
repeatedly. A per-frame threshold would be pure noise. So we build windowed
features over several time scales:

    PERCLOS      fraction of a window with eyes closed  (the classic
                 drowsiness measure used in the literature)
    closed_run   longest consecutive closure -> microsleep vs blink
    rolling std  how erratic the pattern is
    windows      1s (21 frames), 3s (61), 10s (201) at 20 FPS

On top of these we train a gradient-boosted tree. Trees suit this problem:
few features, non-linear thresholds ("PERCLOS above X AND run longer than Y"),
and they stay interpretable — which matters because Guardian must EXPLAIN its
decisions, not just make them.

VALIDATION: Leave-One-Trip-Out, always
--------------------------------------
Same protocol as the CNN attempt so the two are directly comparable. The CNN
scored 18.5/100 because it memorised faces; these features are identity-
invariant by construction, so they should transfer to unseen drivers.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from guardian.challenge2.dataset import CLASS_NAMES
from guardian.challenge2.features import FEATURES_DIR, load_trip_features

# --- dataset toolkit (for ground-truth labels of the practice trips) ---------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STARTERKIT = _PROJECT_ROOT / "starterkit"
if str(_STARTERKIT) not in sys.path:
    sys.path.insert(0, str(_STARTERKIT))
from team_kit.dataset_loader import TripDataset  # noqa: E402


DEFAULT_DATA_ROOT = Path(os.getenv("GUARDIAN_DATA_ROOT", r"C:/guardian_data"))
PRACTICE_TRIP_IDS = tuple(f"T0{i}-Sample" for i in range(1, 7))
MODEL_PATH = _PROJECT_ROOT / "artifacts" / "challenge2_classifier.joblib"

# Eye-closure threshold: above this the eye counts as "closed" for PERCLOS.
# 0.4 sits comfortably between alert (~0.07) and microsleep (~0.66).
EYE_CLOSED_THRESHOLD = 0.4

# Window sizes in frames at 20 FPS: 1s, 3s, 10s.
WINDOWS = (21, 61, 201)


# ===========================================================================
# SECTION 1 — TEMPORAL FEATURE ENGINEERING
# ===========================================================================


def _longest_run_ending_here(mask: np.ndarray) -> np.ndarray:
    """
    For each position, the length of the run of True values ending there.

    Used for "how long have the eyes been closed right now" — the signal that
    separates a 0.2s blink from a 2s microsleep.
    """
    out = np.zeros(len(mask), dtype=float)
    run = 0.0
    for i, flag in enumerate(mask):
        run = run + 1 if flag else 0.0
        out[i] = run
    return out


def build_temporal_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Turn per-frame features into windowed, time-aware features.

    Args:
        raw: one trip's cached feature CSV (see features.py).

    Returns:
        DataFrame with one row per frame and only numeric model inputs.
    """
    df = raw.copy()
    out = pd.DataFrame(index=df.index)

    # --- instantaneous signals (still useful on their own) -----------------
    for col in (
        "eye_blink", "jaw_open", "phone_conf",
        "eye_squint_left", "eye_squint_right",
        "brow_down_left", "brow_down_right",
        "head_pitch", "head_yaw", "head_roll",
    ):
        out[col] = df[col].astype(float)
    out["face_found"] = df["face_found"].astype(float)

    eye_closed = (df["eye_blink"] >= EYE_CLOSED_THRESHOLD).to_numpy()
    out["eye_closed"] = eye_closed.astype(float)

    # --- how long the eyes have been shut ----------------------------------
    # Frames -> seconds (20 FPS) so the number is physically meaningful.
    out["closed_run_sec"] = _longest_run_ending_here(eye_closed) / 20.0

    # --- windowed statistics over several time scales ----------------------
    for window in WINDOWS:
        tag = f"w{window}"
        # `center=True` looks both backwards and forwards: at inference we have
        # the whole trip, so using future frames is legitimate and much more
        # accurate than a causal window. (A real vehicle would use a causal
        # window; we note this in the write-up.)
        roll = df["eye_blink"].rolling(window, center=True, min_periods=1)
        out[f"eb_mean_{tag}"] = roll.mean()
        out[f"eb_std_{tag}"] = roll.std().fillna(0.0)
        out[f"eb_max_{tag}"] = roll.max()

        # PERCLOS: the fraction of the window with eyes closed.
        out[f"perclos_{tag}"] = (
            pd.Series(eye_closed.astype(float))
            .rolling(window, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

        jaw = df["jaw_open"].rolling(window, center=True, min_periods=1)
        out[f"jaw_mean_{tag}"] = jaw.mean()
        out[f"jaw_max_{tag}"] = jaw.max()

        out[f"phone_rate_{tag}"] = (
            df["phone_detected"].astype(float)
            .rolling(window, center=True, min_periods=1).mean()
        )

        # Head movement: a distracted driver's head wanders more.
        out[f"yaw_std_{tag}"] = (
            df["head_yaw"].rolling(window, center=True, min_periods=1)
            .std().fillna(0.0)
        )
        out[f"pitch_mean_{tag}"] = (
            df["head_pitch"].rolling(window, center=True, min_periods=1).mean()
        )

    return out.astype(float).fillna(0.0)


FEATURE_NAMES: Optional[list[str]] = None  # filled on first build


# ===========================================================================
# SECTION 2 — LOADING LABELLED DATA
# ===========================================================================


def load_trip_labels(trip_id: str, data_root: Path = DEFAULT_DATA_ROOT) -> pd.Series:
    """Ground-truth driver_state per frame (practice trips only)."""
    trip = TripDataset(Path(data_root) / trip_id)
    return pd.Series(
        [f.driver_state for f in trip.iter_frames()],
        index=[f.frame_id for f in trip.iter_frames()],
        name="driver_state",
    )


def load_dataset(
    trip_ids: Optional[list[str]] = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    features_dir: Path = FEATURES_DIR,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build (X, y, groups) across trips.

    `groups` holds the trip id per row — that is what makes Leave-One-Trip-Out
    possible.
    """
    global FEATURE_NAMES
    trip_ids = list(trip_ids) if trip_ids else list(PRACTICE_TRIP_IDS)

    frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    groups: list[np.ndarray] = []

    for trip_id in trip_ids:
        raw = load_trip_features(trip_id, features_dir)
        features = build_temporal_features(raw)
        y = load_trip_labels(trip_id, data_root).to_numpy()

        if len(y) != len(features):
            raise ValueError(
                f"{trip_id}: {len(features)} feature rows vs {len(y)} labels."
            )

        frames.append(features)
        labels.append(y)
        groups.append(np.full(len(features), trip_id))

    X = pd.concat(frames, ignore_index=True)
    FEATURE_NAMES = list(X.columns)
    return X, np.concatenate(labels), np.concatenate(groups)


# ===========================================================================
# SECTION 3 — MODEL
# ===========================================================================


def build_classifier(random_state: int = 42):
    """
    A histogram gradient-boosted tree.

    Chosen because: it handles a few dozen numeric features well, learns the
    threshold-style rules this problem needs ("PERCLOS > x AND run > y"),
    trains in seconds, and stays inspectable for the explainability story.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=6,
        # Small leaves would let the model carve out single trips; keeping a
        # floor here is a guard against re-learning identity through the back
        # door.
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=random_state,
    )


def smooth_predictions(labels: np.ndarray, window: int = 41) -> np.ndarray:
    """
    Majority-vote smoothing over a centred window.

    Driver states persist for seconds, but per-frame predictions flicker.
    Voting over ~2s removes isolated wrong frames without erasing real
    transitions.
    """
    if window <= 1:
        return labels
    half = window // 2
    smoothed = labels.copy()
    for i in range(len(labels)):
        lo, hi = max(0, i - half), min(len(labels), i + half + 1)
        values, counts = np.unique(labels[lo:hi], return_counts=True)
        smoothed[i] = values[counts.argmax()]
    return smoothed


# ===========================================================================
# SECTION 4 — METRICS (mirror the benchmark's formula)
# ===========================================================================


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy, macro-F1 over classes PRESENT in y_true, and the composite."""
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = float(accuracy_score(y_true, y_pred))
    present = sorted(set(y_true.tolist()))
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    )
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "composite": 50.0 * accuracy + 50.0 * macro_f1,
    }


# ===========================================================================
# SECTION 5 — LEAVE-ONE-TRIP-OUT CROSS-VALIDATION
# ===========================================================================


@dataclass
class FoldResult:
    val_trip_id: str
    n_train: int
    n_val: int
    accuracy: float
    macro_f1: float
    composite: float
    accuracy_smoothed: float
    composite_smoothed: float
    classes_in_val: list[str]
    classes_missing_from_train: list[str]


@dataclass
class CvReport:
    folds: list[FoldResult] = field(default_factory=list)
    mean_composite: float = 0.0
    mean_composite_smoothed: float = 0.0


def run_loto_cv(
    trip_ids: Optional[list[str]] = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    smooth_window: int = 41,
    verbose: bool = True,
) -> CvReport:
    """Train one classifier per held-out trip and report honest averages."""
    X, y, groups = load_dataset(trip_ids, data_root)
    if verbose:
        print(f"Dataset: {X.shape[0]} frames x {X.shape[1]} temporal features")
        print(f"Classes: {dict(zip(*np.unique(y, return_counts=True)))}\n")

    report = CvReport()
    for held_out in sorted(set(groups)):
        train_mask = groups != held_out
        val_mask = ~train_mask

        model = build_classifier()
        model.fit(X[train_mask], y[train_mask])

        y_true = y[val_mask]
        y_pred = model.predict(X[val_mask])
        y_pred_smooth = smooth_predictions(y_pred, smooth_window)

        raw_metrics = compute_metrics(y_true, y_pred)
        smooth_metrics = compute_metrics(y_true, y_pred_smooth)

        val_classes = sorted(set(y_true.tolist()))
        missing = sorted(set(val_classes) - set(y[train_mask].tolist()))

        report.folds.append(FoldResult(
            val_trip_id=held_out,
            n_train=int(train_mask.sum()),
            n_val=int(val_mask.sum()),
            accuracy=raw_metrics["accuracy"],
            macro_f1=raw_metrics["macro_f1"],
            composite=raw_metrics["composite"],
            accuracy_smoothed=smooth_metrics["accuracy"],
            composite_smoothed=smooth_metrics["composite"],
            classes_in_val=val_classes,
            classes_missing_from_train=missing,
        ))

    if report.folds:
        report.mean_composite = float(np.mean([f.composite for f in report.folds]))
        report.mean_composite_smoothed = float(
            np.mean([f.composite_smoothed for f in report.folds])
        )

    if verbose:
        _print_report(report)
    return report


def _print_report(report: CvReport) -> None:
    print("=" * 88)
    print("CHALLENGE 2 - LOTO cross-validation (identity-invariant features)")
    print("=" * 88)
    print(f"{'held-out trip':14s}{'acc':>7s}{'macroF1':>9s}{'composite':>11s}"
          f"{'+smoothed':>11s}  note")
    print("-" * 88)
    for f in report.folds:
        note = (f"missing in train: {','.join(f.classes_missing_from_train)}"
                if f.classes_missing_from_train else "")
        print(f"{f.val_trip_id:14s}{f.accuracy:>7.3f}{f.macro_f1:>9.3f}"
              f"{f.composite:>11.1f}{f.composite_smoothed:>11.1f}  {note}")
    print("-" * 88)
    print(f"{'MEAN':14s}{'':>7s}{'':>9s}{report.mean_composite:>11.1f}"
          f"{report.mean_composite_smoothed:>11.1f}")

    clean = [f for f in report.folds if not f.classes_missing_from_train]
    if clean and len(clean) != len(report.folds):
        mean_clean = float(np.mean([f.composite_smoothed for f in clean]))
        print(f"{'MEAN (folds with full class coverage, smoothed)':60s}"
              f"{mean_clean:>10.1f}   [{len(clean)}/{len(report.folds)}]")
    print("=" * 88)


# ===========================================================================
# SECTION 6 — FINAL MODEL + INFERENCE
# ===========================================================================


def train_final_model(
    trip_ids: Optional[list[str]] = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    out_path: Path = MODEL_PATH,
) -> Path:
    """Train on every labelled trip and save the model we ship."""
    import joblib

    X, y, _ = load_dataset(trip_ids, data_root)
    model = build_classifier()
    model.fit(X, y)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": list(X.columns),
            "class_names": list(CLASS_NAMES),
            "eye_closed_threshold": EYE_CLOSED_THRESHOLD,
            "windows": list(WINDOWS),
        },
        out_path,
    )
    print(f"Saved classifier: {out_path}  (trained on {len(X)} frames)")
    return Path(out_path)


def predict_trip(
    trip_id: str,
    model_path: Path = MODEL_PATH,
    features_dir: Path = FEATURES_DIR,
    smooth_window: int = 41,
) -> np.ndarray:
    """
    Predict a driver-state label for every frame of a trip.

    Returns:
        Array of label strings, one per frame, ready for the SubmissionBuilder.
    """
    import joblib

    blob = joblib.load(model_path)
    model = blob["model"]

    raw = load_trip_features(trip_id, features_dir)
    X = build_temporal_features(raw)

    # Column order must match training exactly.
    X = X[blob["feature_names"]]

    predictions = model.predict(X)
    return smooth_predictions(predictions, smooth_window)


# ===========================================================================
# SECTION 7 — CLI
# ===========================================================================
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Driver-state classifier.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--smooth-window", type=int, default=41)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--report", default="artifacts/challenge2_classifier_report.json")
    args = parser.parse_args()

    do_cv = args.cv or not args.final
    do_final = args.final or not args.cv

    if do_cv:
        report = run_loto_cv(
            data_root=Path(args.data_root), smooth_window=args.smooth_window
        )
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        print(f"\nReport written to {args.report}")

    if do_final:
        train_final_model(data_root=Path(args.data_root))
