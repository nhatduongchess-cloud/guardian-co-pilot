"""
builder.py
==========

GUARDIAN CO-PILOT — SUBMISSION LAYER
Builds and validates the output CSV files.

WHY THIS MODULE EXISTS
----------------------
Every model in this repo ultimately produces ONE artefact: a CSV per trip at
`predictions/<team>/<trip_id>.csv`. If that file is malformed, the score is
lost no matter how good the model is. This module is the last line of defence.

THREE OUTPUT FORMAT RULES ENCODED HERE
------------------------------------
1. "Never write ground truth into the submitted CSV."
   -> We reject any column whose name looks like ground truth. This matters
      because the reference `baseline_ttc_predictor.py` writes a
      `ground_truth_ttc` column for local reference; submitting that file
      unchanged would violate the rule.

2. "If you skip a challenge, DROP its column entirely -- do not fill junk."
   -> A prediction column only appears if the caller explicitly sets it.

3. "Row count must equal the trip's frame count" (1800 for T0Xd, 600 for
   the -Sample practice trips), with contiguous frame_ids.

SUBMISSION FORMAT
-----------------
    frame_id,timestamp,predicted_ttc,predicted_driver_state,predicted_risk_score
    0,0.000,inf,alert,5
    ...

  - predicted_ttc            : seconds, `inf` when no obstacle is detected
  - predicted_driver_state   : alert|drowsy|yawning|distracted|microsleep
  - predicted_risk_score     : float 0-100
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# ===========================================================================
# SECTION 1 — CONSTANTS (the contract, in one place)
# ===========================================================================

# The only driver-state labels the scorer accepts.
VALID_DRIVER_STATES: frozenset[str] = frozenset(
    {"alert", "drowsy", "yawning", "distracted", "microsleep"}
)

# Columns that must always be present.
BASE_COLUMNS: tuple[str, ...] = ("frame_id", "timestamp")

# Optional prediction columns, one per challenge.
TTC_COLUMN = "predicted_ttc"
DRIVER_STATE_COLUMN = "predicted_driver_state"
RISK_SCORE_COLUMN = "predicted_risk_score"

PREDICTION_COLUMNS: tuple[str, ...] = (
    TTC_COLUMN,
    DRIVER_STATE_COLUMN,
    RISK_SCORE_COLUMN,
)

# Any column matching one of these patterns is treated as leaked ground truth
# and refused. Deliberately broad -- a false alarm costs us a rename, a miss
# costs us a rule violation.
_FORBIDDEN_COLUMN_PATTERN = re.compile(
    r"(ground[_\s]?truth|^gt[_\-]|_gt$|^true[_\-]|_true$|actual|label)",
    re.IGNORECASE,
)


class SubmissionError(ValueError):
    """Raised when a submission would be invalid. Always fail loudly here."""


# ===========================================================================
# SECTION 2 — THE BUILDER
# ===========================================================================


class SubmissionBuilder:
    """
    Accumulates per-frame predictions for ONE trip, then writes a valid CSV.

    Typical use:
        builder = SubmissionBuilder.from_trip(trip)     # reads ids/timestamps
        builder.set_ttc(ttc_values)                     # Challenge 1
        builder.set_driver_state(state_labels)          # Challenge 2
        builder.enable_risk_score(placeholder=0.0)      # Challenge 3
        path = builder.write("predictions", team="GuardianCoPilot")

    A challenge you never call simply has no column -- exactly what the rules
    require for challenges you did not attempt.
    """

    def __init__(
        self,
        trip_id: str,
        frame_ids: Sequence[int],
        timestamps: Sequence[float],
    ) -> None:
        """
        Args:
            trip_id:    e.g. "T01d" -- becomes the output filename.
            frame_ids:  every frame id in the trip, in order.
            timestamps: matching timestamps in seconds.
        """
        if len(frame_ids) != len(timestamps):
            raise SubmissionError(
                f"frame_ids ({len(frame_ids)}) and timestamps "
                f"({len(timestamps)}) must be the same length."
            )
        if len(frame_ids) == 0:
            raise SubmissionError("A submission needs at least one frame.")

        self.trip_id = trip_id
        self._frame_ids = np.asarray(frame_ids, dtype=int)
        self._timestamps = np.asarray(timestamps, dtype=float)

        # Prediction columns start as None = "challenge not attempted".
        self._ttc: Optional[np.ndarray] = None
        self._driver_state: Optional[list[str]] = None
        self._risk_score: Optional[np.ndarray] = None

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_trip(cls, trip) -> "SubmissionBuilder":
        """
        Build directly from a team_kit `TripDataset`.

        Duck-typed on purpose: we only need `trip_id` and the frames' ids and
        timestamps, so this module never has to import the benchmark's kit.
        """
        frame_ids: list[int] = []
        timestamps: list[float] = []
        for frame in trip.iter_frames():
            frame_ids.append(int(frame.frame_id))
            timestamps.append(float(frame.timestamp))
        return cls(trip.trip_id, frame_ids, timestamps)

    @property
    def n_frames(self) -> int:
        return len(self._frame_ids)

    # -- Challenge 1 ---------------------------------------------------------

    def set_ttc(self, values: Iterable[float]) -> "SubmissionBuilder":
        """
        Attach Time-To-Collision predictions (seconds).

        Use `float("inf")` for "no obstacle ahead". NaN is rejected outright:
        a NaN in the submission is silently scored as a miss, so we force the
        caller to decide between a number and `inf`.
        """
        arr = np.asarray(list(values), dtype=float)
        self._check_length(arr, TTC_COLUMN)

        if np.isnan(arr).any():
            bad = int(np.flatnonzero(np.isnan(arr))[0])
            raise SubmissionError(
                f"{TTC_COLUMN} contains NaN (first at index {bad}). "
                f"Use float('inf') when no obstacle is detected."
            )
        if (arr < 0).any():
            bad = int(np.flatnonzero(arr < 0)[0])
            raise SubmissionError(
                f"{TTC_COLUMN} contains a negative value at index {bad}: {arr[bad]}"
            )

        self._ttc = arr
        return self  # allow chaining

    # -- Challenge 2 ---------------------------------------------------------

    def set_driver_state(self, labels: Iterable[str]) -> "SubmissionBuilder":
        """Attach driver-state labels; every value must be one of the 5 classes."""
        values = [str(v) for v in labels]
        self._check_length(values, DRIVER_STATE_COLUMN)

        invalid = sorted(set(values) - VALID_DRIVER_STATES)
        if invalid:
            raise SubmissionError(
                f"{DRIVER_STATE_COLUMN} has invalid labels {invalid}. "
                f"Allowed: {sorted(VALID_DRIVER_STATES)}"
            )

        self._driver_state = values
        return self

    # -- Challenge 3 ---------------------------------------------------------

    def set_risk_score(self, values: Iterable[float]) -> "SubmissionBuilder":
        """
        Attach trip risk scores (0-100).

        NOTE ON SCORING: the benchmark's evaluator does NOT read these numbers.
        The column's presence is what registers us for Challenge 3; the actual
        score is recomputed from trip kinematics plus `near_miss`, which is
        derived from our own `predicted_ttc`. We still validate the range so
        the file stays well-formed and honest.
        """
        arr = np.asarray(list(values), dtype=float)
        self._check_length(arr, RISK_SCORE_COLUMN)

        if np.isnan(arr).any():
            raise SubmissionError(f"{RISK_SCORE_COLUMN} contains NaN.")
        if ((arr < 0) | (arr > 100)).any():
            bad = int(np.flatnonzero((arr < 0) | (arr > 100))[0])
            raise SubmissionError(
                f"{RISK_SCORE_COLUMN} out of range [0, 100] at index {bad}: {arr[bad]}"
            )

        self._risk_score = arr
        return self

    def enable_risk_score(self, placeholder: float = 0.0) -> "SubmissionBuilder":
        """
        Register for Challenge 3 with a constant column.

        Convenience for the common case: since the evaluator ignores the
        values, a constant is enough to opt in. Prefer `set_risk_score()` when
        you genuinely model per-frame risk (better story for a reviewer).
        """
        return self.set_risk_score(np.full(self.n_frames, float(placeholder)))

    # -- output --------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Assemble the submission as a DataFrame (columns in canonical order)."""
        if not any(
            x is not None for x in (self._ttc, self._driver_state, self._risk_score)
        ):
            raise SubmissionError(
                f"Trip {self.trip_id}: no predictions set. Call set_ttc(), "
                f"set_driver_state(), and/or set_risk_score() first."
            )

        data: dict[str, object] = {
            "frame_id": self._frame_ids,
            "timestamp": self._timestamps,
        }
        # Only attempted challenges contribute a column.
        if self._ttc is not None:
            data[TTC_COLUMN] = self._ttc
        if self._driver_state is not None:
            data[DRIVER_STATE_COLUMN] = self._driver_state
        if self._risk_score is not None:
            data[RISK_SCORE_COLUMN] = self._risk_score

        return pd.DataFrame(data)

    def write(
        self,
        root: str | Path = "predictions",
        team: str = "GuardianCoPilot",
        float_format: str = "%.3f",
    ) -> Path:
        """
        Write `<root>/<team>/<trip_id>.csv` and validate it before returning.

        Returns:
            Path to the written file.
        """
        df = self.to_dataframe()

        # Validate BEFORE writing so we never leave a bad file on disk.
        validate_submission(df, expected_frames=self.n_frames, trip_id=self.trip_id)

        out_dir = Path(root) / team
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.trip_id}.csv"

        # `inf` survives to_csv as the literal text "inf", which is what the
        # format specifies. float_format keeps timestamps tidy (0.050 etc).
        df.to_csv(out_path, index=False, float_format=float_format)
        return out_path

    # -- internal ------------------------------------------------------------

    def _check_length(self, values: Sequence, column: str) -> None:
        """Every prediction column must cover exactly one value per frame."""
        if len(values) != self.n_frames:
            raise SubmissionError(
                f"{column}: got {len(values)} values but trip {self.trip_id} "
                f"has {self.n_frames} frames."
            )


# ===========================================================================
# SECTION 3 — VALIDATION (usable on any CSV, including ones we did not build)
# ===========================================================================


def validate_submission(
    df: pd.DataFrame,
    expected_frames: Optional[int] = None,
    trip_id: str = "<unknown>",
) -> dict:
    """
    Check a submission DataFrame against every benchmark rule.

    Raises:
        SubmissionError: on the first violation found.

    Returns:
        A small report dict describing which challenges the file covers.
    """
    # -- forbidden (ground-truth-looking) columns ---------------------------
    leaked = [c for c in df.columns if _FORBIDDEN_COLUMN_PATTERN.search(str(c))]
    if leaked:
        raise SubmissionError(
            f"{trip_id}: submission contains ground-truth-like column(s) {leaked}. "
            f"The rules forbid writing ground truth into the submitted CSV "
            f"(the benchmark's baseline adds 'ground_truth_ttc' -- strip it)."
        )

    # -- required base columns ----------------------------------------------
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise SubmissionError(f"{trip_id}: missing required column(s) {missing}.")

    # -- at least one challenge ---------------------------------------------
    present = [c for c in PREDICTION_COLUMNS if c in df.columns]
    if not present:
        raise SubmissionError(
            f"{trip_id}: no prediction columns. Expected at least one of "
            f"{list(PREDICTION_COLUMNS)}."
        )

    # -- unexpected extra columns -------------------------------------------
    allowed = set(BASE_COLUMNS) | set(PREDICTION_COLUMNS)
    extra = [c for c in df.columns if c not in allowed]
    if extra:
        raise SubmissionError(
            f"{trip_id}: unexpected column(s) {extra}. Only "
            f"{sorted(allowed)} are allowed."
        )

    # -- row count -----------------------------------------------------------
    if expected_frames is not None and len(df) != expected_frames:
        raise SubmissionError(
            f"{trip_id}: has {len(df)} rows but the trip has "
            f"{expected_frames} frames."
        )

    # -- frame ids contiguous and ordered ------------------------------------
    frame_ids = df["frame_id"].to_numpy()
    expected_ids = np.arange(len(df))
    if not np.array_equal(frame_ids, expected_ids):
        raise SubmissionError(
            f"{trip_id}: frame_id must run 0..{len(df) - 1} in order "
            f"(got first={frame_ids[0]}, last={frame_ids[-1]})."
        )

    # -- timestamps ----------------------------------------------------------
    ts = df["timestamp"].to_numpy(dtype=float)
    if np.isnan(ts).any():
        raise SubmissionError(f"{trip_id}: timestamp contains NaN.")
    if (np.diff(ts) <= 0).any():
        raise SubmissionError(f"{trip_id}: timestamp must strictly increase.")

    # -- per-challenge value checks ------------------------------------------
    if TTC_COLUMN in df.columns:
        ttc = pd.to_numeric(df[TTC_COLUMN], errors="coerce").to_numpy(dtype=float)
        if np.isnan(ttc).any():
            bad = int(np.flatnonzero(np.isnan(ttc))[0])
            raise SubmissionError(
                f"{trip_id}: {TTC_COLUMN} has NaN/unparsable value at row {bad}. "
                f"Use 'inf' for no obstacle."
            )
        if (ttc < 0).any():
            raise SubmissionError(f"{trip_id}: {TTC_COLUMN} has negative values.")

    if DRIVER_STATE_COLUMN in df.columns:
        states = df[DRIVER_STATE_COLUMN].astype(str)
        invalid = sorted(set(states) - VALID_DRIVER_STATES)
        if invalid:
            raise SubmissionError(
                f"{trip_id}: {DRIVER_STATE_COLUMN} has invalid label(s) {invalid}."
            )

    if RISK_SCORE_COLUMN in df.columns:
        risk = pd.to_numeric(df[RISK_SCORE_COLUMN], errors="coerce").to_numpy(float)
        if np.isnan(risk).any():
            raise SubmissionError(f"{trip_id}: {RISK_SCORE_COLUMN} has NaN.")
        if ((risk < 0) | (risk > 100)).any():
            raise SubmissionError(
                f"{trip_id}: {RISK_SCORE_COLUMN} outside [0, 100]."
            )

    # -- report --------------------------------------------------------------
    n_inf = (
        int(np.isinf(pd.to_numeric(df[TTC_COLUMN], errors="coerce")).sum())
        if TTC_COLUMN in df.columns
        else 0
    )
    return {
        "trip_id": trip_id,
        "rows": len(df),
        "challenges": {
            "1_ttc": TTC_COLUMN in df.columns,
            "2_driver_state": DRIVER_STATE_COLUMN in df.columns,
            "3_risk_score": RISK_SCORE_COLUMN in df.columns,
        },
        "ttc_inf_frames": n_inf,
    }


def validate_submission_file(
    path: str | Path, expected_frames: Optional[int] = None
) -> dict:
    """Load a CSV from disk and validate it. Convenience wrapper."""
    path = Path(path)
    if not path.exists():
        raise SubmissionError(f"Submission file not found: {path}")
    df = pd.read_csv(path)
    return validate_submission(df, expected_frames=expected_frames, trip_id=path.stem)


def strip_ground_truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with any ground-truth-looking column removed.

    Use this to clean the benchmark's baseline output, which ships a
    `ground_truth_ttc` column for local inspection only.
    """
    drop = [c for c in df.columns if _FORBIDDEN_COLUMN_PATTERN.search(str(c))]
    return df.drop(columns=drop)


# ===========================================================================
# SECTION 4 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    n = 600
    builder = SubmissionBuilder(
        trip_id="T01-Sample",
        frame_ids=list(range(n)),
        timestamps=[i * 0.05 for i in range(n)],
    )

    # Challenge 1: mostly clear road, a hazard near the end.
    ttc = [float("inf")] * n
    for i in range(560, 575):
        ttc[i] = 2.5 - (i - 560) * 0.1
    builder.set_ttc(ttc)

    # Challenge 2 + 3.
    builder.set_driver_state(["alert"] * 400 + ["drowsy"] * 200)
    builder.enable_risk_score(0.0)

    out = builder.write(root="predictions", team="GuardianCoPilot")
    print(f"Wrote: {out}")

    report = validate_submission_file(out, expected_frames=n)
    print(f"Valid. Report: {report}")

    print("\nFirst rows:")
    print(pd.read_csv(out).head(3).to_string(index=False))

    # Show the guard rails firing.
    print("\nGuard rails:")
    for label, action in [
        ("NaN in ttc", lambda: builder.set_ttc([float("nan")] * n)),
        ("bad driver label", lambda: builder.set_driver_state(["sleepy"] * n)),
        ("wrong length", lambda: builder.set_ttc([1.0] * 10)),
        (
            "leaked ground truth",
            lambda: validate_submission(
                pd.DataFrame(
                    {
                        "frame_id": [0],
                        "timestamp": [0.0],
                        "predicted_ttc": [1.0],
                        "ground_truth_ttc": [1.0],
                    }
                )
            ),
        ),
    ]:
        try:
            action()
            print(f"  {label:22s} -> NOT CAUGHT (bug!)")
        except SubmissionError as exc:
            print(f"  {label:22s} -> caught: {str(exc)[:70]}")
