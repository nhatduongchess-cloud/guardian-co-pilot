"""
rldd.py
=======

SIXTY DRIVERS INSTEAD OF SIX.

Everything Guardian knows about drowsiness was learned from six people. That is
the single largest threat to every number in this repository, and the one the
frame-level open-data check in `evaluate.py` could not remove: loose stills have
no clock, so they can confirm that eyelids separate the classes but they cannot
score a rule that is defined over a *window* of time.

This module fixes the missing clock. It reads the blink-feature release that
accompanies

    R. Ghoddoosian, M. Galib, V. Athitsos, "A Realistic Dataset and Baseline
    Temporal Model for Early Drowsiness Detection", CVPR Workshops 2019.
    https://github.com/rezaghoddoosian/Early-Drowsiness-Detection  (MIT)

which is the published companion to UTA-RLDD: 60 participants, each filmed in
three states - alert, low vigilant, drowsy - and reduced to per-blink features.
The release ships the features, not the faces, which is why this is the one
public source Guardian can use end to end without touching anyone's video.

WHAT ONE ROW IS
---------------
A window of 30 consecutive blinks by one person, each blink described by four
numbers in this order:

    0  frequency   blinks per unit time around this blink
    1  amplitude   how far the eyelid travelled
    2  duration    how long the eye stayed shut
    3  velocity    how fast the lid moved

and one label for the window: 0 alert, 5 low vigilant, 10 drowsy.

WHY THIS DATASET AND NOT ANOTHER
--------------------------------
Because of how the authors normalised it, and it is worth being explicit since
it is the same idea Guardian's per-driver learning loop is built on. Every
feature is standardised against *that person's own alert session*: the last
third of their alert video supplies the mean and standard deviation. A duration
of +3 does not mean "a long blink", it means "three standard deviations longer
than this person blinks when they are fresh".

That is the whole argument the project has been making since the CNN failed.
A model handed raw pixels from six drivers learns the six faces. A statistic
referred to a driver's own baseline describes a *state*, and states are the
thing that transfers. Here that claim can finally be tested against strangers.

THE FIVE FOLDS
--------------
The authors ship the data pre-split into five subject-disjoint folds. Their
paper says the split is by participant; the arrays themselves carry no
participant id, so that claim cannot be verified from the files alone. What
*can* be verified is the consequence - no window may appear on both sides of a
split - and `check_integrity()` does exactly that rather than taking it on
trust.

NOTHING IS VENDORED
-------------------
`ensure_release()` clones the authors' MIT-licensed repository into
`artifacts/`, which is git-ignored. This repository ships code. The data stays
the authors'.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Where the authors' release is cached. Git-ignored; see `.gitignore`.
DEFAULT_CACHE_DIR = _PROJECT_ROOT / "artifacts" / "opendata" / "uta_rldd_blinks"

RELEASE_URL = "https://github.com/rezaghoddoosian/Early-Drowsiness-Detection.git"

#: Column order inside a blink row. Fixed by the authors' `Preprocessing.py`.
BLINK_FEATURES: tuple[str, ...] = ("frequency", "amplitude", "duration", "velocity")

#: The authors' scores, in Guardian's vocabulary. `low_vigilant` has no
#: counterpart in the shipped rule engine - Guardian speaks alert / drowsy -
#: and that mismatch is reported rather than papered over.
LABEL_NAMES: dict[int, str] = {0: "alert", 5: "low_vigilant", 10: "drowsy"}

N_FOLDS = 5
WINDOW_BLINKS = 30


@dataclass(frozen=True)
class Provenance:
    """Where the numbers came from and what may be done with them."""

    name: str = "UTA-RLDD blink features (Ghoddoosian et al., CVPRW 2019)"
    url: str = RELEASE_URL
    licence: str = (
        "MIT (the authors' repository). The underlying UTA-RLDD video is "
        "released separately under its own agreement and is NOT used here - "
        "only the published per-blink features are read."
    )
    citation: str = (
        "R. Ghoddoosian, M. Galib, V. Athitsos, 'A Realistic Dataset and "
        "Baseline Temporal Model for Early Drowsiness Detection', "
        "IEEE/CVF CVPR Workshops, 2019."
    )
    subjects: int = 60
    normalisation: str = (
        "Per-subject: each feature is standardised against the last third of "
        "that participant's own alert session."
    )
    caveats: tuple[str, ...] = (
        "Labels are session-level. A window taken from a drowsy session is "
        "labelled drowsy even if the participant happened to be alert in that "
        "minute, so per-window scores are a floor, not a ceiling.",
        "The front-end is dlib eye-aspect-ratio blink detection, not the "
        "MediaPipe blendshapes Guardian runs in the cabin. Conclusions here "
        "are about the METHOD - per-driver baselining plus threshold rules - "
        "and not about the specific numeric cut points shipped in "
        "challenge2/rules.py, which are defined on a different signal.",
        "Windows overlap: consecutive windows share blinks, so the effective "
        "number of independent observations is smaller than the row count.",
    )


PROVENANCE = Provenance()


@dataclass(frozen=True)
class Fold:
    """One subject-disjoint split, exactly as the authors shipped it."""

    index: int
    train_windows: np.ndarray  # (n_train, 30, 4)
    train_labels: np.ndarray  # (n_train,) in {0, 5, 10}
    test_windows: np.ndarray  # (n_test, 30, 4)
    test_labels: np.ndarray  # (n_test,)

    @property
    def n_train(self) -> int:
        return len(self.train_windows)

    @property
    def n_test(self) -> int:
        return len(self.test_windows)


def ensure_release(cache_dir: Path | None = None, *, quiet: bool = False) -> Path:
    """Clone the authors' release once into the git-ignored cache; return its path.

    A shallow clone: we want one commit's worth of `.npy`, not the history of
    someone else's research code.
    """
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    if (cache_dir / "Blinks_30_Fold1.npy").exists():
        return cache_dir
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"[rldd] cloning {RELEASE_URL} -> {cache_dir}")
    subprocess.run(
        ["git", "clone", "--depth", "1", RELEASE_URL, str(cache_dir)],
        check=True,
        capture_output=quiet,
    )
    return cache_dir


def load_folds(cache_dir: Path | None = None) -> list[Fold]:
    """Read all five folds, failing loudly rather than half-loading."""
    root = Path(cache_dir or DEFAULT_CACHE_DIR)
    folds: list[Fold] = []
    for k in range(1, N_FOLDS + 1):
        try:
            train_x = np.load(root / f"Blinks_30_Fold{k}.npy")
            train_y = np.load(root / f"Labels_30_Fold{k}.npy").ravel()
            test_x = np.load(root / f"BlinksTest_30_Fold{k}.npy")
            test_y = np.load(root / f"LabelsTest_30_Fold{k}.npy").ravel()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"fold {k} is missing from {root}. Run ensure_release() first."
            ) from exc
        folds.append(
            Fold(
                index=k,
                train_windows=validate_windows(train_x, f"Blinks_30_Fold{k}"),
                train_labels=validate_labels(train_y, f"Labels_30_Fold{k}"),
                test_windows=validate_windows(test_x, f"BlinksTest_30_Fold{k}"),
                test_labels=validate_labels(test_y, f"LabelsTest_30_Fold{k}"),
            )
        )
        if folds[-1].n_train != len(train_y) or folds[-1].n_test != len(test_y):
            raise ValueError(f"fold {k}: windows and labels disagree in length")
    return folds


def validate_windows(array: np.ndarray, name: str) -> np.ndarray:
    """Shape and finiteness, checked once so nothing downstream has to guess."""
    if array.ndim != 3 or array.shape[1:] != (WINDOW_BLINKS, len(BLINK_FEATURES)):
        raise ValueError(
            f"{name}: expected (n, {WINDOW_BLINKS}, {len(BLINK_FEATURES)}), "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contains non-finite values")
    return array


def validate_labels(array: np.ndarray, name: str) -> np.ndarray:
    """Labels must be the authors' three scores and nothing else."""
    unknown = set(np.unique(array).tolist()) - set(LABEL_NAMES)
    if unknown:
        raise ValueError(f"{name}: unexpected label values {sorted(unknown)}")
    return array.astype(int)


def padding_mask(windows: np.ndarray) -> np.ndarray:
    """True where a blink slot holds a real blink.

    The authors pad sequences shorter than 30 blinks with leading rows of
    zeros. A row of zeros is *not* "an average blink": zero is the per-subject
    mean after standardisation, so folding padding into a window mean quietly
    drags every statistic toward that person's alert baseline. Roughly one slot
    in 1,200 is padding - small, and still worth excluding by name rather than
    by luck.
    """
    return ~(windows == 0).all(axis=2)


def check_integrity(folds: Sequence[Fold]) -> dict:
    """Verify the property that subject-disjoint folds must have.

    The files carry no participant id, so "these folds are split by subject"
    cannot be confirmed directly. Its necessary consequence can: no window may
    appear on both sides of a split, and no window may appear in two different
    folds' test sets. If either fails, the split leaks and every held-out
    number built on it is inflated.
    """
    overlaps = {}
    seen_test: dict[bytes, int] = {}
    cross_fold = 0
    for fold in folds:
        train_sigs = {row.tobytes() for row in fold.train_windows.reshape(fold.n_train, -1)}
        shared = 0
        for row in fold.test_windows.reshape(fold.n_test, -1):
            sig = row.tobytes()
            if sig in train_sigs:
                shared += 1
            if sig in seen_test and seen_test[sig] != fold.index:
                cross_fold += 1
            seen_test.setdefault(sig, fold.index)
        overlaps[f"fold{fold.index}"] = shared
    return {
        "train_test_overlap_per_fold": overlaps,
        "windows_shared_between_test_folds": cross_fold,
        "leak_free": all(v == 0 for v in overlaps.values()) and cross_fold == 0,
    }


def iter_held_out(folds: Sequence[Fold]) -> Iterator[Fold]:
    """The only honest way to read this data: one fold out, the rest in."""
    yield from folds


def summarise_folds(folds: Sequence[Fold]) -> dict:
    """Row counts and class balance, so a reader can see what was scored."""
    per_fold = []
    for fold in folds:
        per_fold.append(
            {
                "fold": fold.index,
                "train_windows": fold.n_train,
                "test_windows": fold.n_test,
                "test_class_counts": {
                    LABEL_NAMES[v]: int((fold.test_labels == v).sum())
                    for v in sorted(LABEL_NAMES)
                },
            }
        )
    total_test = int(sum(f.n_test for f in folds))
    all_labels = np.concatenate([f.test_labels for f in folds])
    return {
        "n_folds": len(folds),
        "window_blinks": WINDOW_BLINKS,
        "features": list(BLINK_FEATURES),
        "total_held_out_windows": total_test,
        "held_out_class_counts": {
            LABEL_NAMES[v]: int((all_labels == v).sum()) for v in sorted(LABEL_NAMES)
        },
        "per_fold": per_fold,
    }


# ---------------------------------------------------------------------------
# Video boundaries: from a window verdict to a session verdict.
# ---------------------------------------------------------------------------
#
# A 30-blink window is a few minutes of one person. The thing a co-pilot
# actually has to get right is the *session*: is this driver, on this drive,
# fit to continue? Guardian answers that by smoothing window verdicts into a
# trip state (`challenge2.rules.majority_smooth`), and the authors of this
# release do the same thing under the name "voting accuracy per video".
#
# Their `Training.py` hard-codes, per fold, the row index at which each test
# video begins. Those indices are read out of their file rather than copied
# into ours, so the numbers stay theirs - and they are then CHECKED, not
# trusted: every window inside one video must carry the same label, because a
# video is one person in one state. If a boundary list did not partition a
# fold that way it would be the wrong list, and `load_video_boundaries` would
# rather raise than silently score against a scrambled grouping.

_START_INDICES_RE = re.compile(
    r"^(?![ \t]*#)[ \t]*start_indices\s*=\s*\[([0-9,\s]*)\]", re.M
)


def _parse_start_indices(training_py: Path) -> list[list[int]]:
    text = training_py.read_text(encoding="utf-8", errors="replace")
    out = []
    for body in _START_INDICES_RE.findall(text):
        values = [int(v) for v in body.replace("\n", "").split(",") if v.strip()]
        if values:
            out.append(values)
    return out


def _partitions_cleanly(starts: Sequence[int], labels: np.ndarray) -> bool:
    """True when every segment these boundaries cut is a single-label run."""
    if not starts or starts[0] != 0 or max(starts) >= len(labels):
        return False
    bounds = list(starts) + [len(labels)]
    for a, b in zip(bounds, bounds[1:]):
        if b <= a or len(set(labels[a:b].tolist())) != 1:
            return False
    return True


def load_video_boundaries(
    cache_dir: Path | None, folds: Sequence[Fold]
) -> dict[int, list[int]]:
    """Match each fold to the authors' boundary list, by checking not by hoping."""
    root = Path(cache_dir or DEFAULT_CACHE_DIR)
    candidates = _parse_start_indices(root / "Training.py")
    if not candidates:
        raise ValueError(f"no start_indices found in {root / 'Training.py'}")

    boundaries: dict[int, list[int]] = {}
    for fold in folds:
        matches = [c for c in candidates if _partitions_cleanly(c, fold.test_labels)]
        if len(matches) != 1:
            raise ValueError(
                f"fold {fold.index}: expected exactly one boundary list to "
                f"partition it into single-label videos, found {len(matches)}"
            )
        boundaries[fold.index] = matches[0]
    return boundaries


def video_segments(starts: Sequence[int], n_rows: int) -> list[tuple[int, int]]:
    """Boundary starts -> [start, stop) pairs covering every row exactly once."""
    bounds = list(starts) + [n_rows]
    return [(a, b) for a, b in zip(bounds, bounds[1:])]
