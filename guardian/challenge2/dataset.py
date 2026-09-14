"""
dataset.py
==========

GUARDIAN CO-PILOT — CHALLENGE 2 (Driver Intelligence)
Data layer: turn cabin images + labels into something a CNN can train on.

THE ONE IDEA THAT MATTERS HERE: Leave-One-Trip-Out (LOTO)
---------------------------------------------------------
Each practice trip is essentially ONE subject showing ONE or two states
(e.g. T05-Sample = 100% microsleep). If we pooled all frames and split them
randomly, the model would learn "this face = microsleep" instead of the real
cue (eyes closed for a long time). It would look great locally and COLLAPSE on
the 10 scored trips (new people it has never seen).

So we never split randomly. We hold out an ENTIRE trip for validation and
train on the others. The validation trip is an unseen person/situation — the
honest mirror of the real scored set. Average over all 6 hold-outs = the
number we trust.

This module only prepares data; training lives in the next file.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# --- the benchmark's dataset toolkit is private and not in this repo ----------------
# Imported lazily so that everything in here which does NOT need trip footage
# (the label contract, the transforms) keeps working without it.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STARTERKIT = _PROJECT_ROOT / "starterkit"


def _trip_dataset_cls():
    """Return team_kit's TripDataset, with a survivable error if it is absent."""
    if str(_STARTERKIT) not in sys.path:
        sys.path.insert(0, str(_STARTERKIT))
    try:
        from team_kit.dataset_loader import TripDataset
    except ImportError as exc:  # pragma: no cover - depends on local checkout
        raise ImportError(
            "This function needs the private trip toolkit (starterkit/team_kit), "
            "which is not distributed with this repository. The open-data "
            "evaluation and the drowsiness demo do not require it."
        ) from exc
    return TripDataset


# ===========================================================================
# SECTION 1 — THE LABEL CONTRACT (fixed, stable ordering)
# ===========================================================================
# The class-index mapping MUST be identical everywhere (training, inference,
# submission). We freeze it here as a single source of truth. Never reorder:
# a model trained with this order outputs logits in this order.
# ---------------------------------------------------------------------------
# The label contract lives in labels.py so it can be imported without torch,
# OpenCV or the private trip toolkit. Re-exported here for existing callers.
from guardian.challenge2.labels import (  # noqa: E402,F401
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
)

# ImageNet statistics — we fine-tune an ImageNet-pretrained backbone, so we
# normalise inputs the same way that backbone was trained.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# All 6 practice trips carry ground-truth driver_state.
PRACTICE_TRIP_IDS: tuple[str, ...] = (
    "T01-Sample", "T02-Sample", "T03-Sample",
    "T04-Sample", "T05-Sample", "T06-Sample",
)


# ===========================================================================
# SECTION 2 — INDEX BUILDING (scan trips once, resolve image paths + labels)
# ===========================================================================


@dataclass(frozen=True)
class DriverSample:
    """One training example: where the image is, what state it shows."""

    image_path: str   # absolute path to the driver frame on disk
    label: str        # one of CLASS_NAMES
    label_idx: int    # its index in CLASS_TO_IDX
    trip_id: str      # which trip it came from (needed for LOTO grouping)
    frame_id: int


def build_driver_index(
    trip_ids: Optional[list[str]] = None,
    data_root: str | Path = r"C:/guardian_data",
) -> list[DriverSample]:
    """
    Scan the given trips and return one DriverSample per labelled frame.

    We resolve each driver image to an actual file path here (trying the
    possible extensions once) so the Dataset's __getitem__ can stay tiny and
    fast — it just reads a file, no JSON parsing, and it pickles cleanly for
    multi-worker DataLoaders.

    Frames whose driver_state is missing/unknown (e.g. redacted scored trips)
    are skipped — you cannot train on a label that is not there.
    """
    import os

    trip_ids = list(trip_ids) if trip_ids else list(PRACTICE_TRIP_IDS)
    data_root = Path(data_root)

    samples: list[DriverSample] = []
    for trip_id in trip_ids:
        trip = _trip_dataset_cls()(data_root / trip_id)
        for frame in trip.iter_frames():
            state = frame.driver_state
            if state not in CLASS_TO_IDX:
                continue  # redacted / unknown -> not trainable
            img_path = _resolve_driver_image(trip.driver_dir, frame.frame_id)
            if img_path is None:
                continue  # image missing on disk -> skip rather than crash
            samples.append(
                DriverSample(
                    image_path=os.fspath(img_path),
                    label=state,
                    label_idx=CLASS_TO_IDX[state],
                    trip_id=trip_id,
                    frame_id=int(frame.frame_id),
                )
            )
    return samples


def _resolve_driver_image(driver_dir: Path, frame_id: int) -> Optional[Path]:
    """Find the driver frame file, trying the extensions the kit may emit."""
    stem = f"frame_{frame_id:06d}"
    for ext in (".png", ".jpg", ".jpeg"):
        path = driver_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def class_distribution(samples: list[DriverSample]) -> dict[str, int]:
    """Count samples per class (for spotting imbalance)."""
    counts = {name: 0 for name in CLASS_NAMES}
    for s in samples:
        counts[s.label] += 1
    return counts


# ===========================================================================
# SECTION 3 — TRANSFORMS (augment for train, deterministic for inference)
# ===========================================================================
# We keep augmentation CONSERVATIVE and, crucially, avoid horizontal flips:
# flipping swaps left/right, which changes "looking to the side" (distracted)
# and head-pose cues. At inference we do resize + normalise only — the README
# stresses determinism (same input -> same output).
# ---------------------------------------------------------------------------


def build_transforms(train: bool, img_size: int = 224):
    """Return a torchvision transform pipeline."""
    from torchvision import transforms

    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            # Mild scale/translation jitter — never a horizontal flip.
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# ===========================================================================
# SECTION 4 — THE DATASET
# ===========================================================================


class DriverStateDataset(Dataset):
    """A torch Dataset over a fixed list of DriverSample."""

    def __init__(self, samples: list[DriverSample], transform) -> None:
        """
        Args:
            samples:   the examples this dataset serves.
            transform: a torchvision transform (from build_transforms).
        """
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        # cv2 reads BGR; models expect RGB. Convert once here.
        bgr = cv2.imread(sample.image_path)
        if bgr is None:
            # A corrupt/unreadable frame: fall back to a black image so one
            # bad file never crashes an epoch. (Rare; see the known CRC issue.)
            bgr = np.zeros((360, 640, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        image = self.transform(rgb)
        return image, sample.label_idx


# ===========================================================================
# SECTION 5 — LEAVE-ONE-TRIP-OUT SPLIT
# ===========================================================================


@dataclass
class LotoFold:
    """One cross-validation fold: which trip is held out, and its datasets."""

    val_trip_id: str
    train_dataset: DriverStateDataset
    val_dataset: DriverStateDataset
    n_train: int
    n_val: int


def leave_one_trip_out(
    all_samples: list[DriverSample],
    img_size: int = 224,
) -> Iterator[LotoFold]:
    """
    Yield one LotoFold per trip: that trip is validation, the rest are training.

    This is the honest evaluation protocol for Challenge 2. Because the
    validation trip's subject never appears in training, the fold score
    reflects generalisation to unseen people — exactly the scored-set setting.
    """
    trip_ids = sorted({s.trip_id for s in all_samples})

    for held_out in trip_ids:
        train_samples = [s for s in all_samples if s.trip_id != held_out]
        val_samples = [s for s in all_samples if s.trip_id == held_out]

        yield LotoFold(
            val_trip_id=held_out,
            train_dataset=DriverStateDataset(
                train_samples, build_transforms(train=True, img_size=img_size)
            ),
            val_dataset=DriverStateDataset(
                val_samples, build_transforms(train=False, img_size=img_size)
            ),
            n_train=len(train_samples),
            n_val=len(val_samples),
        )


# ===========================================================================
# SECTION 6 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    import os
    data_root = os.getenv("GUARDIAN_DATA_ROOT", r"C:/guardian_data")

    print("Building driver index over practice trips...")
    samples = build_driver_index(data_root=data_root)
    print(f"Total labelled driver frames: {len(samples)}")

    print("\nOverall class distribution:")
    for name, n in class_distribution(samples).items():
        print(f"  {name:12s} {n}")

    print("\nPer-trip class distribution (shows the identity-overfit trap):")
    for tid in PRACTICE_TRIP_IDS:
        trip_samples = [s for s in samples if s.trip_id == tid]
        dist = class_distribution(trip_samples)
        active = {k: v for k, v in dist.items() if v}
        print(f"  {tid:12s} n={len(trip_samples):4d}  {active}")

    print("\nLeave-One-Trip-Out folds:")
    for fold in leave_one_trip_out(samples):
        print(f"  hold out {fold.val_trip_id:12s} -> train {fold.n_train:4d} / val {fold.n_val:4d}")

    # Load one example to confirm the tensor shape the model will receive.
    fold = next(leave_one_trip_out(samples))
    image, label = fold.train_dataset[0]
    print(f"\nSample tensor: shape={tuple(image.shape)} dtype={image.dtype} "
          f"label={label} ({IDX_TO_CLASS[label]})")
