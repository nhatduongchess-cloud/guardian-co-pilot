"""
features.py
===========

GUARDIAN CO-PILOT — CHALLENGE 2 (Driver Intelligence)
Identity-invariant feature extraction from cabin images.

WHY THIS EXISTS — the lesson that reshaped Challenge 2
------------------------------------------------------
Fine-tuning a CNN on raw driver pixels FAILED (LOTO composite 18.5/100). The
reason, confirmed by looking at the data: every trip is a different person, and
600 frames at 20 FPS are near-duplicates. The effective sample size is ~6
scenes, not 3600 images, so the network memorised "who is this / which car" and
learned nothing about driver state. On an unseen person it collapsed to one
class.

The fix is to stop feeding it identity. We extract features that mean the SAME
THING on every face:

  * MediaPipe FaceLandmarker blendshapes -- `eyeBlink` (eye closure 0..1) and
    `jawOpen` (mouth opening 0..1). Google's model normalises these per face,
    so 0.6 means "eyes closed" whoever you are.
  * Head pose (pitch/yaw/roll) from the facial transformation matrix.
  * YOLOv8 COCO `cell phone` detection -- the giveaway for `distracted`.

Measured class separation on the practice trips (means):

    class        eyeBlink   jawOpen   phone-rate
    alert          0.065     0.014        7.5%
    distracted     0.075     0.079       73.3%
    drowsy         0.220     0.013        0.0%
    microsleep     0.664     0.024        0.0%
    yawning        0.278     0.437        0.0%

Crucially `drowsy` reads high on two DIFFERENT subjects (T02 0.182, T06 0.258),
which is exactly the cross-identity transfer the CNN could not achieve.

DESIGN — extract once, experiment many times
--------------------------------------------
Feature extraction is slow (MediaPipe runs on CPU). Classification experiments
are fast. So we extract ONCE per trip and cache to
`artifacts/features/<trip_id>.csv`; every later experiment reads the CSV and
runs in seconds. Under time pressure this is the difference between
iterating in seconds and iterating in minutes.

LICENCES (declared in the write-up)
-----------------------------------
  MediaPipe face_landmarker.task  Apache-2.0  (Google)
  YOLOv8n (ultralytics)           AGPL-3.0    -- see README's licence section
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd

# MediaPipe/TF print a lot of noise on import; keep the console readable.
warnings.filterwarnings("ignore")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# --- make the benchmark's dataset toolkit importable -------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STARTERKIT = _PROJECT_ROOT / "starterkit"
if str(_STARTERKIT) not in sys.path:
    sys.path.insert(0, str(_STARTERKIT))


def _trip_dataset_cls():
    """team_kit's TripDataset, imported on use: extracting features needs the
    private trip footage, but loading cached feature CSVs does not."""
    if str(_STARTERKIT) not in sys.path:
        sys.path.insert(0, str(_STARTERKIT))
    try:
        from team_kit.dataset_loader import TripDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Feature extraction needs the private trip toolkit "
            "(starterkit/team_kit). Cached features in artifacts/features/ "
            "load without it."
        ) from exc
    return TripDataset



# ===========================================================================
# SECTION 1 — CONFIGURATION
# ===========================================================================

DEFAULT_DATA_ROOT = Path(os.getenv("GUARDIAN_DATA_ROOT", r"C:/guardian_data"))
FEATURES_DIR = _PROJECT_ROOT / "artifacts" / "features"
FACE_MODEL_PATH = _PROJECT_ROOT / "artifacts" / "models" / "face_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Blendshapes we keep. A curated subset: the ones physiologically tied to
# drowsiness/yawning, rather than all 52 (which would just add noise).
_BLENDSHAPES_OF_INTEREST = (
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "jawOpen",
    "mouthPucker",
    "browDownLeft",
    "browDownRight",
)

# COCO class name YOLO uses for a phone.
_PHONE_CLASS = "cell phone"

# The final column order written to CSV (stable contract for later stages).
FEATURE_COLUMNS = (
    "frame_id", "timestamp", "face_found",
    "eye_blink_left", "eye_blink_right", "eye_blink",
    "eye_squint_left", "eye_squint_right",
    "jaw_open", "mouth_pucker", "brow_down_left", "brow_down_right",
    "head_pitch", "head_yaw", "head_roll",
    "phone_conf", "phone_detected",
)


# ===========================================================================
# SECTION 2 — HELPERS
# ===========================================================================


def ensure_face_model(path: Path = FACE_MODEL_PATH) -> Path:
    """Download the MediaPipe face model once if it is not already present."""
    if path.exists():
        return path
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe face model -> {path}")
    urllib.request.urlretrieve(FACE_MODEL_URL, path)
    return path


def _rotation_to_euler(matrix: np.ndarray) -> tuple[float, float, float]:
    """
    Convert a 3x3 rotation matrix to (pitch, yaw, roll) in degrees.

    Standard ZYX decomposition, with the gimbal-lock case handled so a
    near-vertical head never produces NaNs.
    """
    sy = float(np.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    if sy > 1e-6:
        pitch = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(-matrix[2, 0], sy)
        roll = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:  # gimbal lock
        pitch = np.arctan2(-matrix[1, 2], matrix[1, 1])
        yaw = np.arctan2(-matrix[2, 0], sy)
        roll = 0.0
    return (
        float(np.degrees(pitch)),
        float(np.degrees(yaw)),
        float(np.degrees(roll)),
    )


def _resolve_driver_image(driver_dir: Path, frame_id: int) -> Optional[Path]:
    """Find a driver frame file, trying the extensions the kit may emit."""
    for ext in (".png", ".jpg", ".jpeg"):
        path = driver_dir / f"frame_{frame_id:06d}{ext}"
        if path.exists():
            return path
    return None


# ===========================================================================
# SECTION 3 — THE EXTRACTOR
# ===========================================================================


@dataclass
class ExtractionStats:
    """Small summary returned after processing a trip."""

    trip_id: str
    n_frames: int
    n_face_found: int
    n_phone: int
    seconds: float

    @property
    def face_rate(self) -> float:
        return self.n_face_found / self.n_frames if self.n_frames else 0.0


class DriverFeatureExtractor:
    """
    Runs MediaPipe (face) and YOLOv8 (phone) over a trip's cabin frames.

    Both models are created ONCE and reused across trips -- loading them per
    frame would dominate the runtime.
    """

    def __init__(
        self,
        face_model_path: Path = FACE_MODEL_PATH,
        yolo_weights: str = "yolov8n.pt",
        phone_conf_threshold: float = 0.20,
        device: int | str = 0,
        batch_size: int = 32,
    ) -> None:
        """
        Args:
            face_model_path:      MediaPipe .task bundle (auto-downloaded).
            yolo_weights:         ultralytics weights (auto-downloaded).
            phone_conf_threshold: minimum confidence to call a phone present.
            device:               0 = first CUDA GPU, or "cpu".
            batch_size:           frames per YOLO batch (GPU efficiency).
        """
        self.phone_conf_threshold = phone_conf_threshold
        self.device = device
        self.batch_size = batch_size

        # --- MediaPipe FaceLandmarker (CPU) --------------------------------
        # NOTE: this mediapipe build exposes only the Tasks API, not the legacy
        # `mp.solutions`. Tasks also gives us blendshapes, which is what we want.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        ensure_face_model(face_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(face_model_path)
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._face = vision.FaceLandmarker.create_from_options(options)

        # --- YOLOv8 (GPU) ---------------------------------------------------
        from ultralytics import YOLO

        self._yolo = YOLO(yolo_weights)
        self._yolo_names = self._yolo.names

    # -- per-frame ----------------------------------------------------------

    def _face_features(self, bgr: np.ndarray) -> dict:
        """Extract blendshapes + head pose from one frame. Returns defaults if
        no face is found (e.g. driver turned fully away)."""
        empty = {
            "face_found": False,
            **{self._snake(name): 0.0 for name in _BLENDSHAPES_OF_INTEREST},
            "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
        }

        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        )
        result = self._face.detect(mp_image)
        if not result.face_blendshapes:
            return empty

        scores = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        out = {"face_found": True}
        for name in _BLENDSHAPES_OF_INTEREST:
            out[self._snake(name)] = scores.get(name, 0.0)

        # Head pose from the 4x4 facial transformation matrix (top-left 3x3).
        if result.facial_transformation_matrixes:
            matrix = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
            pitch, yaw, roll = _rotation_to_euler(matrix[:3, :3])
        else:
            pitch = yaw = roll = 0.0
        out.update(head_pitch=pitch, head_yaw=yaw, head_roll=roll)
        return out

    @staticmethod
    def _snake(blendshape_name: str) -> str:
        """'eyeBlinkLeft' -> 'eye_blink_left' so CSV columns stay conventional."""
        chars = []
        for ch in blendshape_name:
            if ch.isupper():
                chars.append("_")
                chars.append(ch.lower())
            else:
                chars.append(ch)
        return "".join(chars)

    def _phone_confidences(self, images: list[np.ndarray]) -> list[float]:
        """Best `cell phone` confidence per image (0.0 if none). Batched on GPU."""
        results = self._yolo.predict(
            images, verbose=False, conf=self.phone_conf_threshold, device=self.device
        )
        confidences = []
        for result in results:
            best = 0.0
            for box in result.boxes:
                if self._yolo_names[int(box.cls)] == _PHONE_CLASS:
                    best = max(best, float(box.conf))
            confidences.append(best)
        return confidences

    # -- per-trip -----------------------------------------------------------

    def extract_trip(
        self,
        trip_id: str,
        data_root: Path = DEFAULT_DATA_ROOT,
        progress_every: int = 300,
    ) -> tuple[pd.DataFrame, ExtractionStats]:
        """
        Extract features for every frame of one trip.

        Handles unreadable frames (there is a known corrupt JPEG in the dataset)
        by carrying the previous frame's features forward -- one bad file must
        never break a 1800-frame run.
        """
        t0 = time.time()
        trip = _trip_dataset_cls()(Path(data_root) / trip_id)

        frame_ids: list[int] = []
        timestamps: list[float] = []
        for frame in trip.iter_frames():
            frame_ids.append(int(frame.frame_id))
            timestamps.append(float(frame.timestamp))

        rows: list[dict] = []
        previous_row: Optional[dict] = None

        for start in range(0, len(frame_ids), self.batch_size):
            chunk_ids = frame_ids[start : start + self.batch_size]

            # Read each image ONCE and feed both models from the same array.
            images: list[Optional[np.ndarray]] = []
            for fid in chunk_ids:
                path = _resolve_driver_image(trip.driver_dir, fid)
                images.append(cv2.imread(str(path)) if path else None)

            # YOLO only sees readable frames; we re-align afterwards.
            readable_idx = [i for i, im in enumerate(images) if im is not None]
            phone_conf = [0.0] * len(images)
            if readable_idx:
                confs = self._phone_confidences([images[i] for i in readable_idx])
                for i, conf in zip(readable_idx, confs):
                    phone_conf[i] = conf

            for offset, fid in enumerate(chunk_ids):
                image = images[offset]
                if image is None:
                    # Corrupt/missing frame: reuse the last good row so the
                    # time series stays continuous instead of gaining a hole.
                    row = dict(previous_row) if previous_row else {
                        "face_found": False,
                        **{self._snake(n): 0.0 for n in _BLENDSHAPES_OF_INTEREST},
                        "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
                        "phone_conf": 0.0, "phone_detected": False,
                    }
                else:
                    row = self._face_features(image)
                    row["phone_conf"] = phone_conf[offset]
                    row["phone_detected"] = (
                        phone_conf[offset] >= self.phone_conf_threshold
                    )

                row["frame_id"] = fid
                row["timestamp"] = timestamps[start + offset]
                # Mean eye closure -- the single most useful drowsiness signal.
                row["eye_blink"] = (
                    row["eye_blink_left"] + row["eye_blink_right"]
                ) / 2.0
                rows.append(row)
                previous_row = row

            if progress_every and (start + self.batch_size) % progress_every < self.batch_size:
                done = min(start + self.batch_size, len(frame_ids))
                print(f"    {trip_id}: {done}/{len(frame_ids)} frames", flush=True)

        df = pd.DataFrame(rows)[list(FEATURE_COLUMNS)]
        stats = ExtractionStats(
            trip_id=trip_id,
            n_frames=len(df),
            n_face_found=int(df["face_found"].sum()),
            n_phone=int(df["phone_detected"].sum()),
            seconds=time.time() - t0,
        )
        return df, stats


# ===========================================================================
# SECTION 4 — CACHING API (what the rest of the pipeline calls)
# ===========================================================================


def feature_path(trip_id: str, features_dir: Path = FEATURES_DIR) -> Path:
    """Where a trip's cached feature CSV lives."""
    return Path(features_dir) / f"{trip_id}.csv"


def load_trip_features(
    trip_id: str, features_dir: Path = FEATURES_DIR
) -> pd.DataFrame:
    """Load cached features. Raises if extraction has not been run yet."""
    path = feature_path(trip_id, features_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached features for {trip_id} at {path}. "
            f"Run: python guardian/challenge2/features.py --trips {trip_id}"
        )
    return pd.read_csv(path)


def extract_trips(
    trip_ids: Iterable[str],
    data_root: Path = DEFAULT_DATA_ROOT,
    features_dir: Path = FEATURES_DIR,
    force: bool = False,
    extractor: Optional[DriverFeatureExtractor] = None,
) -> list[ExtractionStats]:
    """
    Extract and cache features for several trips.

    Skips trips whose CSV already exists unless `force=True` -- so an
    interrupted run can simply be restarted.
    """
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    trip_ids = list(trip_ids)
    pending = [
        t for t in trip_ids if force or not feature_path(t, features_dir).exists()
    ]
    if not pending:
        print("All requested trips already have cached features.")
        return []

    extractor = extractor or DriverFeatureExtractor()

    all_stats: list[ExtractionStats] = []
    for trip_id in pending:
        print(f"[extract] {trip_id}")
        df, stats = extractor.extract_trip(trip_id, data_root=data_root)
        df.to_csv(feature_path(trip_id, features_dir), index=False)
        print(
            f"    done: {stats.n_frames} frames, face {stats.face_rate:.1%}, "
            f"phone {stats.n_phone}, {stats.seconds:.0f}s "
            f"({stats.n_frames / max(stats.seconds, 1e-6):.0f} fps)"
        )
        all_stats.append(stats)
    return all_stats


# ===========================================================================
# SECTION 5 — CLI
# ===========================================================================
if __name__ == "__main__":
    import argparse

    PRACTICE = [f"T0{i}-Sample" for i in range(1, 7)]
    SCORED = [f"T{i:02d}d" for i in range(1, 11)]

    parser = argparse.ArgumentParser(description="Extract driver features.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--trips", nargs="*", default=None,
        help="Trip ids. Default: all 16 (practice + scored).",
    )
    parser.add_argument("--practice-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-extract cached trips.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.trips:
        trips = args.trips
    elif args.practice_only:
        trips = PRACTICE
    else:
        trips = PRACTICE + SCORED

    print(f"Extracting features for {len(trips)} trip(s)")
    stats = extract_trips(
        trips,
        data_root=Path(args.data_root),
        force=args.force,
        extractor=DriverFeatureExtractor(batch_size=args.batch_size),
    )

    if stats:
        total_frames = sum(s.n_frames for s in stats)
        total_seconds = sum(s.seconds for s in stats)
        print(f"\nTotal: {total_frames} frames in {total_seconds:.0f}s "
              f"({total_frames / max(total_seconds, 1e-6):.0f} fps)")
        print(f"Cached in {FEATURES_DIR}")
