"""
blendshapes.py
==============

THE SAME EYES AND JAW, POINTED AT SOMEONE ELSE'S FACES.

`guardian/challenge2/features.py` reads MediaPipe FaceLandmarker blendshapes out
of cabin video. It is also welded to the private trip dataset and to the
benchmark's loader, so it cannot run on anything else. This module extracts the
identical two signals from a plain image, with no dataset and no loader in
sight, so the thresholds can be checked against public faces.

Why blendshapes rather than raw pixels, restated because it is the whole reason
the rule engine beat the CNN: Google normalises these scores per face. An
`eyeBlink` of 0.6 means "this person's eyes are closed" whoever this person is.
A pixel model, handed six drivers, learns the six people instead.

The model file (`face_landmarker.task`, Apache-2.0, Google) is downloaded on
first use into `artifacts/models/` and cached.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACE_MODEL_PATH = _PROJECT_ROOT / "artifacts" / "models" / "face_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

#: Exactly the blendshapes challenge2 keeps. Eye closure drives PERCLOS; jaw
#: drives yawning. Anything else is decoration for this question.
EYE_BLENDSHAPES = ("eyeBlinkLeft", "eyeBlinkRight")
JAW_BLENDSHAPE = "jawOpen"


def ensure_face_model(path: Path = FACE_MODEL_PATH) -> Path:
    """Download the FaceLandmarker bundle once; return its path."""
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FACE_MODEL_URL, path)  # noqa: S310 - fixed Google URL
    return path


@dataclass(frozen=True)
class FaceSignals:
    """What one image tells us about one face."""

    #: Mean of the left and right eye-closure blendshapes, 0..1. This is the
    #: per-frame quantity PERCLOS averages over a window.
    eye_blink: float
    #: Mouth opening, 0..1. Yawning is the only thing that drives this hard.
    jaw_open: float
    #: False when MediaPipe found no face at all — counted, never silently
    #: folded in as a zero, because "no face" and "eyes open" are not the same
    #: statement and averaging them together would quietly bias every mean.
    face_found: bool

    @property
    def eyes_closed(self) -> bool:
        """The convention challenge2 uses when turning a frame into PERCLOS."""
        return self.eye_blink >= 0.5


class BlendshapeExtractor:
    """A thin, reusable wrapper over MediaPipe FaceLandmarker (CPU, image mode)."""

    def __init__(self, face_model_path: Optional[Path] = None) -> None:
        model_path = ensure_face_model(face_model_path or FACE_MODEL_PATH)

        # Imported lazily: mediapipe is a heavy import and the registry/report
        # code must stay usable (and testable) without it.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def __enter__(self) -> "BlendshapeExtractor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        closer = getattr(self._landmarker, "close", None)
        if closer is not None:
            closer()

    def extract(self, rgb: np.ndarray) -> FaceSignals:
        """
        Read eye and jaw signals from one RGB image.

        A frame with no detectable face returns `face_found=False` and zeros;
        callers are expected to exclude it rather than average it in.
        """
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.face_blendshapes:
            return FaceSignals(eye_blink=0.0, jaw_open=0.0, face_found=False)

        scores = {c.category_name: float(c.score) for c in result.face_blendshapes[0]}
        eye = float(np.mean([scores.get(name, 0.0) for name in EYE_BLENDSHAPES]))
        return FaceSignals(
            eye_blink=eye,
            jaw_open=scores.get(JAW_BLENDSHAPE, 0.0),
            face_found=True,
        )


def to_rgb(image) -> np.ndarray:
    """
    Coerce whatever the Hub hands back (PIL image, array) into contiguous RGB uint8.

    MediaPipe rejects anything else, and the datasets library is happy to return
    greyscale or RGBA depending on the upload.
    """
    array = np.array(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
    if array.ndim == 2:  # greyscale
        array = np.stack([array] * 3, axis=-1)
    if array.shape[-1] == 4:  # RGBA
        array = array[:, :, :3]
    return np.ascontiguousarray(array.astype(np.uint8))
