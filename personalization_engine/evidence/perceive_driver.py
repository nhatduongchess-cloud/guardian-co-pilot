"""
perceive_driver.py
==================

VERTICAL 1 — EVIDENCE · the "V3 perception" step for REAL evaluation data.

The scored evaluation trips are REDACTED: the per-frame driver state and TTC
(the things teams must predict) are stripped from the JSON. To feed Vertical 1
with a REAL fatigue signal, we run the team's own Challenge-2 driver-state model
(resnet18, artifacts/challenge2_driver_state.pt) over the REAL driver frames and
turn its output into a per-frame drowsiness probability.

This is the honest "perception → personalization" loop: our model looks at the
real footage, and Vertical 1 personalises on what the model actually saw — no
redacted ground truth, nothing fabricated.

Frames are read straight out of the dataset .zip in memory (no 2 GB extraction).

RUN (from personalization_engine/, with the repo root on the path):
    python -m evidence.perceive_driver
    GUARDIAN_DATASET_ZIP=/path/to/Hackathon_Dataset_Redacted.zip python -m evidence.perceive_driver

Output: evidence/driver_state_realdata.json
"""

import io
import json
import os
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

# The Challenge-2 model lives in the sibling `guardian` package at the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_ZIP = os.getenv(
    "GUARDIAN_DATASET_ZIP",
    str(Path.home() / "Downloads" / "Hackathon_Dataset_Redacted.zip"),
)
_CKPT = os.getenv("GUARDIAN_DRIVER_CKPT", str(_REPO_ROOT / "artifacts" / "challenge2_driver_state.pt"))
_SAMPLE_EVERY = int(os.getenv("GUARDIAN_FRAME_STRIDE", "40"))  # ~45 frames / 90s trip
_OUT = Path(__file__).with_name("driver_state_realdata.json")


def _drowsiness_proxy(probs_row, idx) -> float:
    """
    A PERCLOS-like drowsiness probability in [0,1] from the 5-class softmax:
    drowsy + microsleep + half of yawning. Grounded in the model's real output.
    """
    d = probs_row[idx["drowsy"]] + probs_row[idx["microsleep"]] + 0.5 * probs_row[idx["yawning"]]
    return float(min(1.0, d))


def run() -> dict:
    import cv2
    import numpy as np
    import torch

    from guardian.challenge2.dataset import IDX_TO_CLASS, CLASS_TO_IDX, build_transforms
    from guardian.challenge2.model import load_checkpoint

    zip_path = Path(_DEFAULT_ZIP)
    if not zip_path.exists():
        raise SystemExit(f"Dataset zip not found: {zip_path}\n"
                         f"Set GUARDIAN_DATASET_ZIP to its location.")
    if not Path(_CKPT).exists():
        raise SystemExit(f"Driver-state checkpoint not found: {_CKPT}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(_CKPT, device=device)
    model.eval()
    tf = build_transforms(train=False)
    print(f"model: {meta.arch}  classes={meta.class_names}  device={device}")

    zf = zipfile.ZipFile(zip_path)
    # Discover trip ids from the driver-frame paths.
    all_names = zf.namelist()
    trips = sorted({
        n.split("/")[1] for n in all_names
        if "/driver/frame_" in n and n.endswith(".jpg")
    })
    print(f"trips found: {trips}")

    result = {"source": zip_path.name, "stride": _SAMPLE_EVERY,
              "model": meta.arch, "classes": list(meta.class_names), "trips": {}}

    for tid in trips:
        frames = sorted(n for n in all_names
                        if f"/{tid}/driver/frame_" in n and n.endswith(".jpg"))[::_SAMPLE_EVERY]
        t0 = time.time()
        batch = []
        for n in frames:
            arr = np.frombuffer(zf.read(n), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            batch.append(tf(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x)["logits"], 1).cpu().numpy()
        preds = probs.argmax(1)
        drowsiness = [_drowsiness_proxy(probs[i], CLASS_TO_IDX) for i in range(len(preds))]
        dist = Counter(IDX_TO_CLASS[int(i)] for i in preds)
        dt = time.time() - t0

        result["trips"][tid] = {
            "n_frames_scored": len(frames),
            "class_distribution": dict(dist),
            "drowsiness_series": [round(v, 4) for v in drowsiness],
            "drowsiness_mean": round(float(sum(drowsiness) / len(drowsiness)), 4),
            "drowsiness_max": round(float(max(drowsiness)), 4),
            "seconds": round(dt, 2),
        }
        print(f"  {tid}: {len(frames)} frames · {dict(dist)} · "
              f"drowsiness mean={result['trips'][tid]['drowsiness_mean']:.3f} "
              f"max={result['trips'][tid]['drowsiness_max']:.3f} ({dt:.1f}s)")

    _OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {_OUT}")
    return result


if __name__ == "__main__":
    run()
