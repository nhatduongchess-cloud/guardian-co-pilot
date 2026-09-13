"""
analyze_dataset.py
==================

Runs the FULL Vertical-1 analysis over a reviewer' real frame dataset and renders
an explainable replay you can screen-record:

  real driver frame  ->  Challenge-2 perception (driver state + drowsiness)
                     ->  Vertical 1 (state machine · advisory threshold · giải thích)
                     ->  annotated video overlaid on the ACTUAL frames.

Two output modes:
  * default  : write an MP4 (demo/vertical1_dataset_analysis.mp4) — robust, no display needed.
  * --show   : live OpenCV window (press Q to quit) — record this window directly.

RUN (from personalization_engine/):
    python -m evidence.analyze_dataset                     # all 10 trips -> MP4
    python -m evidence.analyze_dataset --trips T09d,T05d   # only some trips
    python -m evidence.analyze_dataset --show              # live window to record
    GUARDIAN_DATASET_ZIP=... python -m evidence.analyze_dataset
"""

import argparse
import gzip
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[1]
_REPO = _ROOT.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ingestion.feature_store import InMemoryFeatureStore
from memory.store import InMemoryMemoryStore
from models.telemetry import EventType, TelemetryEvent
from reasoning.phi3_runtime import DisabledLLMBackend, Phi3Runtime
from reasoning.reasoning_engine import ReasoningEngine
from services.explanation_service import ExplanationService
from services.memory_service import DriverMemoryService

_ZIP = os.getenv("GUARDIAN_DATASET_ZIP", str(Path.home() / "Downloads" / "Hackathon_Dataset_Redacted.zip"))
_CKPT = os.getenv("GUARDIAN_DRIVER_CKPT", str(_REPO / "artifacts" / "challenge2_driver_state.pt"))
_OUT = _ROOT / "demo" / "vertical1_dataset_analysis.mp4"
_DRIVER = "driver_carla"

W, H = 1280, 720
OUT_FPS = 15
BG = (7, 19, 31); PANEL = (17, 34, 49); INK = (234, 245, 244); MUTED = (150, 173, 178)
CYAN = (80, 223, 202); BLUE = (103, 169, 255); GREEN = (99, 230, 166)
ORANGE = (255, 158, 100); RED = (255, 111, 115); VIOLET = (199, 155, 255)
STATE_COLOR = {"alert": GREEN, "distracted": ORANGE, "drowsy": BLUE, "microsleep": RED, "yawning": VIOLET}

_EVENT_MAP = {
    "motorcycle_cut_in": ("OBJECT_RISK", "xe máy tạt đầu", "WARN"),
    "lead_brake": ("OBJECT_RISK", "xe phía trước phanh gấp", "BRAKE_ASSIST"),
    "pedestrian_jaywalk": ("OBJECT_RISK", "người đi bộ băng qua đường", "BRAKE_ASSIST"),
    "stopped_vehicle_ahead": ("OBJECT_RISK", "xe dừng phía trước", "WARN"),
}


def _font(name, size):
    for p in (rf"C:\Windows\Fonts\{name}", f"/usr/share/fonts/truetype/dejavu/{name}"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_BRAND = _font("segoeuib.ttf", 26); F_H = _font("segoeuib.ttf", 30); F_BIG = _font("segoeuib.ttf", 40)
F_MED = _font("segoeui.ttf", 22); F_LAB = _font("segoeui.ttf", 17); F_BODY = _font("segoeui.ttf", 22)
F_MONO = _font("consola.ttf", 20); F_MONO_S = _font("consola.ttf", 16)


def _wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def _panel(d, box, fill=PANEL, outline=(40, 60, 78)):
    d.rounded_rectangle(box, radius=14, fill=fill, outline=outline, width=1)


def _bar(d, x0, x1, y, frac, color, h=14):
    d.rounded_rectangle((x0, y, x1, y + h), radius=h // 2, fill=(26, 42, 56))
    d.rounded_rectangle((x0, y, int(x0 + max(0.02, frac) * (x1 - x0)), y + h), radius=h // 2, fill=color)


def load_perception():
    import torch
    from guardian.challenge2.dataset import build_transforms, IDX_TO_CLASS, CLASS_TO_IDX
    from guardian.challenge2.model import load_checkpoint
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(_CKPT, device=dev)
    model.eval()
    return model, build_transforms(train=False), IDX_TO_CLASS, CLASS_TO_IDX, dev, torch


def perceive(model, tf, idx, dev, torch, rgb_frames):
    """Return (states, drowsiness[]) for a list of RGB ndarrays."""
    states, drows = [], []
    B = 64
    for i in range(0, len(rgb_frames), B):
        batch = torch.stack([tf(f) for f in rgb_frames[i:i + B]]).to(dev)
        with torch.no_grad():
            probs = torch.softmax(model(batch)["logits"], 1).cpu().numpy()
        for row in probs:
            states.append(idx[int(row.argmax())])
            drows.append(float(min(1.0, row[CLASS_TO_IDX_G["drowsy"]] +
                                   row[CLASS_TO_IDX_G["microsleep"]] +
                                   0.5 * row[CLASS_TO_IDX_G["yawning"]])))
    return states, drows


def compose(driver_img, tid, scen, frame_no, n_frames, state, drow,
            v1_state, thr, floor, conf, expl, phase):
    """Build one 1280x720 annotated frame (PIL RGB) and return a BGR ndarray."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # header
    d.text((40, 30), "GUARDIAN", font=F_BRAND, fill=CYAN)
    d.text((40 + d.textlength("GUARDIAN", font=F_BRAND) + 12, 33),
           "Vertical 1 · phân tích trên dataset thật của BGK", font=F_MED, fill=INK)
    rt = f"{tid} · frame {frame_no}/{n_frames}"
    d.text((W - 40 - d.textlength(rt, font=F_MONO_S), 36), rt, font=F_MONO_S, fill=MUTED)
    d.line((40, 72, W - 40, 72), fill=(40, 60, 78), width=1)
    d.text((40, 84), scen, font=F_H, fill=INK)

    # left: real driver frame
    box_x, box_y, box_w, box_h = 40, 140, 600, 470
    _panel(d, (box_x, box_y, box_x + box_w, box_y + box_h), fill=(2, 8, 13))
    di = driver_img.copy()
    di.thumbnail((box_w - 24, box_h - 60))
    ox = box_x + (box_w - di.width) // 2
    oy = box_y + 16
    img.paste(di, (ox, oy))
    d.text((box_x + 16, box_y + box_h - 34), "camera tài xế (frame thật từ dataset)", font=F_LAB, fill=MUTED)

    # per-frame perception overlay (bottom-left of the image)
    sc = STATE_COLOR.get(state, CYAN)
    d.rectangle((ox, oy + di.height - 42, ox + di.width, oy + di.height), fill=(0, 0, 0))
    d.text((ox + 10, oy + di.height - 38), state.upper(), font=F_BRAND, fill=sc)
    d.text((ox + di.width - 120, oy + di.height - 34), f"drowsy {drow:.2f}", font=F_MONO_S, fill=sc)

    # right column panels
    rx = 668
    _panel(d, (rx, 140, 1240, 300))
    d.text((rx + 20, 158), "PERCEPTION (Challenge-2 · resnet18)", font=F_LAB, fill=MUTED)
    d.text((rx + 20, 186), state, font=F_BIG, fill=sc)
    d.text((rx + 20, 246), "Drowsiness", font=F_LAB, fill=MUTED)
    _bar(d, rx + 130, 1210, 250, drow, RED if drow > 0.6 else ORANGE if drow > 0.3 else GREEN)
    d.text((rx + 20 + 0, 270), f"{drow:.2f}", font=F_MONO_S, fill=MUTED)

    _panel(d, (rx, 316, 1240, 470))
    d.text((rx + 20, 334), "VERTICAL 1 — advisory", font=F_LAB, fill=MUTED)
    d.text((rx + 20, 360), f"Trạng thái: {v1_state}", font=F_MED, fill=CYAN if v1_state == "PERSONALIZED" else INK)
    # gauge
    gx0, gx1, gy = rx + 20, 1210, 424
    d.rounded_rectangle((gx0, gy, gx1, gy + 9), radius=5, fill=(26, 42, 56))
    for val, col, lab in [(0.70, MUTED, "0.70"), (0.85, RED, "floor 0.85")]:
        tx = int(gx0 + val * (gx1 - gx0))
        d.line((tx, gy - 10, tx, gy + 19), fill=col, width=2)
        d.text((tx - d.textlength(lab, font=F_MONO_S) / 2, gy - 30), lab, font=F_MONO_S, fill=col)
    mx = int(gx0 + thr * (gx1 - gx0)); mc = RED if floor else CYAN
    d.ellipse((mx - 10, gy - 5, mx + 10, gy + 14), fill=mc, outline=BG, width=3)
    d.text((rx + 20, 400), f"Ngưỡng mệt: {thr:.3f}", font=F_MED, fill=mc)
    if floor:
        d.text((rx + 250, 400), "SAFETY FLOOR", font=F_MONO_S, fill=RED)

    # explanation panel
    _panel(d, (40, 500, 640, 610))
    d.rounded_rectangle((40, 500, 45, 610), radius=3, fill=CYAN)
    d.text((60, 512), "GIẢI THÍCH TIẾNG VIỆT", font=F_LAB, fill=CYAN)
    yy = 540
    for line in _wrap(d, expl or "…", F_BODY, 560)[:3]:
        d.text((60, yy), line, font=F_BODY, fill=(220, 236, 236)); yy += 28

    # phase / footer
    d.text((668, 500), phase, font=F_MONO_S, fill=CYAN)
    d.text((40, 668), "V1 chỉ TƯ VẤN — lệnh an toàn do V4 (deterministic) quyết định.",
           font=F_MONO_S, fill=(90, 110, 120))

    return np.asarray(img)[:, :, ::-1].copy()  # RGB -> BGR


# module-level class index for the perceive() closure
CLASS_TO_IDX_G = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", default="", help="comma list e.g. T09d,T05d (default: all)")
    ap.add_argument("--stride", type=int, default=12, help="sample every Nth driver frame")
    ap.add_argument("--show", action="store_true", help="live OpenCV window instead of MP4")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames per trip (0=all sampled)")
    args = ap.parse_args()

    if not Path(_ZIP).exists():
        raise SystemExit(f"Dataset zip not found: {_ZIP} (set GUARDIAN_DATASET_ZIP).")

    global CLASS_TO_IDX_G
    model, tf, IDX, C2IDX, dev, torch = load_perception()
    CLASS_TO_IDX_G = C2IDX
    print(f"perception: resnet18 · {dev}")

    zf = zipfile.ZipFile(_ZIP)
    all_names = zf.namelist()
    trips = sorted({n.split("/")[1] for n in all_names if "/driver/frame_" in n and n.endswith(".jpg")})
    if args.trips:
        want = {t.strip() for t in args.trips.split(",")}
        trips = [t for t in trips if t in want]
    print(f"trips: {trips}")

    # V1 stack (one driver across trips)
    features, memory = InMemoryFeatureStore(), InMemoryMemoryStore()
    phi3 = Phi3Runtime(); backend = phi3 if phi3.is_available() else DisabledLLMBackend()
    mem_svc = DriverMemoryService(features, memory)
    exp_svc = ExplanationService(features, memory, ReasoningEngine(backend=backend))

    # output sink
    proc = None; cv2 = None
    if args.show:
        import cv2 as _cv2; cv2 = _cv2
        cv2.namedWindow("Guardian V1 · dataset analysis", cv2.WINDOW_NORMAL)
    else:
        _OUT.parent.mkdir(exist_ok=True)
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
             "-r", str(OUT_FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "veryfast", "-movflags", "+faststart", str(_OUT)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def emit(bgr, repeat=1):
        if args.show:
            cv2.imshow("Guardian V1 · dataset analysis", bgr)
            if cv2.waitKey(int(1000 / OUT_FPS)) & 0xFF == ord("q"):
                raise KeyboardInterrupt
        else:
            for _ in range(repeat):
                proc.stdin.write(bgr.tobytes())

    try:
        for ti, tid in enumerate(trips):
            doc = json.loads(gzip.decompress(zf.read(
                f"Hackathon_Dataset_Redacted/{tid}/{tid}.json.gz")).decode("utf-8"))
            scen = doc.get("metadata", {}).get("description", "").replace("DEBUG 30s: ", "")
            names = sorted(n for n in all_names if f"/{tid}/driver/frame_" in n and n.endswith(".jpg"))[::args.stride]
            if args.max_frames:
                names = names[:args.max_frames]

            # decode + perceive whole trip
            pil_frames = [Image.open(io.BytesIO(zf.read(n))).convert("RGB") for n in names]
            rgb = [np.asarray(p) for p in pil_frames]
            states, drows = perceive(model, tf, IDX, dev, torch, rgb)

            # V1 state BEFORE this trip (what it currently advises)
            before = mem_svc.get_memory(_DRIVER)
            thr_b = mem_svc.get_personalized_threshold(_DRIVER, EventType.FATIGUE_ALERT)

            print(f"  {tid}: {len(names)} frames · states={set(states)} · drowsy~{np.mean(drows):.2f}")

            # play frames
            for k, (p, st, dr) in enumerate(zip(pil_frames, states, drows)):
                bgr = compose(p, tid, scen, k + 1, len(names), st, dr,
                              before.state.value, thr_b.threshold, thr_b.safety_floor_applied,
                              thr_b.confidence, "", f"đang phân tích · chuyến {ti+1}/{len(trips)}")
                emit(bgr)

            # feed V1: fatigue from drowsiness + object events from events_log
            dur = doc.get("metadata", {}).get("duration_sec", 90) or 90
            for j, dr in enumerate(drows):
                mem_svc.ingest_event(TelemetryEvent(
                    trip_id=tid, driver_id=_DRIVER, source="V3_PERCEPTION",
                    event_type="FATIGUE_ALERT", risk_embedding=[dr, 0.5, 0.0],
                    scalar_features={"perclos": dr}))
            headline_expl = ""
            for ev in doc.get("events_log", []):
                m = _EVENT_MAP.get(ev["type"])
                if not m:
                    continue
                et, label, act = m
                e = TelemetryEvent(trip_id=tid, driver_id=_DRIVER, source="V4_WORLD_MODEL",
                                   event_type=et, risk_embedding=[0.0, 0.5, 1.0],
                                   scalar_features={"object_class": label},
                                   safety_kernel_decision={"action": act, "threshold_used": 2.0})
                mem_svc.ingest_event(e)
                if not headline_expl:
                    headline_expl = exp_svc.explain_event(e.event_id).text_vi
            mem = mem_svc.end_trip(_DRIVER, tid)
            thr_a = mem_svc.get_personalized_threshold(_DRIVER, EventType.FATIGUE_ALERT)
            if not headline_expl:
                # no object event -> explain the drowsiest fatigue frame
                evs = features.get_trip_events(tid)
                fe = max((x for x in evs if x.event_type == EventType.FATIGUE_ALERT),
                         key=lambda x: x.scalar_features.perclos or 0)
                headline_expl = exp_svc.explain_event(fe.event_id).text_vi

            # hold ~3s on the result of this trip (last frame + updated V1)
            last = pil_frames[-1]
            for _ in range(OUT_FPS * 3):
                bgr = compose(last, tid, scen, len(names), len(names),
                              states[-1], drows[-1], mem.state.value, thr_a.threshold,
                              thr_a.safety_floor_applied, thr_a.confidence, headline_expl,
                              f"KẾT QUẢ chuyến {ti+1}: {thr_b.threshold:.3f} → {thr_a.threshold:.3f}")
                emit(bgr)
    except KeyboardInterrupt:
        print("stopped.")
    finally:
        if args.show:
            cv2.destroyAllWindows()
        else:
            proc.stdin.close(); proc.wait()
            print(f"\nWrote {_OUT}  ({_OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
