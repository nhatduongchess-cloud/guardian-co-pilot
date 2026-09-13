"""
build_demo_video.py
===================

Renders a real MP4 "video demo" of Vertical 1's personalization story over the
10 real evaluation trips, then (optionally) embeds it into the summary page as
a base64 data URI so the <video> element plays everywhere — local file AND the
published artifact — with no external file needed.

Frames are drawn with Pillow (Vietnamese via Segoe UI) and encoded to browser-safe
H.264 (yuv420p, +faststart) with ffmpeg.

RUN (from personalization_engine/):
    python -m evidence.build_demo_video            # writes demo/vertical1_personalization_demo.mp4
    python -m evidence.build_demo_video --embed    # + inlines it into the summary HTML
"""

import argparse
import base64
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_ROOT = Path(__file__).resolve().parents[1]
_DEMO_DIR = _ROOT / "demo"
_OUT = _DEMO_DIR / "vertical1_personalization_demo.mp4"
_HTML = _ROOT / "Guardian-Vertical1-Personalization.html"

W, H = 1280, 720
SEC_PER_TRIP = 3

# palette (matches the summary page)
BG = (7, 19, 31)
PANEL = (13, 30, 45)
INK = (234, 245, 244)
MUTED = (156, 179, 184)
CYAN = (80, 223, 202)
BLUE = (103, 169, 255)
GREEN = (99, 230, 166)
ORANGE = (255, 158, 100)
RED = (255, 111, 115)
VIOLET = (199, 155, 255)

STATE_COLOR = {"alert": GREEN, "distracted": ORANGE, "drowsy": BLUE,
               "microsleep": RED, "yawning": VIOLET}

TRIPS = [
    ("T01d", "Highway chiều tối · xe máy tạt đầu", "alert", "alert 67% · distracted 33%", 0.18, "COLD_START", 0.70, 0.20, False,
     "Guardian phát hiện xe máy tạt đầu và cảnh báo. Chưa đủ dữ liệu về bạn nên dùng ngưỡng mặc định."),
    ("T02d", "Ngoại ô mưa · người băng đường", "alert", "alert 100%", 0.08, "WARMING", 0.41, 0.40, False,
     "Xe trước phanh gấp — Guardian hỗ trợ phanh. Đang học thói quen lái của bạn để điều chỉnh."),
    ("T03d", "Nội đô giờ cao điểm", "microsleep", "microsleep 100%", 0.97, "PERSONALIZED", 0.85, 0.60, True,
     "Bạn có dấu hiệu vi giấc ngủ — Guardian giữ ngưỡng ở mức an toàn tối đa, không nới lỏng."),
    ("T04d", "Cao tốc sương mù bình minh", "drowsy", "drowsy 93% · alert 7%", 0.93, "PERSONALIZED", 0.85, 0.80, True,
     "Xe dừng trong sương + tài xế buồn ngủ — Guardian cảnh báo và giữ ngưỡng chặn ở 0.85."),
    ("T05d", "Đường quê đơn điệu", "microsleep", "microsleep 100%", 0.99, "PERSONALIZED", 0.85, 1.00, True,
     "Lái đơn điệu gây vi giấc ngủ — Guardian không cho phép nới ngưỡng trễ nguy hiểm."),
    ("T06d", "Ngã tư mưa giữa trưa", "drowsy", "drowsy 84%", 0.87, "PERSONALIZED", 0.85, 1.00, True,
     "Người băng đường — Guardian hỗ trợ phanh. Mức buồn ngủ cao nên ngưỡng bị kẹp an toàn."),
    ("T07d", "Cao tốc chói nắng", "distracted", "distracted 89% · alert 11%", 0.00, "PERSONALIZED", 0.108, 1.00, False,
     "Bạn rất tỉnh táo — Guardian hạ ngưỡng để cảnh báo sớm hơn khi có lệch bất thường."),
    ("T08d", "Khu công trường · đóng làn", "alert", "alert 100%", 0.17, "PERSONALIZED", 0.414, 1.00, False,
     "Người đi bộ — Guardian hỗ trợ phanh. Ngưỡng cá nhân hoá theo mức tỉnh táo của bạn."),
    ("T09d", "Nội đô nửa đêm · tài xế buồn ngủ", "drowsy", "drowsy 100%", 0.86, "PERSONALIZED", 0.85, 1.00, True,
     "Kịch bản tài xế buồn ngủ — model phát hiện đúng, Guardian giữ ngưỡng chặn ở mức an toàn tối đa."),
    ("T10d", "Cao tốc đêm mưa", "yawning", "yawning/distracted", 0.20, "PERSONALIZED", 0.293, 1.00, False,
     "Xe máy tạt đầu — Guardian cảnh báo. Bạn khá tỉnh táo nên ngưỡng được cá nhân hoá thấp."),
]
SM = ["COLD_START", "WARMING", "PERSONALIZED"]


def _font(name, size):
    for p in (rf"C:\Windows\Fonts\{name}", f"/usr/share/fonts/truetype/dejavu/{name}"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.truetype("arial.ttf", size)


F_BRAND = _font("segoeuib.ttf", 30)
F_H = _font("segoeuib.ttf", 34)
F_BIG = _font("segoeuib.ttf", 46)
F_MED = _font("segoeui.ttf", 26)
F_LAB = _font("segoeui.ttf", 19)
F_MONO = _font("consola.ttf", 22)
F_MONO_S = _font("consola.ttf", 17)
F_BODY = _font("segoeui.ttf", 25)


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


def _panel(draw, box, fill=PANEL, outline=(40, 60, 78)):
    draw.rounded_rectangle(box, radius=16, fill=fill, outline=outline, width=1)


def render_trip(i, t):
    tid, scen, state, prob, drow, sm, thr, conf, floor, expl = t
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # header
    d.text((60, 46), "GUARDIAN", font=F_BRAND, fill=CYAN)
    d.text((60 + d.textlength("GUARDIAN", font=F_BRAND) + 14, 50),
           "Vertical 1 — Demo cá nhân hoá", font=F_MED, fill=INK)
    rt = f"Chuyến {i+1}/10 · {tid}"
    d.text((W - 60 - d.textlength(rt, font=F_MONO), 52), rt, font=F_MONO, fill=MUTED)
    d.line((60, 96, W - 60, 96), fill=(40, 60, 78), width=1)

    # scenario
    d.text((60, 116), scen, font=F_H, fill=INK)

    # state machine chips
    x = 60
    y = 172
    for k, s in enumerate(SM):
        on = (s == sm)
        w = d.textlength(s, font=F_MONO_S) + 24
        _panel(d, (x, y, x + w, y + 34),
               fill=CYAN if on else PANEL, outline=CYAN if on else (40, 60, 78))
        d.text((x + 12, y + 8), s, font=F_MONO_S, fill=(3, 25, 27) if on else MUTED)
        x += w + 10
        if k < 2:
            d.text((x - 4, y + 6), "›", font=F_MONO, fill=(70, 90, 100)); x += 18

    # left panel — perception
    _panel(d, (60, 232, 620, 470))
    d.text((84, 252), "PERCEPTION (MODEL ĐỌC FOOTAGE THẬT)", font=F_LAB, fill=MUTED)
    d.text((84, 288), state, font=F_BIG, fill=STATE_COLOR.get(state, CYAN))
    d.text((88, 352), prob, font=F_MONO_S, fill=MUTED)
    d.text((84, 392), "Drowsiness", font=F_LAB, fill=MUTED)
    # drowsiness bar
    bx0, bx1, by = 84, 596, 424
    d.rounded_rectangle((bx0, by, bx1, by + 16), radius=8, fill=(24, 40, 54))
    dc = RED if drow > 0.6 else ORANGE if drow > 0.3 else GREEN
    fillw = int(bx0 + max(0.02, drow) * (bx1 - bx0))
    d.rounded_rectangle((bx0, by, fillw, by + 16), radius=8, fill=dc)
    d.text((bx1 - 60, by + 22), f"{drow:.2f}", font=F_MONO_S, fill=dc)

    # right panel — threshold gauge
    _panel(d, (660, 232, 1220, 470))
    d.text((684, 252), "NGƯỠNG MỆT V1 (ADVISORY)", font=F_LAB, fill=MUTED)
    gx0, gx1, gy = 700, 1180, 330
    d.rounded_rectangle((gx0, gy, gx1, gy + 10), radius=5, fill=(24, 40, 54))
    # ticks
    for val, col, lab in [(0.70, MUTED, "0.70"), (0.85, RED, "floor 0.85")]:
        tx = int(gx0 + val * (gx1 - gx0))
        d.line((tx, gy - 12, tx, gy + 22), fill=col, width=2)
        d.text((tx - d.textlength(lab, font=F_MONO_S) / 2, gy - 34), lab, font=F_MONO_S, fill=col)
    # marker
    mx = int(gx0 + thr * (gx1 - gx0))
    mc = RED if floor else CYAN
    d.ellipse((mx - 11, gy - 6, mx + 11, gy + 16), fill=mc, outline=BG, width=3)
    d.text((mx - d.textlength(f"{thr:.3f}", font=F_MONO) / 2, gy + 28), f"{thr:.3f}", font=F_MONO, fill=mc)
    # confidence
    d.text((684, 392), "Confidence", font=F_LAB, fill=MUTED)
    cx0, cx1, cy = 800, 1080, 400
    d.rounded_rectangle((cx0, cy, cx1, cy + 12), radius=6, fill=(24, 40, 54))
    d.rounded_rectangle((cx0, cy, int(cx0 + conf * (cx1 - cx0)), cy + 12), radius=6, fill=BLUE)
    d.text((cx1 + 14, cy - 4), f"{conf:.0%}", font=F_MONO_S, fill=BLUE)
    if floor:
        _panel(d, (684, 430, 892, 462), fill=(40, 20, 22), outline=RED)
        d.text((700, 437), "SAFETY FLOOR", font=F_MONO_S, fill=RED)

    # explanation
    _panel(d, (60, 500, 1220, 648))
    d.rounded_rectangle((60, 500, 66, 648), radius=3, fill=CYAN)
    d.text((88, 516), "GIẢI THÍCH TIẾNG VIỆT · GROUNDED", font=F_LAB, fill=CYAN)
    yy = 550
    for line in _wrap(d, expl, F_BODY, 1090)[:3]:
        d.text((88, yy), line, font=F_BODY, fill=(220, 236, 236)); yy += 34

    # footer
    d.text((60, 672), "V1 chỉ TƯ VẤN — mọi lệnh an toàn do V4 (deterministic) quyết định.",
           font=F_MONO_S, fill=(90, 110, 120))
    return img


def render_card(title, subtitle):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W / 2 - d.textlength("GUARDIAN", font=F_BRAND) / 2, 250), "GUARDIAN", font=F_BRAND, fill=CYAN)
    d.text((W / 2 - d.textlength(title, font=F_BIG) / 2, 300), title, font=F_BIG, fill=INK)
    for k, line in enumerate(_wrap(d, subtitle, F_MED, 900)):
        d.text((W / 2 - d.textlength(line, font=F_MED) / 2, 380 + k * 36), line, font=F_MED, fill=MUTED)
    return img


def build(embed: bool):
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH.")
    _DEMO_DIR.mkdir(exist_ok=True)
    frames = Path(tempfile.mkdtemp(prefix="v1demo_"))

    seq = 0
    def save(im):
        nonlocal seq
        im.save(frames / f"f{seq:03d}.png"); seq += 1

    save(render_card("Vertical 1 — Personalization", "Demo trên 10 chuyến thật của cuộc thi"))
    for i, t in enumerate(TRIPS):
        save(render_trip(i, t))
    save(render_card("Hoàn tất", "154 test · eval 12/12 · 535 sự kiện thật · chặn hallucinate 100%"))

    cmd = [
        "ffmpeg", "-y", "-framerate", f"1/{SEC_PER_TRIP}",
        "-i", str(frames / "f%03d.png"),
        "-c:v", "libx264", "-r", "30", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-movflags", "+faststart", str(_OUT),
    ]
    print("Encoding:", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    shutil.rmtree(frames, ignore_errors=True)
    size = _OUT.stat().st_size
    print(f"Wrote {_OUT}  ({size/1024:.0f} KB)")

    if embed:
        _embed_into_html(size)


def _embed_into_html(size):
    b64 = base64.b64encode(_OUT.read_bytes()).decode("ascii")
    data_uri = f"data:video/mp4;base64,{b64}"
    html = _HTML.read_text(encoding="utf-8")
    # Replace the whole <video ...>...</video> block with an autoplaying embed.
    new_video = (
        '<video id="demoVideo" controls autoplay muted loop playsinline '
        'style="width:100%;display:block;aspect-ratio:16/9;background:#000;object-fit:contain">\n'
        f'        <source src="{data_uri}" type="video/mp4">\n'
        '      </video>'
    )
    html2 = re.sub(r'<video id="demoVideo".*?</video>', new_video, html, count=1, flags=re.S)
    # Show the video by default; keep the animated stage only as an error fallback.
    html2 = html2.replace('<div class="stage" id="stage">', '<div class="stage" id="stage" style="display:none">')
    if html2 == html:
        print("WARNING: video block not found; HTML unchanged.")
    else:
        _HTML.write_text(html2, encoding="utf-8")
        print(f"Embedded {size/1024:.0f} KB video into {_HTML.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true", help="inline the mp4 into the summary HTML")
    build(ap.parse_args().embed)
