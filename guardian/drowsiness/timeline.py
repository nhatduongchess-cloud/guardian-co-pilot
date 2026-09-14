"""
timeline.py
===========

DRAW THE ANSWER.

One row of ground truth, one row of what the engine said, on a shared time
axis, so agreement and disagreement are visible without reading a table. The
eye-closure trace runs underneath with the PERCLOS threshold marked, because
the interesting question is never "was it right" but "what was it looking at
when it was wrong".

Deliberately hand-written SVG rather than a plotting library:

  * it is text, so a reviewer can diff two runs and see what changed;
  * it drops straight into an HTML page with no runtime and no image hosting;
  * the repository stays installable with pandas and numpy alone.

NO DRIVER IMAGERY IS EVER RENDERED. The footage is licensed academic data
(NTHU-DDD) and is not mine to publish; the signals derived from it are.
"""

from __future__ import annotations

import html
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from guardian.challenge2.labels import CLASS_NAMES, IMPAIRED_STATES
from guardian.drowsiness.detect import TripResult, _segments

#: One colour per state. Impaired states run warm, clear states run cool, so a
#: reviewer can read the strip before reading the legend.
STATE_COLOURS: dict[str, str] = {
    "alert": "#3B7DD8",
    "distracted": "#8C6BD4",
    "drowsy": "#E8963C",
    "yawning": "#D9762B",
    "microsleep": "#C7452F",
}
UNKNOWN_COLOUR = "#9AA0A8"

_INK = "#17191C"
_SUB = "#585D65"
_LINE = "#D9D5CC"


def _colour(state: str) -> str:
    return STATE_COLOURS.get(state, UNKNOWN_COLOUR)


def _strip(
    segments: Sequence[tuple[str, int, int]],
    n_frames: int,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[str]:
    out: list[str] = []
    for state, start, length in segments:
        sx = x + width * (start / n_frames)
        sw = max(width * (length / n_frames), 0.6)
        out.append(
            f'<rect x="{sx:.2f}" y="{y:.1f}" width="{sw:.2f}" height="{height:.1f}" '
            f'fill="{_colour(state)}"><title>{html.escape(state)}</title></rect>'
        )
    return out


def render_timeline(
    result: TripResult,
    eye_closure: Optional[pd.Series] = None,
    perclos_threshold: float = 0.4,
    width: int = 960,
) -> str:
    """
    Render one trip as a standalone SVG string.

    `eye_closure` is the per-frame blend-shape eye signal, if available; the
    chart works without it, just with less to explain a disagreement by.
    """
    n = result.n_frames
    if n == 0:
        raise ValueError(f"{result.trip_id} has no frames to draw.")

    # pad_r has to clear the "0.4 closed" threshold label sitting outside the plot.
    pad_l, pad_r, pad_t = 108, 78, 40
    plot_w = width - pad_l - pad_r
    strip_h, gap = 26, 10
    trace_h = 74 if eye_closure is not None else 0
    height = pad_t + strip_h * 2 + gap * 3 + trace_h + 46

    duration = n / result.fps
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" aria-label="Detection timeline for '
        f'{html.escape(result.trip_id)}" font-family="IBM Plex Sans, system-ui, sans-serif">'
    ]

    recall = "-" if result.recall is None else f"{result.recall:.0%}"
    far = "-" if result.false_alarm_rate is None else f"{result.false_alarm_rate:.0%}"
    parts.append(
        f'<text x="0" y="18" font-size="14" font-weight="600" fill="{_INK}">'
        f'{html.escape(result.trip_id)}</text>'
        f'<text x="{width}" y="18" font-size="11.5" text-anchor="end" fill="{_SUB}" '
        f'font-family="IBM Plex Mono, monospace">'
        f'recall {recall} &#183; false alarms {far} &#183; {duration:.0f}s</text>'
    )

    rows = [
        ("Ground truth", _segments(result.truth)),
        ("Guardian", _segments(result.predicted)),
    ]
    y = pad_t
    for label, segments in rows:
        parts.append(
            f'<text x="{pad_l - 12}" y="{y + strip_h * 0.68:.1f}" font-size="11.5" '
            f'text-anchor="end" fill="{_SUB}">{label}</text>'
        )
        parts += _strip(segments, n, pad_l, y, plot_w, strip_h)
        y += strip_h + gap

    # Disagreement ticks: where the two strips differ, marked once per run.
    truth = np.asarray(result.truth, dtype=object)
    pred = np.asarray(result.predicted, dtype=object)
    wrong = truth != pred
    if wrong.any():
        for state, start, length in _segments(["x" if w else "." for w in wrong]):
            if state != "x":
                continue
            sx = pad_l + plot_w * (start / n)
            sw = max(plot_w * (length / n), 0.6)
            parts.append(
                f'<rect x="{sx:.2f}" y="{pad_t - 5:.1f}" width="{sw:.2f}" height="3" '
                f'fill="{_INK}" opacity="0.55"><title>disagreement</title></rect>'
            )

    if eye_closure is not None:
        values = np.asarray(eye_closure, dtype=float)[:n]
        if len(values) < n:  # pad short feature files rather than crashing
            values = np.concatenate([values, np.full(n - len(values), np.nan)])
        top = y + 6
        lo, hi = 0.0, 1.0
        def py(v: float) -> float:
            return top + trace_h * (1 - (v - lo) / (hi - lo))

        parts.append(
            f'<text x="{pad_l - 12}" y="{top + 12:.1f}" font-size="11.5" '
            f'text-anchor="end" fill="{_SUB}">Eye closure</text>'
        )
        thr_y = py(perclos_threshold)
        parts.append(
            f'<line x1="{pad_l}" y1="{thr_y:.1f}" x2="{pad_l + plot_w}" y2="{thr_y:.1f}" '
            f'stroke="{_INK}" stroke-width="1" stroke-dasharray="4 3" opacity="0.45"/>'
            f'<text x="{pad_l + plot_w + 4}" y="{thr_y + 3.5:.1f}" font-size="9.5" '
            f'fill="{_SUB}" font-family="IBM Plex Mono, monospace">'
            f'{perclos_threshold:.1f} closed</text>'
        )
        pts = []
        for i, v in enumerate(values):
            if not np.isfinite(v):
                continue
            pts.append(f"{pad_l + plot_w * (i / n):.2f},{py(float(v)):.2f}")
        if pts:
            parts.append(
                f'<polyline points="{" ".join(pts)}" fill="none" '
                f'stroke="{_INK}" stroke-width="1.1" opacity="0.75"/>'
            )
        y = top + trace_h

    # Time axis
    axis_y = y + 20
    parts.append(
        f'<line x1="{pad_l}" y1="{axis_y - 8:.1f}" x2="{pad_l + plot_w}" '
        f'y2="{axis_y - 8:.1f}" stroke="{_LINE}" stroke-width="1"/>'
    )
    step = 5 if duration <= 40 else 10
    tick = 0.0
    while tick <= duration + 1e-6:
        tx = pad_l + plot_w * (tick / duration)
        parts.append(
            f'<text x="{tx:.2f}" y="{axis_y + 5:.1f}" font-size="10" '
            f'text-anchor="middle" fill="{_SUB}" '
            f'font-family="IBM Plex Mono, monospace">{tick:.0f}s</text>'
        )
        tick += step

    parts.append("</svg>")
    return "".join(parts)


def render_legend() -> str:
    """A shared legend, rendered once rather than on every chart."""
    items, x = [], 0
    # Clear states first, then the impaired ones, so the legend reads as the
    # severity ramp the colours already are.
    ordered = [s for s in CLASS_NAMES if s not in IMPAIRED_STATES] + [
        s for s in CLASS_NAMES if s in IMPAIRED_STATES
    ]
    for state in ordered:
        mark = " (impaired)" if state in IMPAIRED_STATES else ""
        items.append(
            f'<rect x="{x}" y="4" width="11" height="11" rx="2" fill="{_colour(state)}"/>'
            f'<text x="{x + 16}" y="14" font-size="11.5" fill="{_SUB}">'
            f'{state}{mark}</text>'
        )
        x += 30 + (len(state) + len(mark)) * 6.6
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {int(x)} 20" '
        f'width="100%" font-family="IBM Plex Sans, system-ui, sans-serif" '
        f'role="img" aria-label="State colour legend">{"".join(items)}</svg>'
    )
