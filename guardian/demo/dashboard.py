"""
dashboard.py
============

GUARDIAN CO-PILOT — INTERACTIVE DEMO

A Streamlit app so a judge can *drive* our system instead of watching a video:
pick a trip, scrub the timeline, and see exactly what Guardian saw and why it
decided what it decided.

    streamlit run guardian/demo/dashboard.py

WHY A DASHBOARD AND NOT JUST A VIDEO
------------------------------------
A video proves the output looks right. An interactive dashboard proves the
system is real: the judge chooses the frame, and every panel updates from our
actual cached predictions. It makes the whole pipeline inspectable ("a runnable
dashboard anyone can try") without needing me in the room.

WHAT IS ON SCREEN
-----------------
    Live view    HUD frame (boxes + TTC + driver panel + reason)
    Timelines    TTC, driver state, risk, speed for the whole trip
    Explanation  the plain-language reason behind the current frame's decision
    Fleet view   all trips side by side, the Challenge-3 perspective

Everything reads the cached artifacts produced by `guardian/run_all.py`, so the
dashboard starts instantly and never recomputes a model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make the repo importable when Streamlit runs this file directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from guardian.challenge1.perception import perception_path  # noqa: E402
from guardian.challenge2.features import feature_path  # noqa: E402
from guardian.demo.hud import HudRenderer  # noqa: E402

DATA_ROOT = Path(os.getenv("GUARDIAN_DATA_ROOT", r"C:/guardian_data"))
PRACTICE = [f"T0{i}-Sample" for i in range(1, 7)]
SCORED = [f"T{i:02d}d" for i in range(1, 11)]

STATE_EMOJI = {
    "alert": "🟢", "distracted": "🟠", "yawning": "🟡",
    "drowsy": "🟠", "microsleep": "🔴",
}


# ===========================================================================
# DATA LOADING (cached so scrubbing the slider stays instant)
# ===========================================================================


@st.cache_resource(show_spinner="Loading trip…")
def get_renderer(trip_id: str) -> HudRenderer:
    """One HudRenderer per trip, reused across reruns."""
    return HudRenderer(trip_id, data_root=DATA_ROOT)


@st.cache_data(show_spinner=False)
def available_trips() -> list[str]:
    """Trips that have BOTH caches, so the dashboard can never half-load."""
    return [
        t for t in PRACTICE + SCORED
        if perception_path(t).exists() and feature_path(t).exists()
    ]


@st.cache_data(show_spinner=False)
def trip_timeline(trip_id: str) -> pd.DataFrame:
    """Per-frame table driving all the charts."""
    r = get_renderer(trip_id)
    return pd.DataFrame({
        "frame": np.arange(r.n_frames),
        "time_s": r.timestamps,
        "speed_kmh": r.speeds,
        # inf is unplottable; 30 s reads as "nothing ahead" on the chart.
        "ttc_s": np.where(np.isfinite(r.ttc), r.ttc, 30.0),
        "hazard": np.isfinite(r.ttc),
        "driver_state": r.states,
        "risk": r.risk,
    })


# ===========================================================================
# PAGE
# ===========================================================================


def main() -> None:
    st.set_page_config(page_title="Guardian Co-Pilot", page_icon="🛡️",
                       layout="wide")

    st.title("🛡️ Guardian Co-Pilot")
    st.caption("Observe → Understand → Predict → Protect · "
               "Guardian Co-Pilot")

    trips = available_trips()
    if not trips:
        st.error(
            "No cached trips found. Generate them first:\n\n"
            "```\npython guardian/run_all.py\n```"
        )
        return

    # ---- sidebar -----------------------------------------------------------
    with st.sidebar:
        st.header("Controls")
        trip_id = st.selectbox("Trip", trips, index=0)
        r = get_renderer(trip_id)
        frame = st.slider("Frame", 0, r.n_frames - 1, min(240, r.n_frames - 1))

        st.divider()
        st.subheader("Jump to a hazard")
        hazards = np.where(np.isfinite(r.ttc) & (r.ttc < 3.0))[0]
        if len(hazards):
            st.caption(f"{len(hazards)} frames with TTC < 3 s")
            if st.button("⚠️ Most critical frame", use_container_width=True):
                frame = int(hazards[np.argmin(r.ttc[hazards])])
                st.session_state["_jump"] = frame
        else:
            st.caption("No critical frames in this trip.")

        st.divider()
        st.caption(
            "Scores (benchmark's evaluator, practice trips)\n\n"
            "- Challenge 1 — TTC: **59.1** (baseline 19.7)\n"
            "- Challenge 2 — driver state: **84.5** LOTO\n"
            "- Challenge 3 — derived from our TTC"
        )

    frame = st.session_state.pop("_jump", frame)
    timeline = trip_timeline(trip_id)
    row = timeline.iloc[frame]

    # ---- headline metrics --------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    ttc_value = r.ttc[frame]
    c1.metric("Time to collision",
              "clear" if not np.isfinite(ttc_value) else f"{ttc_value:.1f} s",
              delta="hazard" if np.isfinite(ttc_value) and ttc_value < 3 else None,
              delta_color="inverse")
    state = str(row["driver_state"])
    c2.metric("Driver state", f"{STATE_EMOJI.get(state, '')} {state}")
    c3.metric("Risk score", f"{row['risk']:.0f} / 100")
    c4.metric("Speed", f"{row['speed_kmh']:.1f} km/h")

    # ---- HUD + explanation -------------------------------------------------
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Live view")
        hud = r.render(frame)
        # OpenCV is BGR; Streamlit expects RGB.
        st.image(hud[:, :, ::-1], use_container_width=True)

    with right:
        st.subheader("Why Guardian decided this")
        st.info(r.engine.explain(r.temporal, frame), icon="🧠")

        st.markdown("**Challenge 2 — the signals behind it**")
        t = r.temporal.iloc[frame]
        st.dataframe(
            pd.DataFrame({
                "signal": ["eye closure (10s mean)", "PERCLOS (10s)",
                           "longest closure", "mouth open (3s)", "phone (3s)"],
                "value": [f"{t['eb_mean_w201']:.3f}", f"{t['perclos_w201']:.0%}",
                          f"{t['closed_run_sec']:.1f} s",
                          f"{t['jaw_mean_w61']:.3f}", f"{t['phone_rate_w61']:.0%}"],
                "threshold": ["0.120", "8%", "1.2 s", "0.200", "15%"],
            }),
            hide_index=True, use_container_width=True,
        )

        detections = r._by_frame.get(frame)
        st.markdown("**Challenge 1 — obstacles in the collision cone**")
        if detections is None or detections.empty:
            st.caption("No obstacle detected in this frame.")
        else:
            st.dataframe(
                detections[["target_class", "distance_m", "lateral_m",
                            "confidence"]]
                .round(2).rename(columns={
                    "target_class": "class", "distance_m": "distance (m)",
                    "lateral_m": "lateral (m)", "confidence": "conf"}),
                hide_index=True, use_container_width=True,
            )

    # ---- timelines ---------------------------------------------------------
    st.subheader("Trip timeline")
    tab1, tab2, tab3 = st.tabs(["Collision risk", "Driver state", "Kinematics"])

    with tab1:
        st.line_chart(timeline.set_index("time_s")[["ttc_s"]],
                      height=220, color="#ff4b4b")
        st.caption("30 s means no obstacle in the collision cone. "
                   "Below 1.5 s counts as a near-miss for Challenge 3.")
        st.line_chart(timeline.set_index("time_s")[["risk"]], height=180)

    with tab2:
        counts = timeline["driver_state"].value_counts()
        cols = st.columns(len(counts))
        for col, (name, n) in zip(cols, counts.items()):
            col.metric(f"{STATE_EMOJI.get(name, '')} {name}",
                       f"{100 * n / len(timeline):.0f}%")
        # Numeric encoding so the state sequence is visible as a step chart.
        order = ["alert", "distracted", "yawning", "drowsy", "microsleep"]
        encoded = timeline["driver_state"].map(
            {s: i for i, s in enumerate(order)}
        )
        st.line_chart(
            pd.DataFrame({"time_s": timeline["time_s"],
                          "state (0=alert … 4=microsleep)": encoded}
                         ).set_index("time_s"),
            height=220,
        )

    with tab3:
        st.line_chart(timeline.set_index("time_s")[["speed_kmh"]], height=220)

    # ---- fleet view --------------------------------------------------------
    st.subheader("Fleet view — Challenge 3 perspective")
    st.caption("Every cached trip, compared. This is the view a fleet operator "
               "would use.")

    rows = []
    for t in trips:
        tl = trip_timeline(t)
        rows.append({
            "trip": t,
            "frames": len(tl),
            "near-miss frames (TTC<1.5s)": int((tl["ttc_s"] < 1.5).sum()),
            "hazard frames (TTC<3s)": int((tl["ttc_s"] < 3.0).sum()),
            "mean risk": round(float(tl["risk"].mean()), 1),
            "dominant driver state": tl["driver_state"].mode().iloc[0],
            "max speed": round(float(tl["speed_kmh"].max()), 1),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
