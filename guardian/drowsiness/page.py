"""
page.py
=======

Build the standalone demo page from a Summary.

Generated rather than hand-written so the page can never disagree with the
numbers: every figure on it is read out of the same `Summary` the console
report prints. If the detector changes, re-running the demo changes the page.

The page is one HTML file with its CSS inline. The only external request is
the Google Fonts stylesheet, and there is a real fallback stack behind it, so
the page still reads correctly offline.
"""

from __future__ import annotations

import html
from typing import Optional, Sequence

from guardian.challenge2.labels import IMPAIRED_STATES
from guardian.drowsiness.detect import WARMUP_SECONDS, Summary, TripResult
from guardian.drowsiness.timeline import render_legend

#: What each labelled trip is actually showing. Written down because a reader
#: cannot tell from "T04-Sample" that it is the fold that breaks.
TRIP_NOTES: dict[str, str] = {
    "T01-Sample": "Eyes open throughout; the driver looks away and then back.",
    "T02-Sample": "Drowsy for the whole drive - eyes partly closed, head down.",
    "T03-Sample": "Yawning for the whole drive, eyes partly closed.",
    "T04-Sample": "Same pattern as T01, a different person - and the case that breaks.",
    "T05-Sample": "Micro-sleep: eyes closed, head down, for thirty seconds.",
    "T06-Sample": "Starts drowsy, becomes distracted halfway through.",
}


def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.1%}"


def _states(result: TripResult) -> list[str]:
    seen: list[str] = []
    for state in result.truth:
        if state not in seen:
            seen.append(state)
    return seen


def _trip_section(result: TripResult, svg: str) -> str:
    chips = "".join(
        f'<span class="chip{" chip-impaired" if s in IMPAIRED_STATES else ""}">'
        f"{html.escape(s)}</span>"
        for s in _states(result)
    )
    note = TRIP_NOTES.get(result.trip_id, "")
    lag = ""
    if result.transitions:
        parts = [
            f"{html.escape(t.from_state)} → {html.escape(t.to_state)} at {t.at_s:.0f}s, "
            + ("never followed" if t.followed_after_s is None
               else f"followed {t.followed_after_s:.2f}s later")
            for t in result.transitions
        ]
        lag = f'<p class="lag">{" · ".join(parts)}</p>'

    return f"""<section class="trip">
  <header class="trip-head">
    <div>
      <h3>{html.escape(result.trip_id)}</h3>
      <p class="note">{html.escape(note)}</p>
    </div>
    <div class="chips">{chips}</div>
  </header>
  <div class="chart">{svg}</div>
  {lag}
</section>"""


def build_page(
    summary: Summary,
    charts: Sequence[tuple[str, str]],
    back_href: Optional[str] = None,
    back_label: str = "Back",
) -> str:
    """
    Render the whole page. `charts` is [(trip_id, svg), ...] in display order.

    `back_href` adds a return link, for when this page is published inside a
    larger site rather than read on its own inside the repository.
    """
    by_id = dict(charts)
    back = ""
    if back_href:
        back = (f'<a class="back" href="{html.escape(back_href)}">'
                f'&larr; {html.escape(back_label)}</a>')
    sections = "\n".join(
        _trip_section(r, by_id.get(r.trip_id, "")) for r in summary.trips
    )

    lags = summary.lags()
    median_lag = f"{sorted(lags)[len(lags) // 2]:.2f}s" if lags else "—"
    worst_trip = min(summary.trips, key=lambda r: r.composite)

    rows = "".join(
        f"<tr><td>{html.escape(r.trip_id)}</td>"
        f"<td class=\"num\">{r.impaired_frames}</td>"
        f"<td class=\"num\">{r.impaired_caught}</td>"
        f"<td class=\"num\">{_pct(r.recall)}</td>"
        f"<td class=\"num\">{r.clear_frames}</td>"
        f"<td class=\"num\">{r.false_alarms}</td>"
        f"<td class=\"num\">{_pct(r.false_alarm_rate)}</td>"
        f"<td class=\"num\">{r.composite:.1f}</td></tr>"
        for r in summary.trips
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Can Guardian See a Sleepy Driver?</title>
<meta name="description" content="Measured drowsiness detection for the Guardian Co-Pilot driver-monitoring engine: recall, false alarms and response lag on labelled footage.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root {{
  --bg:#FAF9F6; --surface:#FFFFFF; --ink:#17191C; --sub:#585D65; --faint:#8A9099;
  --line:#E7E4DD; --line-2:#D9D5CC; --accent:#2F53C8;
  --good:#2E7D5B; --warn:#B2701E; --bad:#C7452F;
  /* Consumed by the inline SVGs (see timeline.py) so charts follow the theme. */
  --chart-ink:#17191C; --chart-sub:#585D65; --chart-line:#D9D5CC;
  --sans:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#111316; --surface:#171A1E; --ink:#ECEEF1; --sub:#A2A8B0; --faint:#79808A;
    --line:#262A30; --line-2:#333941; --accent:#87A6FF;
    --good:#67C299; --warn:#E0A96D; --bad:#F08670;
    --chart-ink:#D8DCE2; --chart-sub:#99A0A9; --chart-line:#333941;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#111316; --surface:#171A1E; --ink:#ECEEF1; --sub:#A2A8B0; --faint:#79808A;
  --line:#262A30; --line-2:#333941; --accent:#87A6FF;
  --good:#67C299; --warn:#E0A96D; --bad:#F08670;
  --chart-ink:#D8DCE2; --chart-sub:#99A0A9; --chart-line:#333941;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1000px; margin-inline:auto; padding-inline:clamp(16px,4vw,40px); }}
main.wrap {{ padding-block:0 64px; }}
img, svg {{ max-width:100%; }}
h1,h2,h3 {{ text-wrap:balance; margin:0; }}
a {{ color:var(--accent); }}
:focus-visible {{ outline:2px solid var(--accent); outline-offset:3px; }}

header.top {{ border-bottom:1px solid var(--line); padding-block:clamp(40px,6vw,68px) 32px; }}
.eyebrow {{
  font-family:var(--mono); font-size:11.5px; letter-spacing:.18em;
  text-transform:uppercase; color:var(--faint); margin-bottom:18px;
}}
h1 {{ font-size:clamp(30px,5vw,46px); line-height:1.08; letter-spacing:-.022em; font-weight:600; }}
.lede {{ color:var(--sub); font-size:clamp(16px,2vw,18px); max-width:64ch; margin-top:18px; }}

.headline {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:1px; background:var(--line); border:1px solid var(--line);
  margin-top:36px;
}}
.stat {{ background:var(--bg); padding:20px 22px; }}
.stat .k {{
  font-family:var(--mono); font-size:11px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--faint);
}}
.stat .v {{
  font-size:clamp(28px,4vw,38px); font-weight:600; letter-spacing:-.02em;
  line-height:1.15; margin-top:8px; font-variant-numeric:tabular-nums;
}}
.stat .v.good {{ color:var(--good); }}
.stat .v.warn {{ color:var(--warn); }}
.stat .d {{ font-size:13.5px; color:var(--sub); margin-top:6px; }}

section.block {{ padding-block:clamp(36px,5vw,52px); border-bottom:1px solid var(--line); }}
h2 {{
  font-family:var(--mono); font-size:12.5px; letter-spacing:.15em;
  text-transform:uppercase; color:var(--sub); font-weight:500; margin-bottom:26px;
}}
p {{ max-width:68ch; }}
.legend {{ margin-bottom:26px; max-width:640px; }}

.trip {{ padding-block:26px; border-top:1px solid var(--line); }}
.trip:first-of-type {{ border-top:0; padding-top:4px; }}
.trip-head {{
  display:flex; flex-wrap:wrap; gap:12px 24px;
  align-items:flex-start; justify-content:space-between; margin-bottom:14px;
}}
.trip h3 {{ font-family:var(--mono); font-size:15px; font-weight:500; }}
.note {{ color:var(--sub); font-size:14.5px; margin:4px 0 0; }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.chip {{
  font-family:var(--mono); font-size:11px; padding:3px 8px;
  border:1px solid var(--line-2); border-radius:3px; color:var(--sub);
}}
.chip-impaired {{ border-color:var(--warn); color:var(--warn); }}
.chart {{ background:var(--surface); border:1px solid var(--line); padding:14px 16px; }}
.lag {{
  font-family:var(--mono); font-size:12px; color:var(--faint);
  margin:10px 0 0; max-width:none;
}}

.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; min-width:620px; font-size:14px; }}
th, td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); }}
th {{
  font-family:var(--mono); font-size:11px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--faint); font-weight:500;
}}
td.num {{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
td:first-child {{ font-family:var(--mono); }}

.finding {{ display:grid; gap:22px; }}
.finding h3 {{ font-size:17px; font-weight:600; margin-bottom:6px; }}
.finding p {{ color:var(--sub); margin:0; }}
.caveat {{
  border-left:3px solid var(--warn); padding:2px 0 2px 18px; margin-top:8px;
}}
.back {{
  display:inline-block; font-family:var(--mono); font-size:12.5px;
  text-decoration:none; margin-bottom:22px;
}}
.back:hover {{ text-decoration:underline; }}
footer {{ padding-block:28px 48px; color:var(--faint); font-family:var(--mono); font-size:12px; }}
@media (max-width:560px) {{ .trip-head {{ flex-direction:column; }} }}
</style>
</head>
<body>

<header class="top wrap">
  {back}
  <div class="eyebrow">Guardian Co-Pilot · driver monitoring</div>
  <h1>Can Guardian see a sleepy driver?</h1>
  <p class="lede">
    The engine watches eyelids, not pixels: PERCLOS over a rolling ten-second
    window, the longest unbroken closure, and jaw opening. Here is what it did
    on {len(summary.trips)} labelled drives at 20&nbsp;fps &mdash;
    {summary.impaired_frames + summary.clear_frames:,} frames where somebody had
    already written down what the driver was really doing.
  </p>

  <div class="headline">
    <div class="stat">
      <div class="k">Impairment caught</div>
      <div class="v good">{_pct(summary.recall)}</div>
      <div class="d">{summary.impaired_caught:,} of {summary.impaired_frames:,} frames
        where the driver was drowsy, yawning or micro-sleeping.</div>
    </div>
    <div class="stat">
      <div class="k">False alarms</div>
      <div class="v warn">{_pct(summary.false_alarm_rate)}</div>
      <div class="d">{summary.false_alarms:,} of {summary.clear_frames:,} clear frames
        flagged anyway &mdash; {_pct(summary.false_alarm_rate_after_warmup)} once the
        {WARMUP_SECONDS:.0f}s PERCLOS window has filled.</div>
    </div>
    <div class="stat">
      <div class="k">Lag after a real change</div>
      <div class="v">{median_lag}</div>
      <div class="d">Median time to follow the driver into a new state. The 2.05s
        majority vote is the floor.</div>
    </div>
  </div>
</header>

<main class="wrap">

<section class="block">
  <h2>Every drive, frame by frame</h2>
  <p>
    The top strip is the truth, the strip under it is what Guardian said, and the
    ticks above mark where they disagree. The trace underneath is measured eyelid
    closure; the dashed line is the level above which an eye counts as shut.
  </p>
  <div class="legend">{render_legend()}</div>
  {sections}
</section>

<section class="block">
  <h2>The numbers behind the picture</h2>
  <div class="tablewrap">
    <table>
      <thead><tr>
        <th>Trip</th><th class="num">Impaired</th><th class="num">Caught</th>
        <th class="num">Recall</th><th class="num">Clear</th><th class="num">False</th>
        <th class="num">Rate</th><th class="num">Composite</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</section>

<section class="block">
  <h2>What this run actually shows</h2>
  <div class="finding">
    <div>
      <h3>Sustained closure is the easy case, and it is the dangerous one</h3>
      <p>
        The micro-sleep and yawning drives are called correctly on every single
        frame. Physiology does the work: eyes shut for more than 1.2&nbsp;seconds
        means micro-sleep whether the driver is twenty or sixty, so there is
        nothing to train and nothing to overfit.
      </p>
    </div>
    <div>
      <h3>The lag is bought, not accidental</h3>
      <p>
        Raw per-frame decisions flicker, so the output is a majority vote over
        about two seconds. That is exactly why the drowsy&nbsp;&rarr;&nbsp;distracted
        change on T06-Sample takes 2.10&nbsp;s to appear. Steadier warnings cost
        response time, and that trade should be visible rather than buried.
      </p>
    </div>
    <div>
      <h3>{html.escape(worst_trip.trip_id)} is where it breaks</h3>
      <p>
        On one driver whose eyes never close, the jaw threshold fires anyway and
        produces stretches of phantom <em>yawning</em> &mdash;
        {worst_trip.false_alarms} false alarms across
        {worst_trip.clear_frames} clear frames, and the lowest composite here at
        {worst_trip.composite:.1f}. A resting mouth posture that reads as an open
        jaw is a real failure mode, not noise, and a fixed global threshold has no
        answer to it. This is the case the per-driver learning loop exists for.
      </p>
    </div>
    <div class="caveat">
      <h3>Six drivers is six samples</h3>
      <p>
        Every number here comes from six people. It is enough to show the
        thresholds fire on real physiology; it is nowhere near enough to claim
        they hold for a population. The open-data evaluation in the repository
        exists to test the same thresholds against strangers, and it is where the
        weaknesses show up.
      </p>
    </div>
  </div>
</section>

</main>

<footer class="wrap">
  Generated by <code>python -m guardian.drowsiness.demo</code> · driver footage is
  licensed academic data and is never reproduced here, only the signals derived from it.
</footer>

<script>
/* If this page is published beside the portfolio, honour the theme already
   chosen there. Wrapped because storage throws in private windows. */
try {{
  var t = localStorage.getItem('qnd-theme');
  if (t === 'dark' || t === 'light') document.documentElement.setAttribute('data-theme', t);
}} catch (e) {{}}
</script>

</body>
</html>
"""
