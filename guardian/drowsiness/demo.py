"""
demo.py
=======

Run the drowsiness demo and write its evidence to disk.

    python -m guardian.drowsiness.demo
    python -m guardian.drowsiness.demo --out docs/drowsiness --trips T02-Sample

Produces a console report, one SVG timeline per trip, a machine-readable
report.json, and a self-contained index.html that needs no server and no
network. Everything it writes is derived signal - never driver imagery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from guardian.challenge2.classifier import build_temporal_features
from guardian.challenge2.features import FEATURES_DIR, load_trip_features
from guardian.challenge2.rules import DEFAULT_DATA_ROOT, PRACTICE_TRIP_IDS, RuleEngine
from guardian.drowsiness.detect import (
    DEFAULT_FPS,
    evaluate_trips,
    format_report,
    summarise,
)
from guardian.drowsiness.page import build_page
from guardian.drowsiness.timeline import render_timeline

DEFAULT_OUT = Path("docs/drowsiness")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure and draw Guardian's drowsiness detection."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                        help="Folder holding <trip_id>/<trip_id>.json.gz")
    parser.add_argument("--features-dir", default=str(FEATURES_DIR),
                        help="Folder holding cached per-frame feature CSVs.")
    parser.add_argument("--trips", nargs="*", default=None,
                        help=f"Default: {' '.join(PRACTICE_TRIP_IDS)}")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="Where to write report.json, the SVGs and index.html.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--no-write", action="store_true",
                        help="Print the report without writing any files.")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    features_dir = Path(args.features_dir)
    trips = list(args.trips) if args.trips else list(PRACTICE_TRIP_IDS)

    results = evaluate_trips(
        trips, RuleEngine(), data_root=data_root, features_dir=features_dir, fps=args.fps
    )
    summary = summarise(results)
    print(format_report(summary))

    if args.no_write:
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    charts: list[tuple[str, str]] = []
    for result in results:
        raw = load_trip_features(result.trip_id, features_dir)
        eye = build_temporal_features(raw).get("eye_blink")
        if eye is None:
            eye = raw.get("eye_blink")
        svg = render_timeline(result, eye_closure=eye)
        (out / f"{result.trip_id}.svg").write_text(svg, encoding="utf-8")
        charts.append((result.trip_id, svg))

    (out / "report.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    (out / "index.html").write_text(build_page(summary, charts), encoding="utf-8")

    print(f"\nWrote {len(charts)} timelines, report.json and index.html to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
