"""
evaluate.py
===========

DO GUARDIAN'S THRESHOLDS SURVIVE CONTACT WITH STRANGERS?

The shipped driver-state engine is four physiological thresholds tuned on six
people. They beat every learned model we tried (84.5 vs 18.5 for a CNN), but
"beats a model that memorised six faces" is a low bar, and the honest reading of
docs/WRITEUP.md is that the thresholds were never tested on anyone else.

This module tests them on public datasets containing many more faces.

WHAT CAN AND CANNOT BE CHECKED HERE
-----------------------------------
Guardian's rules are WINDOW statistics: PERCLOS over 10 s, blink rate over 10 s,
jaw mean over 3 s. Public drowsiness datasets are loose collections of stills,
so the windows cannot be reconstructed and no honest reading of this code
produces a "score" for the rule engine.

What *is* directly comparable is the quantity PERCLOS is built from. PERCLOS is
by definition the fraction of frames in a window whose eyes are closed. So the
class-wise closed-frame RATE on an independent dataset is the same quantity the
0.08 (drowsy) and 0.45 (microsleep) thresholds are expressed in, just pooled
over people instead of over a 10-second window. If drowsy sessions do not sit
above 0.08 across hundreds of unseen faces, the threshold does not generalise.

So this reports three things and claims nothing more:
  1. the per-class distribution of eye closure and jaw opening,
  2. the closed-frame rate per class, against the shipped PERCLOS thresholds,
  3. how well the raw signal separates the classes at all (ROC AUC).
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Iterable, Optional, Sequence

from . import sources

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = _PROJECT_ROOT / "docs" / "opendata"


@dataclass(frozen=True)
class Thresholds:
    """
    The shipped values, copied from `guardian/challenge2/rules.py`.

    Duplicated deliberately: this module must run without the private dataset
    (which `rules.py` imports its way toward), and a test asserts the two stay
    in step whenever `rules.py` is importable, so the copy cannot rot silently.
    """

    perclos_drowsy: float = 0.08
    perclos_microsleep: float = 0.45
    eye_blink_mean_drowsy: float = 0.12
    jaw_yawn: float = 0.20
    #: Per-frame eye-closure cut that defines "closed" inside PERCLOS.
    eye_closed_cut: float = 0.50


@dataclass(frozen=True)
class FrameRecord:
    """One evaluated image."""

    label: str
    eye_blink: float
    jaw_open: float
    face_found: bool


def roc_auc(positive: Sequence[float], negative: Sequence[float]) -> Optional[float]:
    """
    Rank-based ROC AUC, ties handled, no third-party dependency.

    Reads as: the probability that a randomly chosen drowsy frame scores higher
    than a randomly chosen alert one. 0.5 is a coin flip.
    """
    if not positive or not negative:
        return None
    merged = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    ranks: list[float] = [0.0] * len(merged)
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[k] = average_rank
        i = j + 1
    positive_rank_sum = sum(r for r, (_, is_pos) in zip(ranks, merged) if is_pos)
    n_pos, n_neg = len(positive), len(negative)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _describe(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": round(median(ordered), 4),
        "mean": round(sum(ordered) / len(ordered), 4),
        "p10": round(ordered[int(0.10 * (len(ordered) - 1))], 4),
        "p90": round(ordered[int(0.90 * (len(ordered) - 1))], 4),
    }


def summarise(records: Iterable[FrameRecord], thresholds: Thresholds | None = None) -> dict:
    """
    Turn evaluated frames into the report body. Pure: no I/O, no MediaPipe.

    Frames where no face was found are counted and then excluded. They are a
    detection failure, not an observation about eyelids, and averaging them in
    as zeros would drag every mean toward "eyes open".
    """
    thresholds = thresholds or Thresholds()
    records = list(records)
    missing = [r for r in records if not r.face_found]
    usable = [r for r in records if r.face_found]

    by_class: dict[str, list[FrameRecord]] = {}
    for record in usable:
        by_class.setdefault(record.label, []).append(record)

    per_class = {}
    for label, rows in sorted(by_class.items()):
        eye = [r.eye_blink for r in rows]
        jaw = [r.jaw_open for r in rows]
        closed = sum(1 for v in eye if v >= thresholds.eye_closed_cut)
        per_class[label] = {
            "frames": len(rows),
            "eye_blink": _describe(eye),
            "jaw_open": _describe(jaw),
            # The PERCLOS-comparable quantity: fraction of frames with eyes closed.
            "closed_frame_rate": round(closed / len(rows), 4) if rows else None,
            "yawn_frame_rate": round(
                sum(1 for v in jaw if v > thresholds.jaw_yawn) / len(rows), 4
            ) if rows else None,
        }

    drowsy = [r.eye_blink for r in by_class.get("drowsy", [])]
    alert = [r.eye_blink for r in by_class.get("alert", [])]
    auc_eye = roc_auc(drowsy, alert)
    auc_jaw = roc_auc(
        [r.jaw_open for r in by_class.get("drowsy", [])],
        [r.jaw_open for r in by_class.get("alert", [])],
    )

    verdict = None
    drowsy_rate = per_class.get("drowsy", {}).get("closed_frame_rate")
    alert_rate = per_class.get("alert", {}).get("closed_frame_rate")
    if drowsy_rate is not None and alert_rate is not None:
        # A threshold earns its keep only if it lies BETWEEN the classes.
        # "Drowsy clears it" alone is half the story: if alert clears it too, the
        # rule fires on wide-awake drivers, which is a false-alarm generator, not
        # a detector. Both halves are reported, and the conjunction is the verdict.
        drowsy_clears = drowsy_rate > thresholds.perclos_drowsy
        alert_stays_under = alert_rate <= thresholds.perclos_drowsy
        verdict = {
            "drowsy_closed_rate_above_perclos_threshold": drowsy_clears,
            "alert_closed_rate_below_perclos_threshold": alert_stays_under,
            "threshold_discriminates": drowsy_clears and alert_stays_under,
            "drowsy_closed_rate_above_alert": drowsy_rate > alert_rate,
            "margin_vs_threshold": round(drowsy_rate - thresholds.perclos_drowsy, 4),
            "alert_margin_vs_threshold": round(alert_rate - thresholds.perclos_drowsy, 4),
            "margin_vs_alert": round(drowsy_rate - alert_rate, 4),
        }

    return {
        "frames_evaluated": len(records),
        "faces_detected": len(usable),
        "face_detection_rate": round(len(usable) / len(records), 4) if records else None,
        "frames_without_face": len(missing),
        "thresholds": asdict(thresholds),
        "per_class": per_class,
        "separation": {
            "eye_blink_auc_drowsy_vs_alert": round(auc_eye, 4) if auc_eye is not None else None,
            "jaw_open_auc_drowsy_vs_alert": round(auc_jaw, 4) if auc_jaw is not None else None,
        },
        "perclos_check": verdict,
    }


@dataclass
class Report:
    dataset: dict
    summary: dict
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        d, s = self.dataset, self.summary
        lines = [
            "# Guardian driver-state signals on an open dataset",
            "",
            f"_Generated {self.generated_at}._",
            "",
            "## Dataset",
            "",
            f"- **Source**: `{d['repo_id']}` (split `{d['split']}`)",
            f"- **Licence**: {d['licence']}",
            f"- **Provenance**: {d['provenance']}",
            f"- **Caveat**: {d['notes']}",
            "",
            "## What this does and does not show",
            "",
            "Guardian's rules are window statistics (PERCLOS over 10 s, jaw mean over",
            "3 s). These are loose stills, so the windows cannot be rebuilt and this is",
            "**not** a score for the rule engine. What is comparable is PERCLOS's own",
            "numerator: the fraction of frames whose eyes are closed.",
            "",
            "## Results",
            "",
            f"- Frames evaluated: **{s['frames_evaluated']}** "
            f"(face detected in {s['faces_detected']}, "
            f"{s['face_detection_rate']} detection rate)",
            "",
            "| class | frames | eye_blink median | closed-frame rate | jaw_open median |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, stats in s["per_class"].items():
            lines.append(
                f"| {label} | {stats['frames']} | {stats['eye_blink'].get('median')} "
                f"| {stats['closed_frame_rate']} | {stats['jaw_open'].get('median')} |"
            )
        sep = s["separation"]
        lines += [
            "",
            f"- Eye-closure ROC AUC (drowsy vs alert): **{sep['eye_blink_auc_drowsy_vs_alert']}**",
            f"- Jaw-opening ROC AUC (drowsy vs alert): **{sep['jaw_open_auc_drowsy_vs_alert']}**",
            "",
        ]
        check = s.get("perclos_check")
        if check:
            cut = s["thresholds"]["perclos_drowsy"]
            lines += ["## Verdict", ""]
            if check["threshold_discriminates"]:
                lines += [
                    f"The shipped PERCLOS threshold of **{cut}** lands **between the "
                    f"two classes** on this dataset: drowsy closes "
                    f"{check['margin_vs_threshold']:+} above it while alert stays "
                    f"{check['alert_margin_vs_threshold']:+} relative to it. On these "
                    "faces - none of them among the six it was tuned on - the "
                    "threshold transfers.",
                ]
            elif check["drowsy_closed_rate_above_perclos_threshold"]:
                lines += [
                    f"**The threshold does not discriminate here.** Drowsy clears "
                    f"{cut} ({check['margin_vs_threshold']:+}), but so does alert "
                    f"({check['alert_margin_vs_threshold']:+}) - a rule set at this "
                    "value would fire on wide-awake drivers. The classes are only "
                    f"{check['margin_vs_alert']:+} apart, so the signal, not the cut "
                    "point, is what failed on this data.",
                ]
            else:
                lines += [
                    f"**The threshold does not fire.** Even the drowsy class stays "
                    f"below {cut} ({check['margin_vs_threshold']:+}), so the rule "
                    "would miss drowsiness entirely on this data.",
                ]
            lines.append("")
        return "\n".join(lines)


def evaluate(
    source_key: str = sources.DEFAULT_KEY,
    limit: int = 1200,
    seed: int = 42,
    out_dir: Path | None = None,
    progress_every: int = 100,
) -> Report:
    """Stream a public dataset, extract the signals, and write the report."""
    from datasets import load_dataset  # local import: heavy, and optional for tests

    from .blendshapes import BlendshapeExtractor, to_rgb

    source = sources.get(source_key)
    print(f"[opendata] {source.repo_id} split={source.split} limit={limit}")
    print(f"[opendata] licence: {source.licence}")

    # Streaming keeps a multi-gigabyte dataset off the disk: we only ever pull
    # the shards we actually read.
    stream = load_dataset(source.repo_id, split=source.split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=2000)

    records: list[FrameRecord] = []
    skipped_unknown_label = 0
    with BlendshapeExtractor() as extractor:
        for row in stream:
            if len(records) >= limit:
                break
            label = source.normalise(row.get(source.label_column))
            if label is None:
                skipped_unknown_label += 1
                continue
            signals = extractor.extract(to_rgb(row[source.image_column]))
            records.append(
                FrameRecord(
                    label=label,
                    eye_blink=signals.eye_blink,
                    jaw_open=signals.jaw_open,
                    face_found=signals.face_found,
                )
            )
            if progress_every and len(records) % progress_every == 0:
                print(f"[opendata]   {len(records)}/{limit} frames")

    if skipped_unknown_label:
        print(f"[opendata] skipped {skipped_unknown_label} rows with unmapped labels")

    report = Report(
        dataset={
            "key": source.key,
            "repo_id": source.repo_id,
            "split": source.split,
            "licence": source.licence,
            "provenance": source.provenance,
            "notes": source.notes,
            "sampled": limit,
            "seed": seed,
        },
        summary=summarise(records),
    )

    out_dir = out_dir or DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{source.key}_report.json").write_text(report.to_json(), encoding="utf-8")
    (out_dir / f"{source.key}_report.md").write_text(report.to_markdown(), encoding="utf-8")
    print(f"[opendata] wrote {out_dir / (source.key + '_report.md')}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dataset", default=sources.DEFAULT_KEY)
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    report = evaluate(args.dataset, args.limit, args.seed, args.out_dir)
    print()
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
