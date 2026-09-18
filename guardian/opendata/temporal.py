"""
temporal.py
===========

THE EXPERIMENT THIS PROJECT HAS BEEN OWING SINCE THE CNN FAILED.

The finding that shaped Guardian was negative: on six drivers, every model we
fitted scored worse across people than a handful of physiological thresholds
(18.5 for the CNN, 23.8 for gradient boosting, 43.7 after invariance work, 84.5
for the rules - see docs/WRITEUP.md). The obvious objection is equally strong:
with six drivers, *of course* a fitted model overfits. A threshold wins by
default when there is nothing to fit on. That objection cannot be answered with
six drivers, and so it was never answered.

This module answers it with sixty, on public data, with the split done by
someone else before we arrived.

WHAT IS COMPARED
----------------
Three predictors, all scored the same way: fit on four folds, scored on the
fifth, repeated five times, never once looking at the held-out drivers.

1.  **Rule** - four thresholds on two blink statistics, in the same escalating
    shape as `challenge2/rules.py`. The cut points are chosen by coordinate
    ascent on the training folds only.
2.  **Fitted model, same information** - gradient boosting on the *identical*
    five statistics the rule sees. This is the fair fight. Any gap is about
    fitting versus thresholding, not about who got better features.
3.  **Fitted model, more information** - logistic regression on the raw
    30x4 window, flattened. It sees everything the rule sees and the ordering
    besides.

A majority-class predictor fixes the floor.

Each one is also scored **in-sample**, on the very folds it was fitted on. The
gap between those two columns is the whole subject of this module. A rule with
four cut points cannot memorise 48 people; a boosted ensemble can try.

WHAT A RESULT HERE DOES NOT LICENCE
-----------------------------------
This data is dlib eye-aspect-ratio blinks. Guardian's cabin pipeline is
MediaPipe blendshapes. The numeric thresholds found here are therefore NOT
transferable to `challenge2/rules.py` and are not copied into it. What
transfers, if anything does, is the shape of the claim: that a statistic
referred to a driver's own baseline, cut at a fixed point, survives contact
with strangers better than a function fitted to other drivers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import rldd
from .rldd import LABEL_NAMES, Fold, video_segments

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "opendata"

#: The window statistics the rule is allowed to see. Deliberately few, and
#: deliberately the ones with a physiological reading.
STAT_NAMES: tuple[str, ...] = (
    "duration_mean",  # how much longer the eyes stay shut than this driver's baseline
    "duration_p90",   # the worst blink in the window - the microsleep analogue
    "frequency_mean",
    "amplitude_mean",
    "velocity_mean",  # drowsy lids move slower, so this is expected to go negative
)

ALERT, LOW, DROWSY = 0, 5, 10


# ---------------------------------------------------------------------------
# From a window of blinks to the handful of numbers a rule may look at.
# ---------------------------------------------------------------------------

def reduce_windows(windows: np.ndarray) -> np.ndarray:
    """(n, 30, 4) blinks -> (n, 5) statistics, ignoring padding.

    Padding rows are all-zero, and zero is this driver's alert mean after
    standardisation. Averaging them in would pull every statistic toward
    "fresh" - a bias that flatters alert windows and blunts drowsy ones. They
    are masked out; a window that is entirely padding (which does not occur in
    the shipped release, but would be silent if it did) reduces to zeros and is
    reported by `padding_report`.
    """
    windows = np.asarray(windows, dtype=float)
    mask = rldd.padding_mask(windows)  # (n, 30)
    counts = mask.sum(axis=1)
    safe = np.maximum(counts, 1)[:, None]

    f_idx = {name: i for i, name in enumerate(rldd.BLINK_FEATURES)}
    out = np.zeros((len(windows), len(STAT_NAMES)), dtype=float)

    def masked_mean(feature: str) -> np.ndarray:
        values = windows[:, :, f_idx[feature]]
        return (np.where(mask, values, 0.0).sum(axis=1)[:, None] / safe).ravel()

    duration = windows[:, :, f_idx["duration"]]
    # np.percentile has no mask, and dropping to a Python loop over ~8,000 rows
    # is cheap and obviously correct, which beats a clever vectorisation that
    # silently includes padding.
    p90 = np.array(
        [
            np.percentile(row[m], 90) if m.any() else 0.0
            for row, m in zip(duration, mask)
        ]
    )

    out[:, STAT_NAMES.index("duration_mean")] = masked_mean("duration")
    out[:, STAT_NAMES.index("duration_p90")] = p90
    out[:, STAT_NAMES.index("frequency_mean")] = masked_mean("frequency")
    out[:, STAT_NAMES.index("amplitude_mean")] = masked_mean("amplitude")
    out[:, STAT_NAMES.index("velocity_mean")] = masked_mean("velocity")
    return out


def padding_report(windows: np.ndarray) -> dict:
    """How much of what we just averaged was actually padding."""
    mask = rldd.padding_mask(np.asarray(windows, dtype=float))
    total = mask.size
    pad = int((~mask).sum())
    return {
        "blink_slots": int(total),
        "padded_slots": pad,
        "padded_fraction": round(pad / total, 6) if total else 0.0,
        "windows_entirely_padding": int((~mask).all(axis=1).sum()),
    }


# ---------------------------------------------------------------------------
# The rule.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlinkRule:
    """Four cut points, in the same escalating shape as the cabin rule engine.

    Read it aloud and it is a sentence about eyelids: if this driver's blinks
    have become much longer than their own baseline, or any single blink in the
    window was very long, they are drowsy; if their blinks are somewhat longer,
    or the lid has slowed down, they are losing vigilance; otherwise they are
    alert.
    """

    duration_mean_drowsy: float = 1.2
    duration_p90_micro: float = 4.0
    duration_mean_low: float = 0.4
    velocity_mean_low: float = -0.4

    def apply(self, stats: np.ndarray) -> np.ndarray:
        d_mean = stats[:, STAT_NAMES.index("duration_mean")]
        d_p90 = stats[:, STAT_NAMES.index("duration_p90")]
        v_mean = stats[:, STAT_NAMES.index("velocity_mean")]

        labels = np.full(len(stats), ALERT, dtype=int)
        low = (d_mean >= self.duration_mean_low) | (v_mean <= self.velocity_mean_low)
        labels[low] = LOW
        drowsy = (d_mean >= self.duration_mean_drowsy) | (d_p90 >= self.duration_p90_micro)
        labels[drowsy] = DROWSY
        return labels


#: Searched by coordinate ascent. Ranges are set from the standardised scale
#: (a value of 1.0 is one within-driver standard deviation), not from peeking
#: at held-out performance.
DEFAULT_GRID: dict[str, tuple[float, ...]] = {
    "duration_mean_drowsy": (0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0),
    "duration_p90_micro": (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0),
    "duration_mean_low": (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "velocity_mean_low": (-1.2, -0.8, -0.6, -0.4, -0.2, 0.0),
}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the three scores.

    Macro, not accuracy: `low_vigilant` and `drowsy` together are 80% of the
    rows, so a predictor that never says `alert` still scores well on accuracy.
    Macro-F1 makes it pay for the class it abandons.
    """
    scores = []
    for cls in (ALERT, LOW, DROWSY):
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        denom = 2 * tp + fp + fn
        if denom == 0:
            # This class is neither in the truth nor in the predictions. Scoring
            # it zero would punish a predictor for a class that was never on the
            # table - on a two-class subset a perfect answer would cap at 0.67 -
            # so it is left out of the average, the way sklearn's macro average
            # leaves out labels absent from both sides. Every class is present
            # in the real evaluation, so this changes no number in the report;
            # it only stops the metric lying on subsets.
            continue
        scores.append(2 * tp / denom)
    return float(np.mean(scores)) if scores else 0.0


def search_rule(
    stats: np.ndarray,
    labels: np.ndarray,
    grid: Optional[dict[str, tuple[float, ...]]] = None,
    start: Optional[BlinkRule] = None,
    max_sweeps: int = 6,
) -> tuple[BlinkRule, float]:
    """Coordinate ascent on macro-F1. Training rows only - never the held-out fold.

    Strict improvement (`>`), so a tie leaves the incumbent in place and the
    defaults survive unless the data actually argues against them. Same
    discipline as `challenge2.rules._select_thresholds`, for the same reason:
    a search that accepts ties wanders, and a wandering search reports the last
    place it happened to stop as if it were a finding.
    """
    grid = grid or DEFAULT_GRID
    current = start or BlinkRule()
    best = macro_f1(labels, current.apply(stats))
    for _ in range(max_sweeps):
        improved = False
        for name, candidates in grid.items():
            for value in candidates:
                if value == getattr(current, name):
                    continue
                trial = BlinkRule(**{**asdict(current), name: value})
                score = macro_f1(labels, trial.apply(stats))
                if score > best:
                    best, current, improved = score, trial, True
        if not improved:
            break
    return current, best


# ---------------------------------------------------------------------------
# The comparators.
# ---------------------------------------------------------------------------

def _fit_predictors(
    train_stats: np.ndarray,
    train_raw: np.ndarray,
    train_y: np.ndarray,
    seed: int,
) -> dict:
    """Fit everything that needs fitting, on training rows only."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    boosted = HistGradientBoostingClassifier(random_state=seed).fit(train_stats, train_y)

    flat = train_raw.reshape(len(train_raw), -1)
    scaler = StandardScaler().fit(flat)
    logistic = LogisticRegression(max_iter=2000, random_state=seed).fit(
        scaler.transform(flat), train_y
    )

    values, counts = np.unique(train_y, return_counts=True)
    majority = int(values[int(np.argmax(counts))])

    return {"boosted": boosted, "logistic": logistic, "scaler": scaler, "majority": majority}


def _predict_all(models: dict, rule: BlinkRule, stats: np.ndarray, raw: np.ndarray) -> dict:
    flat = raw.reshape(len(raw), -1)
    return {
        "rule": rule.apply(stats),
        "boosted_on_stats": models["boosted"].predict(stats),
        "logistic_on_window": models["logistic"].predict(models["scaler"].transform(flat)),
        "majority": np.full(len(stats), models["majority"], dtype=int),
    }


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    per_class = {}
    for value, name in LABEL_NAMES.items():
        support = int((y_true == value).sum())
        hit = int(((y_true == value) & (y_pred == value)).sum())
        per_class[name] = {
            "support": support,
            "recall": round(hit / support, 4) if support else None,
        }
    return {
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "per_class": per_class,
    }


def roc_auc(positive: Sequence[float], negative: Sequence[float]) -> Optional[float]:
    """Rank-based AUC with tie correction; None when either side is empty."""
    pos, neg = np.asarray(positive, dtype=float), np.asarray(negative, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return None
    combined = np.concatenate([pos, neg])
    order = combined.argsort()
    ranks = np.empty(len(combined), dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1)
    # Average the ranks inside each tie group, or ties inflate the AUC.
    unique, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    rank_sum = ranks[: len(pos)].sum()
    return float((rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def feature_separation(stats: np.ndarray, labels: np.ndarray) -> dict:
    """Which statistic actually carries drowsiness, measured not assumed."""
    out = {}
    for i, name in enumerate(STAT_NAMES):
        drowsy = stats[labels == DROWSY, i]
        alert = stats[labels == ALERT, i]
        auc = roc_auc(drowsy, alert)
        # An AUC of 0.29 is not a weak signal, it is a strong signal pointing
        # the other way: drowsy lids move SLOWER, so velocity separates the
        # classes about as well as duration does, in the opposite direction.
        # Reporting the raw AUC alone invites a reader to dismiss it, so the
        # direction and the distance from chance are reported beside it.
        out[name] = {
            "auc_drowsy_vs_alert": round(auc, 4) if auc is not None else None,
            "direction": (
                None if auc is None
                else "higher when drowsy" if auc >= 0.5 else "lower when drowsy"
            ),
            "separation": round(max(auc, 1 - auc), 4) if auc is not None else None,
            "alert_median": round(float(np.median(alert)), 4) if alert.size else None,
            "drowsy_median": round(float(np.median(drowsy)), 4) if drowsy.size else None,
        }
    return out


def vote_by_video(predictions: np.ndarray, starts: Sequence[int]) -> np.ndarray:
    """One verdict per driving session, by majority vote of its windows.

    This is the metric that matches what Guardian actually emits. The cabin
    engine does not publish a fresh opinion every few minutes and leave the
    driver to average them; it smooths window verdicts into a trip state
    (`challenge2.rules.majority_smooth`). Scoring per window and calling it the
    product understates a system that was designed to commit.

    Ties break toward the MORE severe state. That is a safety decision, not a
    statistical one: when a session is genuinely split between alert and
    drowsy, a co-pilot that shrugs is worse than one that speaks up, and the
    asymmetry is stated here rather than buried in an argmax.
    """
    predictions = np.asarray(predictions)
    out = np.empty(len(starts), dtype=int)
    for i, (a, b) in enumerate(video_segments(starts, len(predictions))):
        window = predictions[a:b]
        best_count, best_label = -1, ALERT
        for cls in (ALERT, LOW, DROWSY):  # ascending severity, >= keeps the later
            count = int((window == cls).sum())
            if count >= best_count:
                best_count, best_label = count, cls
        out[i] = best_label
    return out


def video_labels(labels: np.ndarray, starts: Sequence[int]) -> np.ndarray:
    """The true label of each session. Constant within a video by construction."""
    labels = np.asarray(labels)
    return np.array([labels[a] for a, _ in video_segments(starts, len(labels))])


def alarm_view(truth: np.ndarray, pred: np.ndarray) -> dict:
    """The two numbers a driver would actually feel.

    Guardian's cabin engine speaks two states, not three: it either raises a
    drowsiness alarm or it does not. Collapsed to that vocabulary - `drowsy`
    means alarm, anything else means silence - and restricted to the sessions
    whose ground truth is unambiguous (`alert` or `drowsy`, dropping the
    middle), a predictor has exactly two ways to be wrong, and they cost
    completely different things. Missing a drowsy driver is a safety failure.
    Alarming at an alert one is how a driver learns to ignore the system, which
    is a safety failure one drive later.
    """
    keep = (truth == ALERT) | (truth == DROWSY)
    truth, pred = truth[keep], pred[keep]
    alarm = pred == DROWSY
    drowsy = truth == DROWSY
    caught = int((alarm & drowsy).sum())
    missed = int((~alarm & drowsy).sum())
    false_alarm = int((alarm & ~drowsy).sum())
    quiet = int((~alarm & ~drowsy).sum())
    return {
        "sessions": int(keep.sum()),
        "drowsy_caught": caught,
        "drowsy_missed": missed,
        "recall_on_drowsy": round(caught / (caught + missed), 4) if caught + missed else None,
        "false_alarms_on_alert": false_alarm,
        "false_alarm_rate": round(false_alarm / (false_alarm + quiet), 4)
        if false_alarm + quiet
        else None,
    }


# ---------------------------------------------------------------------------
# The cross-validation.
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    n_test_videos: int
    thresholds: dict
    held_out: dict
    held_out_per_video: dict
    in_sample: dict


@dataclass
class TemporalReport:
    dataset: dict
    integrity: dict
    folds: list
    aggregate: dict
    aggregate_per_video: dict
    alarm: dict
    separation: dict
    padding: dict
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        d = self.dataset
        agg = self.aggregate
        lines = [
            "# Guardian's rules against sixty drivers",
            "",
            f"_Generated {self.generated_at}._",
            "",
            "## Dataset",
            "",
            f"- **Source**: {d['name']}",
            f"- **Repository**: {d['url']}",
            f"- **Licence**: {d['licence']}",
            f"- **Citation**: {d['citation']}",
            f"- **Participants**: {d['subjects']}",
            f"- **Normalisation**: {d['normalisation']}",
            "",
            "## Split integrity",
            "",
            "The files carry no participant id, so subject-disjointness cannot be",
            "confirmed directly. Its necessary consequence can be, and was:",
            "",
            f"- Windows appearing in both the train and test side of a fold: "
            f"**{sum(self.integrity['train_test_overlap_per_fold'].values())}**",
            f"- Windows shared between two folds' test sets: "
            f"**{self.integrity['windows_shared_between_test_folds']}**",
            f"- Verdict: **{'leak-free' if self.integrity['leak_free'] else 'LEAKING - every number below is inflated'}**",
            "",
            "## Result",
            "",
            "Fit on four folds, scored on the fifth, five times. `held out` is the",
            "number that counts; `in sample` is the same predictor scored on the",
            "rows it was fitted on, and the gap between them is what this table is",
            "for.",
            "",
            "| predictor | held-out accuracy | held-out macro-F1 | in-sample macro-F1 | gap |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, row in agg.items():
            lines.append(
                f"| {name} | {row['held_out_accuracy']} | {row['held_out_macro_f1']} "
                f"| {row['in_sample_macro_f1']} | {row['overfit_gap']:+} |"
            )
        if self.aggregate_per_video:
            lines += [
                "",
                "## Per session, which is what a co-pilot actually emits",
                "",
                "Guardian does not publish a fresh opinion every few minutes and",
                "leave the driver to average them - it smooths window verdicts into",
                "one trip state. Scored the same way here, by majority vote over each",
                f"driving session ({next(iter(self.aggregate_per_video.values()))['videos']}",
                "held-out sessions in total, every driver unseen):",
                "",
                "| predictor | session accuracy | session macro-F1 |",
                "|---|---:|---:|",
            ]
            for name, row in self.aggregate_per_video.items():
                lines.append(f"| {name} | {row['accuracy']} | {row['macro_f1']} |")

        if self.alarm:
            first = next(iter(self.alarm.values()))
            lines += [
                "",
                "## Collapsed to the two states Guardian actually ships",
                "",
                "The cabin engine either raises a drowsiness alarm or stays quiet.",
                "Restricted to the held-out sessions whose truth is unambiguous",
                f"(`alert` or `drowsy`, {first['sessions']} of them - the middle class",
                "dropped), the two ways of being wrong cost different things: a miss",
                "is a safety failure now, a false alarm is how a driver learns to",
                "ignore the system.",
                "",
                "| predictor | drowsy caught | recall on drowsy | false alarms on alert | false-alarm rate |",
                "|---|---:|---:|---:|---:|",
            ]
            for name, row in self.alarm.items():
                lines.append(
                    f"| {name} | {row['drowsy_caught']}/{row['drowsy_caught'] + row['drowsy_missed']} "
                    f"| {row['recall_on_drowsy']} | {row['false_alarms_on_alert']} "
                    f"| {row['false_alarm_rate']} |"
                )
            lines += [
                "",
                "Read honestly, this table does not favour the rule. It sits at a",
                "quiet, insensitive operating point - almost never wrong about an",
                "alert driver, and silent through most of the drowsy ones - while",
                "logistic regression catches substantially more drowsy sessions for a",
                "few more false alarms, and dominates the rule on this particular",
                "trade-off. Part of that is an objective mismatch worth naming: the",
                "cut points were chosen to maximise three-class macro-F1, which is",
                "not the alarm objective, so this table scores an operating point the",
                "search never aimed at. Part of it is not an excuse - a rule tuned",
                "for macro-F1 is the rule this repository would actually ship, and on",
                "sixty strangers it would stay quiet through two drowsy drives in",
                "three.",
            ]

        lines += [
            "",
            "## Which signal carries it",
            "",
            "An AUC below 0.5 is not a weak signal - it is a strong one pointing the",
            "other way. `separation` is the distance from chance in whichever",
            "direction the statistic actually runs.",
            "",
            "| statistic | AUC drowsy vs alert | direction | separation | alert median | drowsy median |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for name, row in self.separation.items():
            lines.append(
                f"| {name} | {row['auc_drowsy_vs_alert']} | {row['direction']} "
                f"| {row['separation']} | {row['alert_median']} | {row['drowsy_median']} |"
            )
        lines += [
            "",
            "## Thresholds chosen per fold",
            "",
            "Chosen by coordinate ascent on the training folds only. Stability",
            "across folds is itself evidence: a cut point that moves every time is",
            "fitting the drivers, not the state.",
            "",
            "| fold | duration_mean_drowsy | duration_p90_micro | duration_mean_low | velocity_mean_low |",
            "|---|---:|---:|---:|---:|",
        ]
        for fold in self.folds:
            t = fold["thresholds"]
            lines.append(
                f"| {fold['fold']} | {t['duration_mean_drowsy']} | {t['duration_p90_micro']} "
                f"| {t['duration_mean_low']} | {t['velocity_mean_low']} |"
            )
        lines += [
            "",
            "## What this table is not",
            "",
            "It is not a bid for state of the art. The authors' own temporal model",
            "was designed and trained for exactly this task on exactly this data and",
            "reports better accuracy than anything here; a four-threshold rule is not",
            "competing with it. The comparison being made is narrower and, for this",
            "project, more useful: given identical inputs and an identical",
            "subject-disjoint protocol, does fitting a model to other drivers beat",
            "cutting a per-driver statistic at a fixed point? Read the `gap` column",
            "before the score column.",
            "",
            "## Caveats",
            "",
        ]
        for caveat in d["caveats"]:
            lines.append(f"- {caveat}")
        lines.append("")
        return "\n".join(lines)


def cross_validate(
    folds: Sequence[Fold],
    grid: Optional[dict[str, tuple[float, ...]]] = None,
    seed: int = 42,
    verbose: bool = True,
    boundaries: Optional[dict[int, Sequence[int]]] = None,
) -> TemporalReport:
    """The whole experiment: five folds in, one report out."""
    results: list[FoldResult] = []
    held_out_truth: dict[str, list[np.ndarray]] = {}
    held_out_pred: dict[str, list[np.ndarray]] = {}
    in_sample_truth: dict[str, list[np.ndarray]] = {}
    in_sample_pred: dict[str, list[np.ndarray]] = {}
    video_truth: dict[str, list[np.ndarray]] = {}
    video_pred: dict[str, list[np.ndarray]] = {}
    boundaries = boundaries or {}

    all_stats, all_labels = [], []

    for fold in folds:
        train_stats = reduce_windows(fold.train_windows)
        test_stats = reduce_windows(fold.test_windows)
        all_stats.append(test_stats)
        all_labels.append(fold.test_labels)

        rule, train_f1 = search_rule(train_stats, fold.train_labels, grid)
        models = _fit_predictors(train_stats, fold.train_windows, fold.train_labels, seed)

        test_pred = _predict_all(models, rule, test_stats, fold.test_windows)
        train_pred = _predict_all(models, rule, train_stats, fold.train_windows)

        starts = boundaries.get(fold.index)
        per_video: dict[str, dict] = {}

        for name in test_pred:
            held_out_pred.setdefault(name, []).append(test_pred[name])
            held_out_truth.setdefault(name, []).append(fold.test_labels)
            in_sample_pred.setdefault(name, []).append(train_pred[name])
            in_sample_truth.setdefault(name, []).append(fold.train_labels)
            if starts is not None:
                truth = video_labels(fold.test_labels, starts)
                voted = vote_by_video(test_pred[name], starts)
                video_truth.setdefault(name, []).append(truth)
                video_pred.setdefault(name, []).append(voted)
                per_video[name] = _score(truth, voted)

        results.append(
            FoldResult(
                fold=fold.index,
                n_train=fold.n_train,
                n_test=fold.n_test,
                n_test_videos=len(starts) if starts is not None else 0,
                thresholds=asdict(rule),
                held_out={k: _score(fold.test_labels, v) for k, v in test_pred.items()},
                held_out_per_video=per_video,
                in_sample={k: _score(fold.train_labels, v) for k, v in train_pred.items()},
            )
        )
        if verbose:
            r = results[-1].held_out["rule"]
            b = results[-1].held_out["boosted_on_stats"]
            print(
                f"[temporal] fold {fold.index}: train macro-F1 {train_f1:.4f} | "
                f"held-out rule {r['macro_f1']:.4f} / boosted {b['macro_f1']:.4f}"
            )

    # Pooling the five held-out folds gives one number over all 60 drivers,
    # every one of them unseen by the predictor that scored them.
    aggregate = {}
    for name in held_out_pred:
        ho_true = np.concatenate(held_out_truth[name])
        ho_pred = np.concatenate(held_out_pred[name])
        is_true = np.concatenate(in_sample_truth[name])
        is_pred = np.concatenate(in_sample_pred[name])
        ho = _score(ho_true, ho_pred)
        ins = _score(is_true, is_pred)
        aggregate[name] = {
            "held_out_accuracy": ho["accuracy"],
            "held_out_macro_f1": ho["macro_f1"],
            "held_out_per_class_recall": {
                k: v["recall"] for k, v in ho["per_class"].items()
            },
            "in_sample_macro_f1": ins["macro_f1"],
            "overfit_gap": round(ins["macro_f1"] - ho["macro_f1"], 4),
        }

    per_video_aggregate: dict = {}
    alarm_aggregate: dict = {}
    for name in video_pred:
        truth = np.concatenate(video_truth[name])
        pred = np.concatenate(video_pred[name])
        scored = _score(truth, pred)
        alarm_aggregate[name] = alarm_view(truth, pred)
        per_video_aggregate[name] = {
            "videos": int(len(truth)),
            "accuracy": scored["accuracy"],
            "macro_f1": scored["macro_f1"],
            "per_class_recall": {k: v["recall"] for k, v in scored["per_class"].items()},
        }

    stats = np.concatenate(all_stats)
    labels = np.concatenate(all_labels)

    return TemporalReport(
        dataset=asdict(rldd.PROVENANCE) | rldd.summarise_folds(folds),
        integrity=rldd.check_integrity(folds),
        folds=[asdict(r) for r in results],
        aggregate=aggregate,
        aggregate_per_video=per_video_aggregate,
        alarm=alarm_aggregate,
        separation=feature_separation(stats, labels),
        padding=padding_report(np.concatenate([f.test_windows for f in folds])),
    )


def run(
    cache_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    seed: int = 42,
) -> TemporalReport:
    """Fetch if needed, cross-validate, write the report."""
    root = rldd.ensure_release(cache_dir)
    folds = rldd.load_folds(root)
    boundaries = rldd.load_video_boundaries(root, folds)
    report = cross_validate(folds, seed=seed, boundaries=boundaries)

    out_dir = Path(out_dir or DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rldd_report.json").write_text(report.to_json(), encoding="utf-8")
    (out_dir / "rldd_report.md").write_text(report.to_markdown(), encoding="utf-8")
    print(f"[temporal] wrote {out_dir / 'rldd_report.md'}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Guardian's rules against 60 drivers.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    report = run(args.cache_dir, args.out_dir, args.seed)
    print()
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
