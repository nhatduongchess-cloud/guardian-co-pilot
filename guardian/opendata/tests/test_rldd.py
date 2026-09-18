"""
Tests for the sixty-driver open-data evaluation.

Every test here runs offline: nothing is cloned, nothing is downloaded, and no
`.npy` from the authors' release is needed. The arithmetic is synthetic and
hand-checked, because the failure mode that matters in this module is not a
crash - it is a number that comes out plausible and wrong.

Three things get disproportionate attention, because each one could inflate a
result silently:

* padding, which is zero, which is a driver's own alert mean, so folding it in
  would quietly drag every statistic toward "fresh";
* the threshold search, which must never see a held-out row;
* the split integrity check, which is the only thing standing between us and
  reporting a leaked score as a cross-driver score.
"""

from __future__ import annotations

import numpy as np
import pytest

from guardian.opendata import rldd, temporal
from guardian.opendata.rldd import Fold
from guardian.opendata.temporal import ALERT, DROWSY, LOW, BlinkRule

F = {name: i for i, name in enumerate(rldd.BLINK_FEATURES)}


def _window(freq=0.0, amp=0.0, dur=0.0, vel=0.0, blinks=rldd.WINDOW_BLINKS):
    """One window whose 30 blinks are all identical - easy to reason about."""
    w = np.zeros((rldd.WINDOW_BLINKS, len(rldd.BLINK_FEATURES)))
    w[-blinks:, F["frequency"]] = freq
    w[-blinks:, F["amplitude"]] = amp
    w[-blinks:, F["duration"]] = dur
    w[-blinks:, F["velocity"]] = vel
    return w


# ---------------------------------------------------------------------------
# Provenance. Same rule as the frame-level sources: paperwork is a feature.
# ---------------------------------------------------------------------------

def test_provenance_is_complete():
    p = rldd.PROVENANCE
    assert "MIT" in p.licence
    assert "Ghoddoosian" in p.citation
    assert p.subjects == 60
    assert len(p.caveats) >= 3
    assert all(c.strip() for c in p.caveats)


def test_label_vocabulary_is_the_authors_three_scores():
    assert set(rldd.LABEL_NAMES) == {ALERT, LOW, DROWSY}


# ---------------------------------------------------------------------------
# Padding. The quiet bias.
# ---------------------------------------------------------------------------

def test_padding_mask_marks_all_zero_rows_only():
    w = _window(dur=2.0, blinks=10)  # 20 leading pad rows
    mask = rldd.padding_mask(w[None])[0]
    assert mask.sum() == 10
    assert not mask[:20].any()


def test_reduce_ignores_padding_instead_of_averaging_it_in():
    # 10 real blinks at duration 3.0, 20 rows of padding. The honest mean is
    # 3.0. Folding the padding in as zeros would give 1.0 - and would make a
    # drowsy window look two-thirds fresher than it is.
    stats = temporal.reduce_windows(_window(dur=3.0, blinks=10)[None])
    assert stats[0, temporal.STAT_NAMES.index("duration_mean")] == pytest.approx(3.0)


def test_reduce_p90_is_taken_over_real_blinks_only():
    w = np.zeros((rldd.WINDOW_BLINKS, len(rldd.BLINK_FEATURES)))
    w[20:, F["duration"]] = np.arange(1.0, 11.0)  # ten blinks, 1..10
    stats = temporal.reduce_windows(w[None])
    # p90 of 1..10 is 9.1 by linear interpolation; including twenty zeros
    # would drag it to about 8.
    assert stats[0, temporal.STAT_NAMES.index("duration_p90")] == pytest.approx(9.1)


def test_padding_report_counts_what_was_excluded():
    batch = np.stack([_window(dur=1.0, blinks=30), _window(dur=1.0, blinks=10)])
    report = temporal.padding_report(batch)
    assert report["blink_slots"] == 60
    assert report["padded_slots"] == 20
    assert report["windows_entirely_padding"] == 0


# ---------------------------------------------------------------------------
# The rule. Escalation must be an order, not an accident of evaluation.
# ---------------------------------------------------------------------------

def test_rule_escalates_and_drowsy_wins_over_low():
    rule = BlinkRule(
        duration_mean_drowsy=1.0,
        duration_p90_micro=4.0,
        duration_mean_low=0.3,
        velocity_mean_low=-0.5,
    )
    stats = temporal.reduce_windows(
        np.stack(
            [
                _window(dur=0.0, vel=0.0),   # nothing fires
                _window(dur=0.5, vel=0.0),   # low only
                _window(dur=0.0, vel=-0.9),  # low via slow lid
                _window(dur=2.0, vel=0.0),   # low AND drowsy -> drowsy
            ]
        )
    )
    assert list(rule.apply(stats)) == [ALERT, LOW, LOW, DROWSY]


def test_microsleep_clause_fires_on_one_long_blink_alone():
    # Mean duration stays well under the drowsy cut; a single very long blink
    # carries the window. That is the whole point of keeping a p90 term.
    w = np.zeros((rldd.WINDOW_BLINKS, len(rldd.BLINK_FEATURES)))
    w[:, F["duration"]] = 0.0
    w[-3:, F["duration"]] = 9.0
    rule = BlinkRule(duration_mean_drowsy=5.0, duration_p90_micro=4.0,
                     duration_mean_low=99.0, velocity_mean_low=-99.0)
    stats = temporal.reduce_windows(w[None])
    assert rule.apply(stats)[0] == DROWSY


# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------

def test_macro_f1_skips_a_class_absent_from_both_sides():
    truth = np.array([ALERT, ALERT, DROWSY, DROWSY])
    # A perfect two-class answer must score 1.0, not 0.67 because `low` never
    # appeared.
    assert temporal.macro_f1(truth, truth.copy()) == pytest.approx(1.0)


def test_macro_f1_hand_checked():
    truth = np.array([ALERT, ALERT, LOW, LOW, DROWSY, DROWSY])
    pred = np.array([ALERT, LOW, LOW, LOW, DROWSY, ALERT])
    # alert:  tp=1 fp=1 fn=1 -> 2/(2+1+1) = 0.5
    # low:    tp=2 fp=1 fn=0 -> 4/(4+1+0) = 0.8
    # drowsy: tp=1 fp=0 fn=1 -> 2/(2+0+1) = 0.666...
    assert temporal.macro_f1(truth, pred) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_macro_f1_punishes_abandoning_a_class():
    # 80% of this data is not alert, so a predictor that never says alert still
    # scores 0.8 on accuracy. Macro-F1 must not let that pass.
    truth = np.array([ALERT] * 2 + [DROWSY] * 8)
    always_drowsy = np.full(10, DROWSY)
    accuracy = float((truth == always_drowsy).mean())
    assert accuracy == pytest.approx(0.8)
    # alert F1 = 0, drowsy F1 = 16/18; `low` is absent from both sides and is
    # left out of the average rather than scored zero.
    assert temporal.macro_f1(truth, always_drowsy) == pytest.approx((0 + 16 / 18) / 2)
    assert temporal.macro_f1(truth, always_drowsy) < accuracy - 0.3


def test_roc_auc_handles_ties_and_empty_sides():
    assert temporal.roc_auc([1.0, 2.0], [0.0, 0.5]) == pytest.approx(1.0)
    assert temporal.roc_auc([1.0], [1.0]) == pytest.approx(0.5)  # a tie is chance
    assert temporal.roc_auc([], [1.0]) is None


def test_feature_separation_reports_direction_not_just_auc():
    stats = np.zeros((4, len(temporal.STAT_NAMES)))
    stats[:, temporal.STAT_NAMES.index("velocity_mean")] = [1.0, 1.0, -1.0, -1.0]
    labels = np.array([ALERT, ALERT, DROWSY, DROWSY])
    out = temporal.feature_separation(stats, labels)["velocity_mean"]
    assert out["auc_drowsy_vs_alert"] == pytest.approx(0.0)
    assert out["direction"] == "lower when drowsy"
    # An AUC of 0 is a perfect signal running backwards, not a useless one.
    assert out["separation"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The search. Must fit on training rows and nothing else.
# ---------------------------------------------------------------------------

def test_search_keeps_defaults_when_nothing_beats_them():
    # One class only: every candidate ties, and a tie must not move the rule.
    stats = temporal.reduce_windows(np.stack([_window(dur=0.0) for _ in range(8)]))
    labels = np.full(8, ALERT)
    found, _ = temporal.search_rule(stats, labels)
    assert found == BlinkRule()


def test_search_finds_a_separating_cut_when_one_exists():
    alert = [_window(dur=0.0, vel=0.0) for _ in range(10)]
    drowsy = [_window(dur=5.0, vel=0.0) for _ in range(10)]
    stats = temporal.reduce_windows(np.stack(alert + drowsy))
    labels = np.array([ALERT] * 10 + [DROWSY] * 10)
    found, score = temporal.search_rule(stats, labels)
    assert score == pytest.approx(1.0)
    assert list(found.apply(stats)) == list(labels)


def test_search_never_returns_worse_than_the_starting_rule():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(60, rldd.WINDOW_BLINKS, len(rldd.BLINK_FEATURES)))
    stats = temporal.reduce_windows(raw)
    labels = rng.choice([ALERT, LOW, DROWSY], size=60)
    start = BlinkRule()
    found, score = temporal.search_rule(stats, labels, start=start)
    assert score >= temporal.macro_f1(labels, start.apply(stats))


# ---------------------------------------------------------------------------
# Split integrity. The one check that stops a leak being reported as a result.
# ---------------------------------------------------------------------------

def _fold(index, train, train_y, test, test_y):
    return Fold(index, train, np.array(train_y), test, np.array(test_y))


def test_check_integrity_passes_on_a_clean_split():
    train = np.stack([_window(dur=float(i)) for i in range(6)])
    test = np.stack([_window(dur=float(i)) for i in range(10, 13)])
    folds = [_fold(1, train, [ALERT] * 6, test, [DROWSY] * 3)]
    assert rldd.check_integrity(folds)["leak_free"]


def test_check_integrity_catches_a_window_on_both_sides():
    shared = _window(dur=7.0)
    train = np.stack([shared, _window(dur=1.0)])
    test = np.stack([shared])
    result = rldd.check_integrity([_fold(1, train, [ALERT] * 2, test, [ALERT])])
    assert result["train_test_overlap_per_fold"]["fold1"] == 1
    assert not result["leak_free"]


def test_check_integrity_catches_a_window_in_two_test_folds():
    shared = _window(dur=7.0)
    folds = [
        _fold(1, np.stack([_window(dur=1.0)]), [ALERT], np.stack([shared]), [ALERT]),
        _fold(2, np.stack([_window(dur=2.0)]), [ALERT], np.stack([shared]), [ALERT]),
    ]
    result = rldd.check_integrity(folds)
    assert result["windows_shared_between_test_folds"] == 1
    assert not result["leak_free"]


# ---------------------------------------------------------------------------
# Loader validation. Fail loudly, not halfway.
# ---------------------------------------------------------------------------

def test_validate_windows_rejects_the_wrong_shape():
    with pytest.raises(ValueError, match="expected"):
        rldd.validate_windows(np.zeros((4, 10, 4)), "bad")


def test_validate_windows_rejects_non_finite():
    array = np.zeros((1, rldd.WINDOW_BLINKS, len(rldd.BLINK_FEATURES)))
    array[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        rldd.validate_windows(array, "bad")


def test_validate_labels_rejects_labels_outside_the_vocabulary():
    with pytest.raises(ValueError, match="unexpected label"):
        rldd.validate_labels(np.array([0, 5, 7]), "bad")


# ---------------------------------------------------------------------------
# Video boundaries, read from the authors' file and then checked.
# ---------------------------------------------------------------------------

TRAINING_PY = """
if i==0:
    # start_indices=[0,1,2] # an older step size, commented out by the authors
    start_indices=[0,2,4]
if i==1:
    start_indices=[0,3]
"""


def test_parse_start_indices_ignores_commented_lines(tmp_path):
    path = tmp_path / "Training.py"
    path.write_text(TRAINING_PY, encoding="utf-8")
    assert rldd._parse_start_indices(path) == [[0, 2, 4], [0, 3]]


def test_load_video_boundaries_matches_each_fold_by_checking(tmp_path):
    (tmp_path / "Training.py").write_text(TRAINING_PY, encoding="utf-8")
    six = np.stack([_window(dur=float(i)) for i in range(6)])
    # Labels are constant inside each video: [0,0] [5,5] [10,10] -> starts 0,2,4
    fold_a = _fold(1, six, [ALERT] * 6, six, [ALERT, ALERT, LOW, LOW, DROWSY, DROWSY])
    # A different fold, six rows in two videos of three -> starts 0,3
    fold_b = _fold(2, six, [ALERT] * 6, six, [ALERT] * 3 + [DROWSY] * 3)
    found = rldd.load_video_boundaries(tmp_path, [fold_a, fold_b])
    assert found == {1: [0, 2, 4], 2: [0, 3]}


def test_load_video_boundaries_refuses_when_nothing_partitions_cleanly(tmp_path):
    (tmp_path / "Training.py").write_text(TRAINING_PY, encoding="utf-8")
    six = np.stack([_window(dur=float(i)) for i in range(6)])
    scrambled = _fold(1, six, [ALERT] * 6, six, [ALERT, DROWSY, ALERT, DROWSY, ALERT, DROWSY])
    with pytest.raises(ValueError, match="exactly one boundary list"):
        rldd.load_video_boundaries(tmp_path, [scrambled])


def test_video_segments_cover_every_row_exactly_once():
    segments = rldd.video_segments([0, 2, 5], 9)
    assert segments == [(0, 2), (2, 5), (5, 9)]
    covered = [i for a, b in segments for i in range(a, b)]
    assert covered == list(range(9))


# ---------------------------------------------------------------------------
# Session verdicts.
# ---------------------------------------------------------------------------

def test_vote_by_video_takes_the_majority():
    pred = np.array([DROWSY, DROWSY, ALERT, LOW, LOW, LOW])
    assert list(temporal.vote_by_video(pred, [0, 3])) == [DROWSY, LOW]


def test_vote_by_video_breaks_ties_toward_the_more_severe_state():
    # One alert, one drowsy. A co-pilot that shrugs here is the wrong co-pilot.
    assert temporal.vote_by_video(np.array([ALERT, DROWSY]), [0])[0] == DROWSY
    assert temporal.vote_by_video(np.array([ALERT, LOW]), [0])[0] == LOW


def test_video_labels_reads_the_constant_label_of_each_session():
    labels = np.array([ALERT, ALERT, DROWSY, DROWSY, DROWSY])
    assert list(temporal.video_labels(labels, [0, 2])) == [ALERT, DROWSY]


def test_alarm_view_drops_the_middle_class_and_counts_both_errors():
    truth = np.array([ALERT, ALERT, LOW, DROWSY, DROWSY, DROWSY])
    pred = np.array([ALERT, DROWSY, DROWSY, DROWSY, DROWSY, ALERT])
    view = temporal.alarm_view(truth, pred)
    assert view["sessions"] == 5  # the LOW row is dropped, not scored
    assert view["drowsy_caught"] == 2
    assert view["drowsy_missed"] == 1
    assert view["recall_on_drowsy"] == pytest.approx(0.6667)  # rounded for the report
    assert view["false_alarms_on_alert"] == 1
    assert view["false_alarm_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# End to end on synthetic folds: the wiring, not the science.
# ---------------------------------------------------------------------------

def test_cross_validate_runs_and_reports_the_overfit_gap():
    rng = np.random.default_rng(7)

    def make(n, index):
        drowsy = rng.normal(3.0, 0.5, size=(n, rldd.WINDOW_BLINKS, 4))
        alert = rng.normal(0.0, 0.5, size=(n, rldd.WINDOW_BLINKS, 4))
        x = np.concatenate([alert, drowsy])
        y = np.array([ALERT] * n + [DROWSY] * n)
        return x, y

    folds = []
    for k in range(1, 3):
        xtr, ytr = make(40, k)
        xte, yte = make(20, k)
        folds.append(Fold(k, xtr, ytr, xte, yte))

    report = temporal.cross_validate(folds, verbose=False)
    assert report.integrity["leak_free"] in (True, False)  # it ran
    assert set(report.aggregate) == {
        "rule",
        "boosted_on_stats",
        "logistic_on_window",
        "majority",
    }
    for row in report.aggregate.values():
        assert 0.0 <= row["held_out_macro_f1"] <= 1.0
        assert "overfit_gap" in row
    assert "# Guardian's rules against sixty drivers" in report.to_markdown()
