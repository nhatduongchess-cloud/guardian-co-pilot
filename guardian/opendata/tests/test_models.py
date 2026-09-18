"""
Tests for the two trained models.

Offline, as everywhere else in this package: no Hub, no clone, no `.npy`. The
tests that need torch skip cleanly when it is absent, because every other
predictor in `temporal.py` runs without it and the report is designed to
degrade rather than break.

The test that matters most is `test_validation_group_is_disjoint_from_training`.
Early stopping chooses the model, so if the validation rows overlap the
training rows the network is selected on data it has memorised, and the
held-out number stops meaning anything. That property cannot be eyeballed - it
is checked.
"""

from __future__ import annotations

import numpy as np
import pytest

from guardian.opendata import image_model, rldd, sequence_model
from guardian.opendata.rldd import Fold

torch = pytest.importorskip("torch", reason="recurrent model is optional")

F = {name: i for i, name in enumerate(rldd.BLINK_FEATURES)}


def _window(value=0.0, blinks=rldd.WINDOW_BLINKS, feature="duration"):
    w = np.zeros((rldd.WINDOW_BLINKS, len(rldd.BLINK_FEATURES)))
    w[-blinks:, F[feature]] = value
    return w


def _fold(index, train, train_y, test, test_y):
    return Fold(index, train, np.array(train_y), test, np.array(test_y))


# ---------------------------------------------------------------------------
# Scaling.
# ---------------------------------------------------------------------------

def test_scaler_fits_on_real_blinks_and_leaves_padding_at_zero():
    windows = np.stack([_window(4.0, blinks=10), _window(8.0, blinks=30)])
    scaled = sequence_model.RobustScaler().fit(windows).transform(windows)
    padding = ~rldd.padding_mask(windows)
    assert (scaled[padding] == 0).all()
    # The 20 padded slots must not have contributed to the median. Fitting over
    # everything would put the median near zero and shift every real blink.
    assert scaled[~padding].std() > 0


def test_scaler_clips_instead_of_letting_one_outlier_set_the_scale():
    rng = np.random.default_rng(0)
    windows = rng.normal(0.0, 1.0, size=(40, rldd.WINDOW_BLINKS, 4))
    windows[0, 0, F["frequency"]] = 10_000.0
    scaled = sequence_model.RobustScaler().fit(windows).transform(windows)
    assert np.abs(scaled).max() <= sequence_model.CLIP + 1e-6


def test_scaler_survives_a_feature_with_no_spread():
    windows = np.ones((5, rldd.WINDOW_BLINKS, 4)) * 3.0
    scaled = sequence_model.RobustScaler().fit(windows).transform(windows)
    assert np.isfinite(scaled).all()


# ---------------------------------------------------------------------------
# The validation split. The leak test.
# ---------------------------------------------------------------------------

def _linked_folds():
    """Two folds whose training arrays really do contain the other's test rows.

    This mirrors the shipped release, where fold k's training array is exactly
    the union of the other folds' test arrays.
    """
    a_test = np.stack([_window(float(i)) for i in range(4)])
    b_test = np.stack([_window(float(i)) for i in range(10, 14)])
    c_test = np.stack([_window(float(i)) for i in range(20, 24)])
    return [
        _fold(1, np.concatenate([b_test, c_test]), [0] * 4 + [10] * 4, a_test, [5] * 4),
        _fold(2, np.concatenate([a_test, c_test]), [5] * 4 + [10] * 4, b_test, [0] * 4),
        _fold(3, np.concatenate([a_test, b_test]), [5] * 4 + [0] * 4, c_test, [10] * 4),
    ]


def test_validation_group_is_disjoint_from_training():
    folds = _linked_folds()
    train_x, train_y, val_x, val_y, val_index = sequence_model.split_validation_group(
        folds[0], folds
    )
    assert val_index in (2, 3)
    assert len(train_x) + len(val_x) == folds[0].n_train
    assert len(train_y) == len(train_x) and len(val_y) == len(val_x)
    train_sig = {r.tobytes() for r in train_x.reshape(len(train_x), -1)}
    val_sig = {r.tobytes() for r in val_x.reshape(len(val_x), -1)}
    assert not (train_sig & val_sig), "early stopping would select on training rows"


def test_validation_group_comes_from_a_single_other_fold():
    folds = _linked_folds()
    _, _, val_x, _, val_index = sequence_model.split_validation_group(folds[0], folds)
    chosen = next(f for f in folds if f.index == val_index)
    chosen_sig = {r.tobytes() for r in chosen.test_windows.reshape(chosen.n_test, -1)}
    val_sig = {r.tobytes() for r in val_x.reshape(len(val_x), -1)}
    assert val_sig <= chosen_sig, "validation rows must be one fold's drivers, not a mix"


def test_validation_group_can_be_requested_by_index():
    folds = _linked_folds()
    *_, val_index = sequence_model.split_validation_group(folds[0], folds, validation_index=3)
    assert val_index == 3


def test_validation_group_refuses_when_the_folds_do_not_overlap():
    a = _fold(1, np.stack([_window(1.0)]), [0], np.stack([_window(2.0)]), [0])
    b = _fold(2, np.stack([_window(3.0)]), [0], np.stack([_window(9.0)]), [0])
    with pytest.raises(ValueError, match="driver-disjoint validation split"):
        sequence_model.split_validation_group(a, [a, b])


def test_validation_group_needs_more_than_one_fold():
    only = _fold(1, np.stack([_window(1.0)]), [0], np.stack([_window(2.0)]), [0])
    with pytest.raises(ValueError, match="at least two folds"):
        sequence_model.split_validation_group(only, [only])


# ---------------------------------------------------------------------------
# Training end to end, on data a linear model could also solve.
# ---------------------------------------------------------------------------

def _separable_group(n, level, rng):
    windows = rng.normal(0.0, 0.2, size=(n, rldd.WINDOW_BLINKS, 4))
    windows[:, :, F["duration"]] += level
    return windows


def test_train_fold_learns_a_separable_signal_and_predicts_the_vocabulary():
    rng = np.random.default_rng(3)
    groups = []
    for _ in range(3):
        x = np.concatenate(
            [_separable_group(12, lvl, rng) for lvl in (0.0, 3.0, 6.0)]
        )
        y = np.array([0] * 12 + [5] * 12 + [10] * 12)
        groups.append((x, y))

    folds = []
    for k in range(3):
        others = [g for i, g in enumerate(groups) if i != k]
        train_x = np.concatenate([g[0] for g in others])
        train_y = np.concatenate([g[1] for g in others])
        folds.append(Fold(k + 1, train_x, train_y, groups[k][0], groups[k][1]))

    trained = sequence_model.train_fold(
        folds[0],
        folds,
        sequence_model.TrainingConfig(epochs=30, patience=8, hidden=16),
        verbose=False,
    )
    predicted = trained.predict(folds[0].test_windows)
    assert set(np.unique(predicted)) <= {0, 5, 10}
    assert len(predicted) == folds[0].n_test
    assert trained.best_epoch >= 1
    assert trained.n_validation > 0
    from guardian.opendata.temporal import macro_f1

    assert macro_f1(folds[0].test_labels, predicted) > 0.6


def test_trained_fold_records_which_fold_it_validated_on():
    rng = np.random.default_rng(5)
    groups = [
        (
            np.concatenate([_separable_group(6, lvl, rng) for lvl in (0.0, 4.0)]),
            np.array([0] * 6 + [10] * 6),
        )
        for _ in range(3)
    ]
    folds = []
    for k in range(3):
        others = [g for i, g in enumerate(groups) if i != k]
        folds.append(
            Fold(
                k + 1,
                np.concatenate([g[0] for g in others]),
                np.concatenate([g[1] for g in others]),
                groups[k][0],
                groups[k][1],
            )
        )
    trained = sequence_model.train_fold(
        folds[0], folds, sequence_model.TrainingConfig(epochs=3, patience=3), verbose=False
    )
    assert trained.validation_fold != trained.fold
    assert trained.n_train + trained.n_validation == folds[0].n_train


# ---------------------------------------------------------------------------
# The image experiment. Its honesty is the part under test.
# ---------------------------------------------------------------------------

def test_image_scoring_is_hand_checked():
    drowsy = image_model.CLASS_ORDER.index("drowsy")
    alert = image_model.CLASS_ORDER.index("alert")
    y_true = np.array([drowsy, drowsy, drowsy, drowsy, alert, alert])
    y_pred = np.array([drowsy, drowsy, drowsy, alert, alert, drowsy])
    scores = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.6])
    result = image_model._score("x", y_true, y_pred, scores)
    assert result.drowsy_recall == pytest.approx(0.75)
    assert result.alert_recall == pytest.approx(0.5)
    assert result.balanced_accuracy == pytest.approx(0.625)
    assert result.accuracy == pytest.approx(4 / 6, abs=1e-4)


def test_balanced_accuracy_refuses_to_reward_guessing_the_big_class():
    drowsy = image_model.CLASS_ORDER.index("drowsy")
    alert = image_model.CLASS_ORDER.index("alert")
    y_true = np.array([drowsy] * 9 + [alert])
    all_drowsy = np.full(10, drowsy)
    result = image_model._score("x", y_true, all_drowsy, np.full(10, 0.9))
    assert result.accuracy == pytest.approx(0.9)
    assert result.balanced_accuracy == pytest.approx(0.5)  # exactly chance


def test_report_leads_with_the_caveat_not_the_number():
    report = image_model.ImageReport(
        trained_on={"repo_id": "a/b", "licence": "none", "provenance": "p"},
        protocol="random_frame_split - test protocol",
        in_dataset={
            "name": "in", "n": 10, "accuracy": 0.99, "balanced_accuracy": 0.99,
            "drowsy_recall": 1.0, "alert_recall": 0.98, "auc": 0.99,
        },
        cross_dataset={
            "name": "cross", "n": 10, "accuracy": 0.5, "balanced_accuracy": 0.49,
            "drowsy_recall": 0.6, "alert_recall": 0.38, "auc": 0.5,
        },
        collapse={"balanced_accuracy_drop": -0.5, "verdict": "The weights do not transfer."},
        epochs=3,
    )
    markdown = report.to_markdown()
    caveat_position = markdown.index("frames, not")
    number_position = markdown.index("0.99")
    assert caveat_position < number_position, "the number must not arrive before the warning"
    assert "not comparable" in markdown
    assert "random_frame_split" in markdown


def test_image_classes_are_guardian_vocabulary_and_match_the_registry():
    assert set(image_model.CLASS_ORDER) == {"alert", "drowsy"}
    from guardian.opendata import sources

    for source in set(sources.REGISTRY.values()):
        assert set(source.label_map.values()) <= set(image_model.CLASS_ORDER)
