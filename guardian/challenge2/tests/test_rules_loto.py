"""
Tests for the LOTO threshold search (`loto_threshold_search`).

Two of these are *faithfulness* tests: they prove the fast paths the search
uses give byte-for-byte the same answer as the shipped engine, so the number
the search reports is the number the deployed rule engine would earn. The rest
prove the search itself behaves — recovers a known threshold, keeps the default
when the default is best, and computes an honest held-out mean.

None of these need the private trip footage: synthetic temporal frames are
injected through `trip_data=`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from guardian.challenge2.classifier import smooth_predictions
from guardian.challenge2.rules import (
    RuleEngine,
    RuleThresholds,
    _rule_labels,
    _RULE_COLUMNS,
    loto_threshold_search,
    majority_smooth,
)

CLASSES = ("alert", "distracted", "drowsy", "microsleep", "yawning")


# --------------------------------------------------------------------------
# Faithfulness: the fast paths must equal the shipped ones
# --------------------------------------------------------------------------

@pytest.mark.parametrize("window", [1, 2, 3, 5, 40, 41, 101])
def test_majority_smooth_matches_reference(window):
    rng = np.random.default_rng(0)
    labels = np.array(rng.choice(CLASSES, size=350), dtype=object)
    fast = majority_smooth(labels, window)
    ref = smooth_predictions(labels, window)
    assert list(fast) == list(ref)


def test_majority_smooth_handles_single_class_and_short_input():
    labels = np.array(["drowsy"] * 7, dtype=object)
    assert list(majority_smooth(labels, 41)) == list(labels)
    one = np.array(["alert"], dtype=object)
    assert list(majority_smooth(one, 41)) == ["alert"]


def test_rule_labels_match_engine_unsmoothed():
    rng = np.random.default_rng(1)
    n = 500
    temporal = pd.DataFrame({
        "phone_rate_w61": rng.uniform(0, 1, n),
        "eb_mean_w201":   rng.uniform(0, 0.9, n),
        "perclos_w201":   rng.uniform(0, 1, n),
        "closed_run_sec": rng.uniform(0, 3, n),
        "jaw_mean_w61":   rng.uniform(0, 0.6, n),
    })
    cols = {c: temporal[c].to_numpy() for c in _RULE_COLUMNS}
    for th in (RuleThresholds(),
               RuleThresholds(jaw_open=0.30, perclos_micro=0.55),
               RuleThresholds(phone_rate=0.05, perclos_drowsy=0.05)):
        mine = _rule_labels(cols, th)
        engine = RuleEngine(th).predict(temporal, smooth=False)
        assert list(mine) == list(engine)


# --------------------------------------------------------------------------
# Synthetic trip builders
# --------------------------------------------------------------------------

def _segment(state, n):
    """A block of `n` frames whose feature values make the DEFAULT thresholds
    emit exactly `state`."""
    base = {c: np.zeros(n) for c in _RULE_COLUMNS}
    if state == "alert":
        pass
    elif state == "drowsy":
        base["perclos_w201"][:] = 0.20   # >0.08 drowsy, <0.45 micro
        base["eb_mean_w201"][:] = 0.20   # >0.12
    elif state == "microsleep":
        base["perclos_w201"][:] = 0.60   # >0.45
        base["closed_run_sec"][:] = 2.0  # >1.2
    elif state == "yawning":
        base["jaw_mean_w61"][:] = 0.40   # >0.20 (overrides all)
    elif state == "distracted":
        base["phone_rate_w61"][:] = 0.80  # >0.15
    else:
        raise ValueError(state)
    return base


def _trip(states_and_lengths):
    cols = {c: [] for c in _RULE_COLUMNS}
    y = []
    for state, n in states_and_lengths:
        seg = _segment(state, n)
        for c in _RULE_COLUMNS:
            cols[c].append(seg[c])
        y += [state] * n
    frame = pd.DataFrame({c: np.concatenate(cols[c]) for c in _RULE_COLUMNS})
    return frame, np.array(y, dtype=object)


# --------------------------------------------------------------------------
# The search itself
# --------------------------------------------------------------------------

def test_perfect_and_keeps_default_when_separable():
    # Three trips, each cleanly separable by the shipped defaults. No fold has
    # any reason to move, so every fold keeps the default and scores 100.
    trips = {}
    for i, mix in enumerate([
        [("alert", 100), ("drowsy", 100), ("microsleep", 100)],
        [("alert", 100), ("yawning", 100), ("distracted", 100)],
        [("alert", 100), ("drowsy", 100), ("yawning", 100)],
    ]):
        trips[f"S0{i}"] = _trip(mix)

    report = loto_threshold_search(trip_data=trips, smooth_window=1, verbose=False)

    assert len(report.folds) == 3
    assert report.mean_val_composite == 100.0
    assert report.n_selecting_default == 3
    assert all(f.matches_default for f in report.folds)


def test_search_moves_off_default_to_fix_false_yawning():
    # Reproduces the real T04 failure mode: genuinely-drowsy frames whose jaw
    # sits just above the default 0.20, so the default mislabels them `yawning`.
    # Raising jaw_open past 0.22 fixes it — the search must discover that.
    def drowsy_with_high_jaw(n):
        seg = _segment("drowsy", n)
        seg["jaw_mean_w61"][:] = 0.22   # just above the 0.20 default
        return seg

    def trip():
        cols = {c: [] for c in _RULE_COLUMNS}
        y = []
        for state, n, seg in [("alert", 100, _segment("alert", 100)),
                              ("drowsy", 100, drowsy_with_high_jaw(100))]:
            for c in _RULE_COLUMNS:
                cols[c].append(seg[c])
            y += [state] * n
        frame = pd.DataFrame({c: np.concatenate(cols[c]) for c in _RULE_COLUMNS})
        return frame, np.array(y, dtype=object)

    trips = {"A": trip(), "B": trip()}
    report = loto_threshold_search(trip_data=trips, smooth_window=1, verbose=False)

    # Every fold should have raised jaw_open above the default and recovered a
    # perfect score.
    assert all(f.thresholds["jaw_open"] > 0.20 for f in report.folds)
    assert all(not f.matches_default for f in report.folds)
    assert report.mean_val_composite == 100.0


def test_requires_at_least_two_trips():
    one = {"only": _trip([("alert", 50), ("drowsy", 50)])}
    with pytest.raises(ValueError):
        loto_threshold_search(trip_data=one, verbose=False)


def test_report_is_json_serialisable():
    import json
    from dataclasses import asdict

    trips = {
        "A": _trip([("alert", 80), ("drowsy", 80)]),
        "B": _trip([("alert", 80), ("microsleep", 80)]),
    }
    report = loto_threshold_search(trip_data=trips, smooth_window=1, verbose=False)
    blob = json.dumps(asdict(report), indent=2)
    back = json.loads(blob)
    assert "mean_val_composite" in back
    assert len(back["folds"]) == 2
