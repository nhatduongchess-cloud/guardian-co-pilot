"""
Tests for the open-data evaluation.

Every test here runs offline: no dataset download, no MediaPipe, no GPU. The
statistics are the part that can be wrong in a way that quietly flatters the
result, so that is the part under test.
"""

from __future__ import annotations

import math

import pytest

from guardian.opendata import FrameRecord, Thresholds, roc_auc, sources, summarise


# --------------------------------------------------------------------------
# Registry: provenance is a feature, so it is tested like one.
# --------------------------------------------------------------------------

def test_every_source_declares_licence_and_provenance():
    for source in set(sources.REGISTRY.values()):
        assert source.licence.strip(), f"{source.key} has no licence statement"
        assert source.provenance.strip(), f"{source.key} has no provenance"
        assert source.repo_id.count("/") == 1


def test_label_maps_only_produce_guardian_vocabulary():
    for source in set(sources.REGISTRY.values()):
        assert set(source.label_map.values()) <= {"drowsy", "alert"}


def test_normalise_handles_index_name_and_unknown():
    ddd = sources.get("ddd")
    assert ddd.normalise(0) == "drowsy"
    assert ddd.normalise("Non Drowsy") == "alert"
    assert ddd.normalise("non drowsy") == "alert"  # case-insensitive
    assert ddd.normalise("banana") is None


def test_aliases_resolve_and_unknown_key_lists_options():
    assert sources.get("akahana") is sources.get("ddd")
    with pytest.raises(KeyError) as excinfo:
        sources.get("nope")
    assert "ddd" in str(excinfo.value)


# --------------------------------------------------------------------------
# AUC
# --------------------------------------------------------------------------

def test_auc_perfect_separation_is_one():
    assert roc_auc([0.9, 0.8, 0.7], [0.1, 0.2, 0.3]) == 1.0


def test_auc_reversed_separation_is_zero():
    assert roc_auc([0.1, 0.2], [0.8, 0.9]) == 0.0


def test_auc_identical_distributions_is_half():
    assert roc_auc([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_auc_needs_both_classes():
    assert roc_auc([], [0.1]) is None
    assert roc_auc([0.1], []) is None


def test_auc_handles_ties_without_bias():
    # Hand-counted over all 9 (positive, negative) pairs: 3 clear wins for the
    # 1.0, 2 more for 0.5 > 0.0, and 4 ties worth half each = 7/9.
    value = roc_auc([1.0, 0.5, 0.5], [0.5, 0.5, 0.0])
    assert math.isclose(value, 7 / 9, abs_tol=1e-6)


# --------------------------------------------------------------------------
# summarise(): the numbers that end up in the report
# --------------------------------------------------------------------------

def _records(spec):
    return [FrameRecord(label=l, eye_blink=e, jaw_open=j, face_found=f) for l, e, j, f in spec]


def test_frames_without_a_face_are_excluded_not_counted_as_open_eyes():
    """A missed detection must not be averaged in as 'eyes open'."""
    records = _records([
        ("drowsy", 0.9, 0.0, True),
        ("drowsy", 0.9, 0.0, True),
        ("drowsy", 0.0, 0.0, False),  # no face: must be excluded
    ])
    out = summarise(records)
    assert out["frames_evaluated"] == 3
    assert out["faces_detected"] == 2
    assert out["frames_without_face"] == 1
    # Median must come from the two real frames only.
    assert out["per_class"]["drowsy"]["eye_blink"]["median"] == 0.9
    assert out["per_class"]["drowsy"]["closed_frame_rate"] == 1.0


def test_closed_frame_rate_uses_the_perclos_cut():
    thresholds = Thresholds()
    records = _records([
        ("alert", 0.49, 0.0, True),   # just open
        ("alert", 0.50, 0.0, True),   # exactly at the cut counts as closed
        ("alert", 0.10, 0.0, True),
        ("alert", 0.10, 0.0, True),
    ])
    out = summarise(records, thresholds)
    assert out["per_class"]["alert"]["closed_frame_rate"] == 0.25


def test_perclos_check_reports_margins_against_shipped_threshold():
    # drowsy closes 60% of frames, alert 0% -> clears 0.08 comfortably.
    records = _records(
        [("drowsy", 0.9, 0.0, True)] * 6
        + [("drowsy", 0.1, 0.0, True)] * 4
        + [("alert", 0.1, 0.0, True)] * 10
    )
    out = summarise(records)
    check = out["perclos_check"]
    assert check["drowsy_closed_rate_above_perclos_threshold"] is True
    assert check["drowsy_closed_rate_above_alert"] is True
    assert math.isclose(check["margin_vs_threshold"], 0.6 - 0.08, abs_tol=1e-6)
    assert math.isclose(check["margin_vs_alert"], 0.6, abs_tol=1e-6)


def test_threshold_that_both_classes_clear_is_not_called_a_pass():
    """
    The failure mode the n7 dataset actually exposed.

    Drowsy closes 19% of frames and alert closes 17%: drowsy is above the 0.08
    threshold, so a naive check says "clears". But alert is above it too, so the
    rule would fire on wide-awake drivers. That is a false-alarm generator, and
    the verdict must not read as a pass.
    """
    records = _records(
        [("drowsy", 0.9, 0.0, True)] * 19 + [("drowsy", 0.1, 0.0, True)] * 81
        + [("alert", 0.9, 0.0, True)] * 17 + [("alert", 0.1, 0.0, True)] * 83
    )
    check = summarise(records)["perclos_check"]
    assert check["drowsy_closed_rate_above_perclos_threshold"] is True
    assert check["alert_closed_rate_below_perclos_threshold"] is False
    assert check["threshold_discriminates"] is False, "must not read as a pass"


def test_threshold_between_the_classes_is_a_pass():
    """The ddd result: drowsy above the cut, alert below it."""
    records = _records(
        [("drowsy", 0.9, 0.0, True)] * 16 + [("drowsy", 0.1, 0.0, True)] * 84
        + [("alert", 0.9, 0.0, True)] * 2 + [("alert", 0.1, 0.0, True)] * 98
    )
    check = summarise(records)["perclos_check"]
    assert check["threshold_discriminates"] is True
    assert check["alert_closed_rate_below_perclos_threshold"] is True


def test_a_threshold_that_fails_is_reported_as_failing():
    """The report must be able to say 'no'. A check that cannot fail is decoration."""
    records = _records([("drowsy", 0.1, 0.0, True)] * 10 + [("alert", 0.1, 0.0, True)] * 10)
    check = summarise(records)["perclos_check"]
    assert check["drowsy_closed_rate_above_perclos_threshold"] is False
    assert check["margin_vs_threshold"] < 0


def test_yawn_rate_uses_strict_jaw_threshold():
    records = _records([
        ("drowsy", 0.0, 0.20, True),  # exactly at threshold is NOT a yawn (> not >=)
        ("drowsy", 0.0, 0.21, True),
    ])
    out = summarise(records)
    assert out["per_class"]["drowsy"]["yawn_frame_rate"] == 0.5


def test_summarise_survives_empty_input():
    out = summarise([])
    assert out["frames_evaluated"] == 0
    assert out["per_class"] == {}
    assert out["perclos_check"] is None


# --------------------------------------------------------------------------
# Drift guard: the copied thresholds must match the shipped engine.
# --------------------------------------------------------------------------

def test_thresholds_match_the_shipped_rule_engine_when_importable():
    """
    `Thresholds` duplicates values from challenge2/rules.py so this module can
    run without the private dataset. If that module is importable, the copies
    must agree - otherwise the duplication has silently rotted.
    """
    rules = pytest.importorskip(
        "guardian.challenge2.rules",
        reason="rules.py needs the private trip dataset / starter kit",
    )
    shipped = rules.RuleThresholds()
    ours = Thresholds()
    for theirs, mine in (
        ("perclos_drowsy", ours.perclos_drowsy),
        ("perclos_microsleep", ours.perclos_microsleep),
        ("jaw_yawn", ours.jaw_yawn),
    ):
        if hasattr(shipped, theirs):
            assert getattr(shipped, theirs) == mine, f"{theirs} drifted"
