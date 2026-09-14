"""
Tests for the drowsiness demo.

These run offline on synthetic results - no dataset, no feature CSVs. What is
under test is the measurement, not the detector: a recall number that quietly
counts the wrong frames is worse than no number, because it will be believed.
"""

from __future__ import annotations

import pytest

from guardian.challenge2.labels import CLASS_NAMES, IMPAIRED_STATES
from guardian.drowsiness.detect import (
    Transition,
    TripResult,
    _segments,
    _transitions,
    summarise,
)
from guardian.drowsiness.timeline import STATE_COLOURS, render_legend, render_timeline


def result(truth, predicted, trip_id="TEST", fps=20.0, **kw) -> TripResult:
    """Build a TripResult directly so the maths can be checked by hand."""
    truly = [s in IMPAIRED_STATES for s in truth]
    flagged = [s in IMPAIRED_STATES for s in predicted]
    defaults = dict(
        trip_id=trip_id,
        fps=fps,
        truth=tuple(truth),
        predicted=tuple(predicted),
        accuracy=0.0,
        macro_f1=0.0,
        composite=0.0,
        impaired_frames=sum(truly),
        impaired_caught=sum(t and f for t, f in zip(truly, flagged)),
        clear_frames=sum(not t for t in truly),
        false_alarms=sum((not t) and f for t, f in zip(truly, flagged)),
        false_alarms_after_warmup=0,
        clear_frames_after_warmup=0,
        transitions=(),
    )
    defaults.update(kw)
    return TripResult(**defaults)


# --------------------------------------------------------------------------
# what counts as impairment
# --------------------------------------------------------------------------

def test_distracted_is_not_counted_as_impairment():
    """
    Eyes-off-road is a different failure from falling asleep. If `distracted`
    counted, a strong distraction score could hide a weak drowsiness one - the
    exact claim this demo exists to test.
    """
    assert "distracted" not in IMPAIRED_STATES
    assert IMPAIRED_STATES == {"drowsy", "yawning", "microsleep"}


def test_every_label_has_a_colour():
    """A missing colour would render a state in the 'unknown' grey, silently."""
    for state in CLASS_NAMES:
        assert state in STATE_COLOURS


def test_flagging_a_different_impairment_still_counts_as_caught():
    """
    Calling a micro-sleep 'drowsy' is a multi-class error but not a safety
    failure: the car still knows the driver is unfit. Recall must reflect that,
    and the composite score is where the class confusion shows up.
    """
    r = result(["microsleep"] * 10, ["drowsy"] * 10)
    assert r.recall == 1.0


# --------------------------------------------------------------------------
# rates
# --------------------------------------------------------------------------

def test_recall_and_false_alarm_rate_are_hand_checkable():
    truth = ["drowsy"] * 4 + ["alert"] * 6
    pred = ["drowsy"] * 3 + ["alert"] * 6 + ["drowsy"]
    r = result(truth, pred)
    assert r.impaired_frames == 4 and r.impaired_caught == 3
    assert r.recall == pytest.approx(0.75)
    assert r.clear_frames == 6 and r.false_alarms == 1
    assert r.false_alarm_rate == pytest.approx(1 / 6)


def test_rates_are_none_rather_than_zero_when_undefined():
    """
    A trip with no impaired frames has NO recall - reporting 0.0 would read as
    'caught nothing' and drag any average down with a number that means nothing.
    """
    clear = result(["alert"] * 5, ["alert"] * 5)
    assert clear.recall is None
    impaired = result(["drowsy"] * 5, ["drowsy"] * 5)
    assert impaired.false_alarm_rate is None


def test_summary_pools_frames_instead_of_averaging_rates():
    """
    One trip with 100 impaired frames at 100% and one with 1 frame at 0% is
    99.0% recall, not the 50% an average of the two rates would give.
    """
    big = result(["drowsy"] * 100, ["drowsy"] * 100)
    tiny = result(["drowsy"], ["alert"])
    s = summarise([big, tiny])
    assert s.impaired_frames == 101 and s.impaired_caught == 100
    assert s.recall == pytest.approx(100 / 101)


def test_summarise_refuses_an_empty_run():
    with pytest.raises(ValueError):
        summarise([])


# --------------------------------------------------------------------------
# transitions
# --------------------------------------------------------------------------

def test_transition_lag_is_measured_from_the_real_change():
    truth = ["alert"] * 20 + ["drowsy"] * 20
    pred = ["alert"] * 30 + ["drowsy"] * 10          # 10 frames late at 20 FPS
    (t,) = _transitions(truth, pred, fps=20.0)
    assert t.at_s == pytest.approx(1.0)
    assert t.from_state == "alert" and t.to_state == "drowsy"
    assert t.followed_after_s == pytest.approx(0.5)


def test_a_transition_that_never_arrives_is_none_not_zero():
    truth = ["alert"] * 10 + ["drowsy"] * 10
    pred = ["alert"] * 20
    (t,) = _transitions(truth, pred, fps=20.0)
    assert t.followed_after_s is None


def test_never_followed_transitions_are_excluded_from_the_median():
    """
    Otherwise the worst possible outcome - never noticing - would be invisible
    in the headline lag, or worse, improve it.
    """
    good = result(["alert"], ["alert"], transitions=(
        Transition(at_s=1.0, from_state="alert", to_state="drowsy",
                   followed_after_s=0.5),
    ))
    missed = result(["alert"], ["alert"], transitions=(
        Transition(at_s=1.0, from_state="alert", to_state="drowsy",
                   followed_after_s=None),
    ))
    assert summarise([good, missed]).lags() == [0.5]


def test_segments_run_length_encode_in_order():
    assert _segments(["a", "a", "b", "a"]) == [("a", 0, 2), ("b", 2, 1), ("a", 3, 1)]


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def test_timeline_renders_valid_standalone_svg():
    import xml.etree.ElementTree as ET

    r = result(["alert"] * 30 + ["drowsy"] * 30, ["alert"] * 35 + ["drowsy"] * 25)
    svg = render_timeline(r)
    root = ET.fromstring(svg)                    # raises if malformed
    assert root.tag.endswith("svg")
    assert "TEST" in svg


def test_legend_is_valid_svg_and_names_every_state():
    import xml.etree.ElementTree as ET

    svg = render_legend()
    ET.fromstring(svg)
    for state in CLASS_NAMES:
        assert state in svg


def test_timeline_refuses_an_empty_trip():
    with pytest.raises(ValueError):
        render_timeline(result([], []))


def test_chart_furniture_is_themed_not_hardcoded():
    """
    Regression: the first version painted text and traces in fixed near-black,
    so on a dark page the eye-closure trace and the chart title were invisible.
    Everything that is not a state colour must resolve through a CSS variable.
    """
    r = result(["drowsy"] * 20, ["drowsy"] * 20)
    svg = render_timeline(r, eye_closure=[0.5] * 20)
    for token in ("--chart-ink", "--chart-sub", "--chart-line"):
        assert token in svg, f"{token} missing - chart will not follow the theme"
    for hardcoded in ('fill="#17191C"', 'stroke="#17191C"', 'fill="#585D65"'):
        assert hardcoded not in svg, f"{hardcoded} is not theme-aware"


def test_timeline_never_embeds_driver_imagery():
    """
    The footage is licensed academic data. The chart must stay a drawing of
    signals - if an <image> or a data URI ever appears here, publishing the
    output would republish somebody's face.
    """
    r = result(["drowsy"] * 20, ["drowsy"] * 20)
    svg = render_timeline(r)
    assert "<image" not in svg
    assert "data:image" not in svg
