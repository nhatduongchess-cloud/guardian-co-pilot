"""
Tests for the Slow Path.

These run offline against synthetic world states - no dataset, no simulator.
What is under test is the thing that can quietly go wrong: an explanation that
omits the actual cause, or states one that is not true.
"""

from __future__ import annotations

import pytest

from guardian.decision.planner import BehaviorMode, WarningLevel
from guardian.decision.safety_kernel import (
    CheckResult,
    CheckStatus,
    CommandSource,
    VerifiedCommand,
)
from guardian.explain import GuardianExplainer, SLOW_PATH_BUDGET_MS, explain
from guardian.world_model.state import (
    ContextState,
    DriverState,
    SceneState,
    TrackedObject,
    VehicleState,
    WorldModel,
)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def world(
    driver_state="alert",
    driver_reason="",
    obj=True,
    ttc=1.4,
    distance=8.0,
    surface="dry",
    night=False,
    perclos=0.0,
    eye_closure=0.0,
):
    objects = []
    if obj:
        objects.append(TrackedObject(
            object_id=1, object_class="car", distance_m=distance, lateral_m=0.2,
            closing_speed_mps=12.5, ttc_s=ttc,
        ))
    return WorldModel(
        frame_id=100,
        timestamp=5.0,
        scene=SceneState(objects=objects, min_ttc_s=ttc if obj else float("inf")),
        driver=DriverState(
            state=driver_state, explanation=driver_reason, perclos=perclos,
            eye_closure=eye_closure,
            longest_closure_s=1.6 if driver_state == "microsleep" else 0.0,
        ),
        vehicle=VehicleState(speed_kmh=50.0),
        context=ContextState(surface=surface, is_night=night),
    )


def command(
    behavior=BehaviorMode.EMERGENCY_BRAKE,
    brake=80.0,
    level=WarningLevel.CRITICAL,
    checks=(),
    source=CommandSource.PLANNER,
    hmi="Phanh!",
):
    return VerifiedCommand(
        brake_pct=brake, warning_level=level, hmi_message=hmi,
        behavior=behavior, source=source, checks=list(checks),
    )


# --------------------------------------------------------------------------
# the explanation must contain the actual cause
# --------------------------------------------------------------------------

def test_hazard_appears_in_the_headline_with_distance_and_ttc():
    ex = explain(world(), command(), lang="vi")
    assert "xe ô tô" in ex.headline
    assert "8,0" in ex.headline, "distance must be stated, in VI decimal form"
    assert "1,4" in ex.headline, "TTC must be stated"
    assert "Phanh khẩn cấp" in ex.headline
    assert "80%" in ex.headline


def test_english_uses_english_words_and_dot_decimals():
    ex = explain(world(), command(), lang="en")
    assert "a car" in ex.headline
    assert "8.0" in ex.headline
    assert "Emergency braking" in ex.headline


def test_drowsy_driver_is_named_and_credited_for_the_earlier_intervention():
    """Guardian's whole claim is that the driver moved the threshold. Say so."""
    ex = explain(world(driver_state="drowsy"), command(), lang="vi")
    assert "buồn ngủ" in ex.headline
    assert "sớm hơn" in ex.headline, "must state it acted earlier because of the driver"


def test_alert_driver_is_not_blamed_for_the_intervention():
    """The inverse: an alert driver must NOT get the 'acted earlier' clause."""
    ex = explain(world(driver_state="alert"), command(), lang="vi")
    assert "sớm hơn" not in ex.headline


def test_physiological_evidence_is_rebuilt_from_the_numbers():
    """
    The driver factor must quote the monitor's measurements, not its prose.
    """
    ex = explain(
        world(driver_state="microsleep", perclos=0.68, eye_closure=0.71),
        command(), lang="vi",
    )
    driver_text = " ".join(f.text for f in ex.factors if f.kind == "driver")
    assert "0,71" in driver_text      # eye closure, Vietnamese decimal comma
    assert "68%" in driver_text       # PERCLOS as a percentage
    assert "1,6" in driver_text       # longest closure, a real micro-sleep


def test_vietnamese_output_does_not_leak_the_english_audit_string():
    """
    Regression: DriverState.explanation is an English sentence written for the
    audit log. It used to be interpolated straight into the Vietnamese message,
    producing "Tài xế buồn ngủ - drowsy: average eye closure 0.26 ...". A
    driver-facing message must be in one language.
    """
    audit_prose = "drowsy: average eye closure 0.26 and PERCLOS 14% over 10s"
    ex = explain(
        world(driver_state="drowsy", driver_reason=audit_prose,
              perclos=0.14, eye_closure=0.26),
        command(
            checks=[CheckResult(name="Comfort & Drivability",
                                status=CheckStatus.DAMPENED,
                                message="rate limited to 15%/frame")],
        ),
        lang="vi",
    )
    spoken = ex.headline + " " + " ".join(f.text for f in ex.factors)
    for english in ("average eye closure", "over 10s", "rate limited", "frame"):
        assert english not in spoken, f"English leaked into Vietnamese: {english!r}"


def test_the_same_evidence_still_reaches_an_english_listener():
    ex = explain(
        world(driver_state="drowsy", perclos=0.14, eye_closure=0.26),
        command(), lang="en",
    )
    driver_text = " ".join(f.text for f in ex.factors if f.kind == "driver")
    assert "0.26" in driver_text and "14%" in driver_text


def test_no_hazard_and_no_intervention_reads_as_normal():
    ex = explain(
        world(obj=False, driver_state="alert"),
        command(behavior=BehaviorMode.CRUISE, brake=0.0, level=WarningLevel.NONE),
        lang="vi",
    )
    assert "Không có nguy hiểm" in ex.headline
    assert ex.severity == "none"


# --------------------------------------------------------------------------
# kernel: only report what actually changed
# --------------------------------------------------------------------------

def test_kernel_veto_is_surfaced():
    checks = [CheckResult(name="ttc_floor", status=CheckStatus.VETOED,
                          message="TTC dưới ngưỡng an toàn")]
    ex = explain(world(), command(checks=checks), lang="vi")
    assert any("chặn lệnh" in f.text for f in ex.factors)


def test_the_kernels_own_check_names_are_translated():
    """
    The kernel numbers its checks ("4. Comfort & Drivability"). The first
    version of this lookup keyed on the raw name, so every real check silently
    fell through to the unnamed phrasing. Pin the names the kernel actually
    emits, so renaming a check there fails here instead of quietly degrading
    what the driver hears.
    """
    for name in ("1. System Health", "2. Rule-based Safety",
                 "3. Collision Check", "4. Comfort & Drivability"):
        ex = explain(
            world(),
            command(checks=[CheckResult(name=name, status=CheckStatus.DAMPENED,
                                        message="engineering detail")]),
            lang="vi",
        )
        kernel_text = " ".join(f.text for f in ex.factors if f.kind == "kernel")
        assert "kiểm tra" in kernel_text, f"{name!r} did not translate"
        assert "engineering detail" not in kernel_text


def test_passing_checks_are_not_mentioned():
    """A kernel that approved everything has told the driver nothing."""
    checks = [
        CheckResult(name="a", status=CheckStatus.PASS, message="fine"),
        CheckResult(name="b", status=CheckStatus.PASS, message="also fine"),
    ]
    ex = explain(world(), command(checks=checks), lang="vi")
    assert not any(f.kind == "kernel" for f in ex.factors)


def test_failsafe_source_is_announced():
    ex = explain(world(), command(source=CommandSource.FAILSAFE, hmi="Mất cảm biến"),
                 lang="vi")
    assert any("chế độ an toàn" in f.text for f in ex.factors)


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------

def test_wet_night_is_mentioned_once_not_twice():
    ex = explain(world(surface="wet", night=True), command(), lang="vi")
    context = [f for f in ex.factors if f.kind == "context"]
    assert len(context) == 1
    assert "ướt" in context[0].text and "tối" in context[0].text


def test_dry_day_adds_no_context_noise():
    ex = explain(world(surface="dry", night=False), command(), lang="vi")
    assert not any(f.kind == "context" for f in ex.factors)


def test_absurd_ttc_is_not_quoted():
    """'TTC 214 s' is technically true and useless. Distance still appears."""
    ex = explain(world(ttc=214.0, distance=90.0), command(), lang="vi")
    assert "214" not in ex.headline
    assert "90,0" in ex.headline


# --------------------------------------------------------------------------
# determinism, latency, robustness
# --------------------------------------------------------------------------

def test_same_input_gives_identical_output():
    w, c = world(driver_state="drowsy"), command()
    assert explain(w, c, lang="vi").headline == explain(w, c, lang="vi").headline


def test_latency_is_far_inside_the_slow_path_budget():
    ex = explain(world(), command(), lang="vi")
    assert ex.within_budget
    assert ex.latency_ms < 50, f"templates should be ~instant, got {ex.latency_ms} ms"


def test_unknown_object_class_does_not_crash_the_message():
    w = world()
    w.scene.objects[0] = TrackedObject(
        object_id=2, object_class="wombat", distance_m=5.0, lateral_m=0.0,
        closing_speed_mps=3.0, ttc_s=1.0,
    )
    ex = explain(w, command(), lang="vi")
    assert "vật cản" in ex.headline, "unknown classes fall back, never blank"


def test_unsupported_language_is_rejected_loudly():
    with pytest.raises(ValueError):
        GuardianExplainer(lang="fr")


# --------------------------------------------------------------------------
# the optional narrator must never be able to break safety output
# --------------------------------------------------------------------------

def test_narrator_can_replace_the_headline():
    ex = GuardianExplainer(lang="vi", narrator=lambda e: "Phanh vì xe phía trước.").explain(
        world(), command()
    )
    assert ex.headline == "Phanh vì xe phía trước."
    assert ex.narrated is True


def test_a_crashing_narrator_falls_back_to_the_template():
    def broken(_):
        raise RuntimeError("model unavailable")

    ex = GuardianExplainer(lang="vi", narrator=broken).explain(world(), command())
    assert ex.narrated is False
    assert "Phanh khẩn cấp" in ex.headline, "must still explain itself"


def test_an_empty_narration_falls_back_rather_than_going_silent():
    ex = GuardianExplainer(lang="vi", narrator=lambda e: "   ").explain(world(), command())
    assert ex.narrated is False
    assert ex.headline.strip()
