"""
Degraded operation: what Guardian does when its inputs stop being trustworthy.

The design principle is that every component failure has a DEFINED fallback
rather than bringing the system down, and that when information is missing the
system leans toward the safest available action instead of assuming the data is
still good.

These tests pin that behaviour. They are deliberately adversarial: each one
hands the kernel broken inputs and asserts it neither acts confidently on
nonsense nor silently does nothing.
"""

from __future__ import annotations

import math

from guardian.decision.planner import BehaviorMode, CandidateAction, WarningLevel
from guardian.decision.safety_kernel import (
    FAILSAFE_BRAKE_PCT,
    MAX_BRAKE_RATE_PCT_PER_FRAME,
    MAX_STATE_AGE_S,
    CommandSource,
    SafetyKernel,
)
from guardian.explain import explain
from guardian.world_model.state import (
    ContextState,
    DriverState,
    SceneState,
    TrackedObject,
    VehicleState,
    WorldModel,
)


def make_world(timestamp=1.0, ttc=2.0, gear="D", speed=50.0, with_object=True):
    objects = []
    if with_object:
        objects.append(TrackedObject(
            object_id=1, object_class="car", distance_m=20.0, lateral_m=0.1,
            closing_speed_mps=10.0, ttc_s=ttc,
        ))
    return WorldModel(
        frame_id=int(timestamp * 20),
        timestamp=timestamp,
        scene=SceneState(objects=objects, min_ttc_s=ttc),
        driver=DriverState(state="alert"),
        vehicle=VehicleState(speed_kmh=speed, gear=gear),
        context=ContextState(),
    )


def demand(brake=100.0, behavior=BehaviorMode.EMERGENCY_BRAKE):
    """A planner asking for the strongest possible intervention."""
    return CandidateAction(
        behavior=behavior, brake_pct=brake, warning_level=WarningLevel.CRITICAL,
        hmi_message="BRAKE", rationale="test demand",
    )


# --------------------------------------------------------------------------
# Untrustworthy inputs must not produce confident action
# --------------------------------------------------------------------------

def test_nan_ttc_does_not_produce_a_confident_emergency_brake():
    """Nonsense in must not become a hard brake out."""
    kernel = SafetyKernel()
    kernel.reset()
    world = make_world(ttc=math.nan)
    command = kernel.verify(demand(), world)

    assert command.source is CommandSource.FAILSAFE
    assert command.brake_pct <= FAILSAFE_BRAKE_PCT, (
        "a kernel that hard-brakes on NaN is acting on data it knows is broken"
    )


def test_stale_perception_triggers_the_degraded_path():
    """
    A world model older than MAX_STATE_AGE_S describes a world that has moved
    on. Acting on it confidently is worse than admitting the gap.
    """
    kernel = SafetyKernel()
    kernel.reset()
    kernel.verify(demand(brake=0.0, behavior=BehaviorMode.CRUISE), make_world(timestamp=1.0))

    stale = make_world(timestamp=1.0 + MAX_STATE_AGE_S + 1.0)
    command = kernel.verify(demand(), stale)

    assert command.source is CommandSource.FAILSAFE
    assert command.brake_pct <= FAILSAFE_BRAKE_PCT


def test_degraded_mode_tells_the_driver_to_take_control():
    """Silently degrading is the failure mode that kills trust."""
    kernel = SafetyKernel()
    kernel.reset()
    command = kernel.verify(demand(), make_world(ttc=math.nan))

    assert command.hmi_message.strip(), "degraded mode must say something"
    assert "control" in command.hmi_message.lower() or "degrad" in command.hmi_message.lower()


def test_degraded_mode_still_produces_a_driver_facing_explanation():
    """
    Integration: the Slow Path must work hardest exactly when the system is
    least confident. A degraded car that cannot explain itself is the case the
    driver most needs a sentence for.
    """
    kernel = SafetyKernel()
    kernel.reset()
    world = make_world(ttc=math.nan)
    command = kernel.verify(demand(), world)

    for lang in ("vi", "en"):
        ex = explain(world, command, lang=lang)
        assert ex.headline.strip()
        assert any(f.kind == "kernel" for f in ex.factors), (
            "the driver should be told the system fell back"
        )


# --------------------------------------------------------------------------
# The kernel is the last word
# --------------------------------------------------------------------------

def test_emergency_braking_is_deliberately_not_rate_limited():
    """
    Rate limiting exists for comfort, and an emergency stop is the one case
    where comfort has to lose. Ramping to full brake over several frames would
    add metres to the stopping distance exactly when there are none to spare,
    so the emergency path is exempt by design - not by oversight.
    """
    kernel = SafetyKernel()
    kernel.reset()
    command = kernel.verify(demand(brake=100.0), make_world(timestamp=1.0, ttc=1.0))
    assert command.brake_pct == 100.0


def test_non_emergency_assist_braking_is_rate_limited():
    """
    Outside an emergency the brake ramps instead of stepping, so an assist
    intervention never snatches at the car.
    """
    kernel = SafetyKernel()
    kernel.reset()
    world = make_world(timestamp=1.0, ttc=6.0)  # well outside the emergency window
    command = kernel.verify(
        demand(brake=100.0, behavior=BehaviorMode.ASSIST_BRAKE), world
    )
    assert command.brake_pct <= MAX_BRAKE_RATE_PCT_PER_FRAME + 1e-6


def test_no_autonomous_braking_in_reverse():
    """A hard rule: Guardian does not brake the car for the driver in reverse."""
    kernel = SafetyKernel()
    kernel.reset()
    command = kernel.verify(demand(brake=80.0), make_world(gear="R", ttc=1.0))
    assert command.brake_pct == 0.0


# --------------------------------------------------------------------------
# Degraded does not mean dead
# --------------------------------------------------------------------------

def test_the_kernel_always_returns_a_usable_command():
    """
    Whatever it is handed, `verify` returns something the vehicle can act on.
    An exception here would take the assist offline mid-drive.
    """
    kernel = SafetyKernel()
    broken = [
        make_world(ttc=math.nan),
        make_world(ttc=math.inf),
        make_world(ttc=-5.0),
        make_world(with_object=False),
        make_world(speed=-10.0),
    ]
    for world in broken:
        kernel.reset()
        command = kernel.verify(demand(), world)
        assert 0.0 <= command.brake_pct <= 100.0
        assert command.hmi_message is not None
        assert command.audit_trail().strip(), "every decision stays auditable"


def test_every_degraded_decision_is_auditable():
    kernel = SafetyKernel()
    kernel.reset()
    command = kernel.verify(demand(), make_world(ttc=math.nan))
    trail = command.audit_trail()
    assert "failsafe" in trail.lower() or "health" in trail.lower()
