"""
safety_kernel.py
================

GUARDIAN CO-PILOT — V4 SAFETY KERNEL

The last gate before anything reaches the vehicle:

    Candidate Action ──► SAFETY KERNEL ──► Verified Command ──► V2 Cockpit HMI

The Guardian proposal states the rule this file implements: *planning proposes
an action, but only the Safety Kernel — deterministic, no AI — may issue a
command to the vehicle.* Everything here is arithmetic and explicit rules. No
model, no learned weights, no randomness. It must stay small enough to read in
one sitting and argue about line by line, because it is the component that can
brake a car.

THE FIVE CHECKS (in the order the architecture specifies)
---------------------------------------------------------
    1. System Health Check    are the inputs trustworthy at all?
    2. Rule-based Safety Check hard invariants that hold regardless of context
    3. Collision Check (TTC)   independent physics — may ESCALATE the planner
    4. Comfort & Drivability   rate-limit the command, unless it is an emergency
    5. Fail-safe Strategy      what to do when we cannot trust the picture

Health runs first on purpose: if the driver camera dropped out or perception is
stale, nothing computed downstream deserves to be believed, so there is no
point checking collisions against numbers we do not trust.

THREE POWERS OVER THE PLANNER
-----------------------------
    ESCALATE  physics demands harder braking than proposed -> kernel raises it
    DAMPEN    the command is jerky -> rate-limit it, EXCEPT in an emergency
    VETO      inputs are invalid -> discard the plan, apply the fail-safe

The kernel may always make the vehicle SAFER than the planner asked. It may
only make it less aggressive for comfort, and never during an emergency.

AUDITABILITY
------------
Every command carries the list of checks that ran, what each decided, and why.
That trail is what makes an intervention explainable after the fact — to the
driver, to an engineer, and to a judge asking "why did it brake there?".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from guardian.decision.planner import (
    BehaviorMode,
    CandidateAction,
    WarningLevel,
)
from guardian.world_model.state import EMERGENCY_DECEL_MPS2, WorldModel


# ===========================================================================
# SECTION 1 — LIMITS AND VOCABULARY
# ===========================================================================

#: Maximum change in brake percentage per frame under normal conditions.
#: At 20 FPS, 15 %/frame reaches full braking in ~0.35 s — firm but not a jolt.
MAX_BRAKE_RATE_PCT_PER_FRAME = 15.0

#: Below this TTC the situation is an emergency: comfort limits are suspended.
EMERGENCY_TTC_S = 1.5

#: A world model older than this is stale and must not be acted on.
MAX_STATE_AGE_S = 0.5

#: Brake level applied by the fail-safe when the picture cannot be trusted but
#: a hazard was last seen. Firm enough to matter, gentle enough not to be a
#: hazard itself if the reading was wrong.
FAILSAFE_BRAKE_PCT = 30.0


class CheckStatus(str, Enum):
    """Outcome of one kernel check."""

    PASS = "pass"          # nothing to change
    ESCALATED = "escalated"  # kernel demanded MORE braking than proposed
    DAMPENED = "dampened"    # kernel reduced the command (comfort/jerk)
    VETOED = "vetoed"        # kernel discarded the plan entirely


class CommandSource(str, Enum):
    """Who ultimately decided the command that leaves this file."""

    PLANNER = "planner"      # planner's proposal survived unchanged
    KERNEL = "kernel"        # kernel modified it
    FAILSAFE = "failsafe"    # kernel overrode it completely


@dataclass
class CheckResult:
    """What one check decided, and why."""

    name: str
    status: CheckStatus
    message: str
    brake_before: float = 0.0
    brake_after: float = 0.0

    def describe(self) -> str:
        change = ""
        if self.brake_before != self.brake_after:
            change = f" [{self.brake_before:.0f}% -> {self.brake_after:.0f}%]"
        return f"{self.name:26s} {self.status.value:10s}{change}  {self.message}"


@dataclass
class VerifiedCommand:
    """
    The only object in Guardian authorised to reach the vehicle.

    Mirrors the "Verified Command" box of the architecture: a longitudinal
    command, an HMI command, and the audit trail that justifies both.
    """

    brake_pct: float
    warning_level: WarningLevel
    hmi_message: str
    behavior: BehaviorMode
    source: CommandSource
    checks: list[CheckResult] = field(default_factory=list)
    explanation: str = ""

    @property
    def is_intervention(self) -> bool:
        return self.brake_pct > 0.0

    def audit_trail(self) -> str:
        """Full, human-readable record of how this command was reached."""
        lines = [f"COMMAND  brake {self.brake_pct:5.1f}%  "
                 f"{self.behavior.value}  [{self.warning_level.value}]  "
                 f"source={self.source.value}"]
        lines += ["  " + c.describe() for c in self.checks]
        if self.explanation:
            lines.append(f"  WHY: {self.explanation}")
        return "\n".join(lines)

    def to_vss(self) -> dict[str, float | str]:
        """Signals the kernel writes to the CarSky KUKSA bus."""
        return {
            # The signal the AEB test bench scores.
            "Vehicle.Chassis.Brake.PedalPosition": round(self.brake_pct, 1),
            "Vehicle.Guardian.Behavior": self.behavior.value,
            "Vehicle.Guardian.WarningLevel": self.warning_level.value,
        }


# ===========================================================================
# SECTION 2 — THE KERNEL
# ===========================================================================


class SafetyKernel:
    """Deterministic verifier. Holds only the minimum state needed for rate limits."""

    def __init__(
        self,
        max_brake_rate: float = MAX_BRAKE_RATE_PCT_PER_FRAME,
        emergency_ttc: float = EMERGENCY_TTC_S,
    ) -> None:
        self.max_brake_rate = max_brake_rate
        self.emergency_ttc = emergency_ttc
        self._last_brake_pct: float = 0.0
        self._last_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Clear rate-limit history (call between trips)."""
        self._last_brake_pct = 0.0
        self._last_timestamp = None

    # -- main entry point ----------------------------------------------------

    def verify(self, candidate: CandidateAction, world: WorldModel) -> VerifiedCommand:
        """Run the five checks and return the command that may reach the vehicle."""
        checks: list[CheckResult] = []
        brake = float(candidate.brake_pct)
        source = CommandSource.PLANNER

        # --- 1. System Health ------------------------------------------------
        health = self._check_system_health(world)
        checks.append(health)
        if health.status is CheckStatus.VETOED:
            command = self._failsafe(world, checks)
            self._remember(command.brake_pct, world.timestamp)
            return command

        # --- 2. Rule-based Safety --------------------------------------------
        # Hard rules produce a CEILING, not merely a value. A later check may
        # never exceed it: an invariant that a downstream stage can overrule is
        # not an invariant. (This bit us in testing — the collision check was
        # happily re-applying brake in reverse gear after the rule zeroed it.)
        rules, ceiling = self._check_rules(brake, world)
        checks.append(rules)
        if rules.brake_after != brake:
            brake = rules.brake_after
            source = CommandSource.KERNEL

        # --- 3. Collision Check (independent physics) ------------------------
        collision = self._check_collision(brake, world)
        checks.append(collision)
        if collision.brake_after != brake:
            brake = collision.brake_after
            source = CommandSource.KERNEL

        # Enforce the ceiling from the hard rules.
        if brake > ceiling:
            checks.append(CheckResult(
                "2b. Rule Ceiling", CheckStatus.DAMPENED,
                f"hard rule caps braking at {ceiling:.0f}% - physics wanted more, "
                f"so the driver is asked to take over",
                brake, ceiling,
            ))
            brake = ceiling
            source = CommandSource.KERNEL

        # --- 4. Comfort & Drivability ----------------------------------------
        comfort = self._check_comfort(brake, world)
        checks.append(comfort)
        if comfort.brake_after != brake:
            brake = comfort.brake_after
            source = CommandSource.KERNEL

        # --- assemble ---------------------------------------------------------
        behavior, warning, message = self._final_presentation(
            candidate, brake, world
        )
        command = VerifiedCommand(
            brake_pct=brake,
            warning_level=warning,
            hmi_message=message,
            behavior=behavior,
            source=source,
            checks=checks,
            explanation=self._explain(candidate, brake, world),
        )
        self._remember(brake, world.timestamp)
        return command

    # -- check 1: system health ---------------------------------------------

    def _check_system_health(self, world: WorldModel) -> CheckResult:
        """
        Are the inputs trustworthy? Anything NaN, stale, or physically absurd
        means we are reasoning about a world that may not exist.
        """
        problems: list[str] = []

        if not math.isfinite(world.vehicle.speed_kmh):
            problems.append("speed is not a number")
        if world.vehicle.speed_kmh < -1.0 or world.vehicle.speed_kmh > 400.0:
            problems.append(f"speed out of range ({world.vehicle.speed_kmh:.0f} km/h)")

        # The scene's TTC is what every threshold below is compared against, and
        # NaN loses every comparison SILENTLY: `nan < EMERGENCY_TTC_S` is False,
        # so a broken perception frame would be indistinguishable from a clear
        # road. Infinity is not a problem - it is the honest way to say "nothing
        # ahead" - but NaN and negative time are both impossible worlds.
        ttc = world.scene.min_ttc_s
        if math.isnan(ttc):
            problems.append("scene TTC is not a number (perception produced NaN)")
        elif ttc < 0.0:
            problems.append(f"scene TTC is negative ({ttc:.2f}s)")

        for obj in world.scene.objects:
            if math.isnan(obj.distance_m) or obj.distance_m < 0.0:
                problems.append(
                    f"object {obj.object_id} has an impossible distance "
                    f"({obj.distance_m})"
                )
                break
            if math.isnan(obj.ttc_s):
                problems.append(f"object {obj.object_id} has a NaN TTC")
                break

        # A driver state we do not recognise means the DMS pipeline is broken;
        # we must not silently treat an unknown driver as an attentive one.
        from guardian.world_model.state import DRIVER_REACTION_TIME_S
        if world.driver.state not in DRIVER_REACTION_TIME_S:
            problems.append(f"unknown driver state '{world.driver.state}'")

        if self._last_timestamp is not None:
            age = world.timestamp - self._last_timestamp
            if age > MAX_STATE_AGE_S:
                problems.append(f"state gap of {age:.2f}s (stale perception)")

        if problems:
            return CheckResult(
                "1. System Health", CheckStatus.VETOED,
                "; ".join(problems),
            )
        return CheckResult("1. System Health", CheckStatus.PASS, "inputs valid")

    # -- check 2: hard rules -------------------------------------------------

    def _check_rules(
        self, brake: float, world: WorldModel
    ) -> tuple[CheckResult, float]:
        """
        Invariants that hold no matter what the planner believes.

        Returns:
            (result, ceiling) — the adjusted command AND the hard upper bound
            that no later check may exceed. Returning the ceiling separately is
            what makes these rules genuinely inviolable.
        """
        original = brake
        ceiling = 100.0
        notes: list[str] = []

        clamped = max(0.0, min(100.0, brake))
        if clamped != brake:
            notes.append(f"brake clamped to [0,100] from {brake:.0f}")
            brake = clamped

        # Guardian does not autonomously brake a parked or reversing vehicle:
        # low-speed manoeuvring belongs to the driver, and a surprise
        # intervention there is itself a hazard.
        if world.vehicle.gear in ("P", "R"):
            ceiling = 0.0
            if brake > 0:
                notes.append(f"no autonomous braking in gear {world.vehicle.gear}")
                brake = 0.0

        # Below walking pace an intervention achieves little and is a nuisance
        # during parking.
        if world.vehicle.speed_kmh < 5.0:
            ceiling = 0.0
            if brake > 0:
                notes.append(f"speed {world.vehicle.speed_kmh:.1f} km/h below "
                             f"intervention floor")
                brake = 0.0

        if brake == original and ceiling >= 100.0:
            return (
                CheckResult("2. Rule-based Safety", CheckStatus.PASS,
                            "no invariant violated", original, brake),
                ceiling,
            )
        return (
            CheckResult("2. Rule-based Safety", CheckStatus.DAMPENED,
                        "; ".join(notes) or f"ceiling set to {ceiling:.0f}%",
                        original, brake),
            ceiling,
        )

    # -- check 3: collision physics ------------------------------------------

    def _check_collision(self, brake: float, world: WorldModel) -> CheckResult:
        """
        Recompute, independently of the planner, the braking actually required.

        This is the kernel's veto power used in the safe direction: if physics
        says we need more brake than was proposed, we raise it. The kernel
        never lowers braking here — only the comfort check may do that, and
        only outside an emergency.
        """
        target = world.scene.critical_object
        if target is None or not math.isfinite(world.scene.min_ttc_s):
            return CheckResult("3. Collision Check", CheckStatus.PASS,
                               "no obstacle in the collision cone", brake, brake)

        required = self._required_brake_pct(world)
        if required > brake:
            return CheckResult(
                "3. Collision Check", CheckStatus.ESCALATED,
                (f"TTC {world.scene.min_ttc_s:.2f}s at "
                 f"{target.closing_speed_mps:.1f} m/s needs {required:.0f}%"),
                brake, required,
            )
        return CheckResult(
            "3. Collision Check", CheckStatus.PASS,
            f"TTC {world.scene.min_ttc_s:.2f}s, {brake:.0f}% is sufficient",
            brake, brake,
        )

    @staticmethod
    def _required_brake_pct(world: WorldModel) -> float:
        """
        Brake percentage needed to stop within the distance available.

        Solves  d = v*t_delay + v^2 / (2a)  for the deceleration a, then
        expresses it as a fraction of what the surface can deliver.
        """
        target = world.scene.critical_object
        if target is None:
            return 0.0

        closing = max(target.closing_speed_mps, 0.1)
        available = max(target.distance_m - closing * 0.2, 0.1)  # minus latency
        needed_decel = closing ** 2 / (2.0 * available)

        max_decel = world.context.available_decel(EMERGENCY_DECEL_MPS2)
        if max_decel <= 0:
            return 100.0
        return float(max(0.0, min(100.0, 100.0 * needed_decel / max_decel)))

    # -- check 4: comfort ----------------------------------------------------

    def _check_comfort(self, brake: float, world: WorldModel) -> CheckResult:
        """
        Rate-limit the pedal so commands are smooth — but never at the cost of
        an emergency stop.

        Jerk matters: a command jumping 0 -> 100% in one frame is unpleasant
        and can itself destabilise the vehicle. Outside an emergency we ramp.
        Inside one, comfort is irrelevant and this check stands down.
        """
        is_emergency = (
            math.isfinite(world.scene.min_ttc_s)
            and world.scene.min_ttc_s <= self.emergency_ttc
        )
        if is_emergency:
            return CheckResult(
                "4. Comfort & Drivability", CheckStatus.PASS,
                f"emergency (TTC {world.scene.min_ttc_s:.2f}s) - limits suspended",
                brake, brake,
            )

        delta = brake - self._last_brake_pct
        if abs(delta) <= self.max_brake_rate:
            return CheckResult("4. Comfort & Drivability", CheckStatus.PASS,
                               "within rate limit", brake, brake)

        limited = self._last_brake_pct + math.copysign(self.max_brake_rate, delta)
        limited = max(0.0, min(100.0, limited))
        return CheckResult(
            "4. Comfort & Drivability", CheckStatus.DAMPENED,
            f"rate limited to {self.max_brake_rate:.0f}%/frame",
            brake, limited,
        )

    # -- check 5: fail-safe --------------------------------------------------

    def _failsafe(self, world: WorldModel, checks: list[CheckResult]) -> VerifiedCommand:
        """
        What to do when the picture cannot be trusted.

        Guardian degrades gracefully rather than guessing. If a hazard was
        visible before the inputs went bad, apply moderate braking and hand
        control back to the driver with a clear message. Otherwise take no
        longitudinal action and simply tell the driver the assist is degraded.
        Silently continuing to act on untrusted data would be the dangerous
        option.
        """
        hazard_seen = math.isfinite(world.scene.min_ttc_s)
        brake = FAILSAFE_BRAKE_PCT if hazard_seen else 0.0
        message = (
            "Assist degraded - take control (braking gently)"
            if hazard_seen else
            "Driver assist degraded - take control"
        )
        checks.append(CheckResult(
            "5. Fail-safe Strategy", CheckStatus.VETOED,
            f"plan discarded; {'moderate braking' if hazard_seen else 'no action'}"
            f" and handover requested",
            0.0, brake,
        ))
        return VerifiedCommand(
            brake_pct=brake,
            warning_level=WarningLevel.CRITICAL,
            hmi_message=message,
            behavior=BehaviorMode.WARN,
            source=CommandSource.FAILSAFE,
            checks=checks,
            explanation="inputs failed the health check; Guardian will not act "
                        "on data it cannot trust",
        )

    # -- presentation --------------------------------------------------------

    def _final_presentation(
        self, candidate: CandidateAction, brake: float, world: WorldModel
    ) -> tuple[BehaviorMode, WarningLevel, str]:
        """Re-derive behaviour and HMI wording from the FINAL brake level."""
        if brake >= 80.0:
            return (BehaviorMode.EMERGENCY_BRAKE, WarningLevel.CRITICAL,
                    "Emergency braking")
        if brake > 0.0:
            return (BehaviorMode.ASSIST_BRAKE, WarningLevel.CRITICAL,
                    "Braking assist active")
        # No braking: keep whatever the planner wanted to say.
        return (candidate.behavior, candidate.warning_level, candidate.hmi_message)

    @staticmethod
    def _explain(candidate: CandidateAction, brake: float, world: WorldModel) -> str:
        """One sentence a driver could understand."""
        driver = world.driver
        if brake <= 0.0:
            if math.isfinite(world.scene.min_ttc_s):
                return (f"obstacle {world.scene.min_ttc_s:.1f}s ahead; "
                        f"{driver.state} driver has {world.safety_margin_s:+.1f}s "
                        f"of margin and is expected to brake")
            return f"road clear; driver monitored as {driver.state}"

        return (f"braking {brake:.0f}%: obstacle at TTC "
                f"{world.scene.min_ttc_s:.1f}s, this {driver.state} driver needs "
                f"{world.time_needed_s():.1f}s to stop but has "
                f"{world.scene.min_ttc_s:.1f}s")

    def _remember(self, brake: float, timestamp: float) -> None:
        self._last_brake_pct = brake
        self._last_timestamp = timestamp


# ===========================================================================
# SECTION 3 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    from guardian.decision.planner import PlanningEngine
    from guardian.world_model.state import (
        ContextState, DriverState, SceneState, TrackedObject, VehicleState,
    )

    def frame(state: str, speed_kmh: float, ttc: float, closing_mps: float,
              gear: str = "D", timestamp: float = 0.0) -> WorldModel:
        target = TrackedObject(
            object_id=1, object_class="vehicle",
            distance_m=closing_mps * ttc, lateral_m=0.0,
            closing_speed_mps=closing_mps, ttc_s=ttc,
        )
        return WorldModel(
            frame_id=0, timestamp=timestamp,
            scene=SceneState(objects=[target], min_ttc_s=ttc),
            driver=DriverState(state=state),
            vehicle=VehicleState(speed_kmh=speed_kmh, gear=gear),
            context=ContextState(),
        )

    planner = PlanningEngine()

    print("A. NORMAL VERIFICATION - drowsy driver, TTC 2.5s at 50 km/h")
    print("=" * 88)
    kernel = SafetyKernel()
    world = frame("drowsy", 50.0, 2.5, 50.0 / 3.6)
    cmd = kernel.verify(planner.plan(world), world)
    print(cmd.audit_trail())

    print("\n\nB. KERNEL ESCALATES A TOO-GENTLE PLAN")
    print("=" * 88)
    kernel = SafetyKernel()
    world = frame("alert", 60.0, 1.2, 60.0 / 3.6)
    weak = planner.plan(world)
    weak.brake_pct = 20.0          # pretend the planner under-reacted
    weak.rationale = "(artificially weakened to demonstrate escalation)"
    cmd = kernel.verify(weak, world)
    print(cmd.audit_trail())

    print("\n\nC. COMFORT RATE-LIMIT (non-emergency) vs EMERGENCY OVERRIDE")
    print("=" * 88)
    kernel = SafetyKernel()
    calm = frame("drowsy", 50.0, 4.0, 50.0 / 3.6, timestamp=0.00)
    cmd = kernel.verify(planner.plan(calm), calm)
    print("non-emergency, from 0%:")
    print("  " + [c for c in cmd.checks if c.name.startswith("4")][0].describe())
    print(f"  -> issued {cmd.brake_pct:.0f}% (planner wanted "
          f"{planner.plan(calm).brake_pct:.0f}%)")

    kernel = SafetyKernel()
    urgent = frame("drowsy", 50.0, 1.0, 50.0 / 3.6, timestamp=0.00)
    cmd = kernel.verify(planner.plan(urgent), urgent)
    print("emergency (TTC 1.0s), from 0%:")
    print("  " + [c for c in cmd.checks if c.name.startswith("4")][0].describe())
    print(f"  -> issued {cmd.brake_pct:.0f}% immediately, no ramp")

    print("\n\nD. HARD RULES - no autonomous braking in reverse")
    print("=" * 88)
    kernel = SafetyKernel()
    world = frame("drowsy", 10.0, 1.5, 10.0 / 3.6, gear="R")
    cmd = kernel.verify(planner.plan(world), world)
    print(cmd.audit_trail())

    print("\n\nE. FAIL-SAFE - driver monitoring returns garbage")
    print("=" * 88)
    kernel = SafetyKernel()
    world = frame("alert", 50.0, 2.0, 50.0 / 3.6)
    candidate = planner.plan(world)
    world.driver.state = "sensor_error"     # DMS pipeline broke
    cmd = kernel.verify(candidate, world)
    print(cmd.audit_trail())

    print("\n\nF. VSS OUTPUT (what the kernel publishes to CarSky)")
    print("=" * 88)
    for path, value in cmd.to_vss().items():
        print(f"  {path:46s} {value}")
