"""
state.py
========

GUARDIAN CO-PILOT — V4 WORLD MODEL (Unified State)

The keystone of the Guardian architecture. Four streams flow in, one object
flows out:

    Scene   (objects, lane, intent)          <- V3 Perception
    Driver  (DMS, fatigue, attention)        <- V3 Driver Monitoring
    Vehicle (speed, brake, accel)            <- CAN / kinematics
    Context (weather, surface, speed limit)  <- map / trip metadata
        |
        v
    WorldModel  -->  Planning Engine  -->  Safety Kernel  -->  Cockpit HMI

Everything downstream reads THIS object and nothing else. That is what keeps
the Safety Kernel from reaching into perception internals, and it is what makes
the same kernel runnable offline (on the dataset) and live (on CarSky).

THE IDEA THAT MAKES PERSONALISATION A SAFETY INPUT
--------------------------------------------------
Guardian's pitch says a novice or impaired driver needs earlier warnings. It is
easy to state and easy to fake with hand-picked thresholds. We ground it in
physics instead: each driver state maps to a REACTION TIME, and reaction time
enters the collision equation directly.

    total time needed = reaction_time + braking_time
    braking_time      = speed / deceleration

A microsleeping driver is not "a bit slower" — they are ~4 s away from touching
the pedal, which at 90 km/h is 100 m of road. That is not a tuning constant, it
is the reason the intervention threshold must move. The Safety Kernel consumes
`driver.reaction_time_s` and needs no special case for "drowsy".

Reaction times below come from the driver-monitoring / human-factors range
(alert drivers ~0.7-1.5 s brake reaction; impairment and distraction push this
up several-fold). They are documented estimates, not measurements from this
dataset, and are declared as such in the write-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ===========================================================================
# SECTION 1 — DRIVER REACTION MODEL
# ===========================================================================
# The single place where "who is driving" becomes "how much time do we have".
# ---------------------------------------------------------------------------

#: Estimated brake-reaction time per driver state, in seconds.
DRIVER_REACTION_TIME_S: dict[str, float] = {
    "alert": 1.0,        # attentive driver, eyes on road
    "yawning": 1.5,      # brief lapse, still fundamentally engaged
    "distracted": 1.8,   # eyes/mind off road (phone) — must re-orient first
    "drowsy": 2.5,       # degraded vigilance, slow motor response
    "microsleep": 4.0,   # effectively absent; assume no timely intervention
}

#: How much of the driver's own capability we can count on (0..1).
#: Used by the HMI to decide how firmly to intervene, not by the kernel.
DRIVER_CAPABILITY: dict[str, float] = {
    "alert": 1.00,
    "yawning": 0.80,
    "distracted": 0.55,
    "drowsy": 0.40,
    "microsleep": 0.05,
}

#: Comfortable vs emergency deceleration on dry tarmac (m/s^2).
COMFORT_DECEL_MPS2 = 3.0
EMERGENCY_DECEL_MPS2 = 8.0

#: Friction multipliers applied to deceleration by road surface.
SURFACE_FRICTION: dict[str, float] = {
    "dry": 1.00,
    "wet": 0.70,
    "snow": 0.35,
    "ice": 0.20,
}


# ===========================================================================
# SECTION 2 — THE FOUR SUB-STATES
# ===========================================================================


@dataclass
class TrackedObject:
    """One obstacle as the Safety Kernel needs to see it."""

    object_id: int
    object_class: str          # walker | bike | vehicle
    distance_m: float          # longitudinal, ego-centre to target-centre
    lateral_m: float           # +right / -left of the ego axis
    closing_speed_mps: float   # positive = approaching
    ttc_s: float               # inf when not on a collision course
    confidence: float = 1.0

    @property
    def in_path(self) -> bool:
        """Inside the collision cone recovered from the ground truth."""
        return abs(self.lateral_m) <= 2.5


@dataclass
class SceneState:
    """V3 Perception output: what the car can see."""

    objects: list[TrackedObject] = field(default_factory=list)
    min_ttc_s: float = math.inf
    lane_id: Optional[int] = None

    @property
    def critical_object(self) -> Optional[TrackedObject]:
        """The in-path object with the smallest TTC — what we would hit first."""
        candidates = [o for o in self.objects if o.in_path and math.isfinite(o.ttc_s)]
        return min(candidates, key=lambda o: o.ttc_s) if candidates else None


@dataclass
class DriverState:
    """V3 Driver Monitoring output: who is driving and how fit they are."""

    state: str = "alert"           # one of the five DMS classes
    perclos: float = 0.0           # fraction of the window with eyes closed
    eye_closure: float = 0.0       # mean blendshape value
    longest_closure_s: float = 0.0
    phone_rate: float = 0.0        # fraction of window with a phone visible
    explanation: str = ""          # the plain-language reason from the rules

    @property
    def reaction_time_s(self) -> float:
        """Estimated time before this driver could begin braking."""
        return DRIVER_REACTION_TIME_S.get(self.state, 1.5)

    @property
    def capability(self) -> float:
        """How much of the driving task this driver can still carry (0..1)."""
        return DRIVER_CAPABILITY.get(self.state, 0.5)

    @property
    def fatigue_level(self) -> float:
        """0-100 fatigue indicator for VSS / the cockpit gauge."""
        # PERCLOS is the physiological driver of fatigue; the sustained-closure
        # term makes a long single closure count for more than frequent blinks.
        return float(min(100.0, 100.0 * self.perclos + 20.0 * self.longest_closure_s))

    @property
    def distraction_level(self) -> float:
        """0-100 distraction indicator, driven by phone presence."""
        return float(min(100.0, 100.0 * self.phone_rate))

    @property
    def is_eyes_on_road(self) -> bool:
        return self.state in ("alert", "yawning") and self.perclos < 0.3


@dataclass
class VehicleState:
    """Ego kinematics, straight from CAN / the trip log."""

    speed_kmh: float = 0.0
    longitudinal_accel: float = 0.0
    lateral_accel: float = 0.0
    brake_pct: float = 0.0
    gear: str = "D"

    @property
    def speed_mps(self) -> float:
        return self.speed_kmh / 3.6

    def braking_distance_m(self, decel_mps2: float) -> float:
        """v^2 / 2a — distance needed to stop at a given deceleration."""
        if decel_mps2 <= 0:
            return math.inf
        return self.speed_mps ** 2 / (2.0 * decel_mps2)


@dataclass
class ContextState:
    """Environment: what the road is like right now."""

    speed_limit_kmh: float = 50.0
    surface: str = "dry"
    weather: str = "clear"
    is_night: bool = False

    @property
    def friction(self) -> float:
        return SURFACE_FRICTION.get(self.surface, 1.0)

    def available_decel(self, base_decel: float) -> float:
        """Deceleration actually achievable on this surface."""
        return base_decel * self.friction


# ===========================================================================
# SECTION 3 — THE UNIFIED STATE
# ===========================================================================


@dataclass
class WorldModel:
    """
    One frame of Guardian's understanding of the world.

    Built either offline (from a dataset trip) or live (from CarSky bus
    signals). Downstream stages depend on this class alone.
    """

    frame_id: int
    timestamp: float
    scene: SceneState = field(default_factory=SceneState)
    driver: DriverState = field(default_factory=DriverState)
    vehicle: VehicleState = field(default_factory=VehicleState)
    context: ContextState = field(default_factory=ContextState)

    # -- derived safety quantities ------------------------------------------

    def time_needed_s(self, emergency: bool = True) -> float:
        """
        Total time this driver+vehicle needs to avoid a collision:

            reaction time (driver-dependent)  +  braking time (closing speed)

        IMPORTANT — we brake against the CLOSING speed, not our own speed.
        Avoiding a collision means nulling the *relative* velocity: following a
        car at 90 km/h that is also doing 90 km/h needs no braking at all.
        Using ego speed here would declare an emergency on every motorway and
        make the model unable to tell driver states apart. Where no closing
        speed is known we fall back to ego speed, which is the correct
        worst case: a stationary obstacle.
        """
        base = EMERGENCY_DECEL_MPS2 if emergency else COMFORT_DECEL_MPS2
        decel = self.context.available_decel(base)
        if decel <= 0:
            return math.inf

        target = self.scene.critical_object
        closing = (
            target.closing_speed_mps
            if target is not None and target.closing_speed_mps > 0
            else self.vehicle.speed_mps
        )
        braking_time = closing / decel
        return self.driver.reaction_time_s + braking_time

    @property
    def safety_margin_s(self) -> float:
        """
        TTC minus the time we need. Negative means we are already too late for
        the driver to handle it alone — the point where Guardian must act.
        """
        if not math.isfinite(self.scene.min_ttc_s):
            return math.inf
        return self.scene.min_ttc_s - self.time_needed_s(emergency=True)

    @property
    def risk_score(self) -> float:
        """
        0-100 fused risk: how close we are to trouble, amplified by how little
        the driver can contribute. Used by the cockpit gauge and Challenge 3.
        """
        # Hazard term: 0 when the margin is comfortable, 100 when it is gone.
        margin = self.safety_margin_s
        if math.isfinite(margin):
            hazard = float(max(0.0, min(100.0, 100.0 * (1.0 - margin / 3.0))))
        else:
            hazard = 0.0

        # Driver term: an impaired driver is a standing risk even on empty road.
        driver_risk = 100.0 * (1.0 - self.driver.capability)

        # The hazard dominates, but an impaired driver raises the floor.
        return float(min(100.0, max(hazard, 0.5 * driver_risk)))

    # -- interoperability ----------------------------------------------------

    def to_vss(self) -> dict[str, float | bool]:
        """
        Project the world model onto COVESA VSS signal paths.

        This is the bridge to CarSky: publishing this dict to the KUKSA
        databroker makes Guardian's understanding visible to every other node
        on the virtual vehicle. Paths follow the COVESA tree; verify them
        against the `vss.json` artifact registered on the platform.
        """
        signals: dict[str, float | bool] = {
            "Vehicle.Speed": round(self.vehicle.speed_kmh, 2),
            "Vehicle.Driver.FatigueLevel": round(self.driver.fatigue_level, 1),
            "Vehicle.Driver.DistractionLevel": round(self.driver.distraction_level, 1),
            "Vehicle.Driver.IsEyesOnRoad": self.driver.is_eyes_on_road,
            "Vehicle.Driver.AttentiveProbability": round(100.0 * self.driver.capability, 1),
        }
        # Guardian's own branch for quantities VSS has no standard path for.
        signals["Vehicle.Guardian.MinTTC"] = (
            round(self.scene.min_ttc_s, 3) if math.isfinite(self.scene.min_ttc_s) else -1.0
        )
        signals["Vehicle.Guardian.SafetyMargin"] = (
            round(self.safety_margin_s, 3) if math.isfinite(self.safety_margin_s) else -1.0
        )
        signals["Vehicle.Guardian.RiskScore"] = round(self.risk_score, 1)
        signals["Vehicle.Guardian.DriverState"] = self.driver.state
        return signals

    def summary(self) -> str:
        """One human-readable line, for logs and the HUD."""
        ttc = ("clear" if not math.isfinite(self.scene.min_ttc_s)
               else f"{self.scene.min_ttc_s:.1f}s")
        return (f"t={self.timestamp:6.2f}s  {self.vehicle.speed_kmh:5.1f}km/h  "
                f"TTC {ttc:>6s}  need {self.time_needed_s():.1f}s  "
                f"driver {self.driver.state} (react {self.driver.reaction_time_s:.1f}s)  "
                f"risk {self.risk_score:3.0f}")


# ===========================================================================
# SECTION 4 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    def scenario(state: str, speed_kmh: float, ttc: float, closing_mps: float,
                 surface: str = "dry") -> WorldModel:
        """Build one hypothetical frame: a stationary-ish obstacle ahead."""
        target = TrackedObject(
            object_id=1, object_class="vehicle",
            distance_m=closing_mps * ttc, lateral_m=0.0,
            closing_speed_mps=closing_mps, ttc_s=ttc,
        )
        return WorldModel(
            frame_id=0, timestamp=0.0,
            scene=SceneState(objects=[target], min_ttc_s=ttc),
            driver=DriverState(state=state),
            vehicle=VehicleState(speed_kmh=speed_kmh),
            context=ContextState(surface=surface),
        )

    print("Why driver state changes the intervention point")
    print("=" * 78)
    print("Urban case: 50 km/h, stationary obstacle, TTC 3.0 s, dry road.\n")
    print(f"{'driver':12s}{'react':>7s}{'brake':>8s}{'need':>8s}{'margin':>9s}"
          f"{'risk':>7s}  verdict")
    print("-" * 78)

    for state in ("alert", "yawning", "distracted", "drowsy", "microsleep"):
        wm = scenario(state, speed_kmh=50.0, ttc=3.0, closing_mps=50.0 / 3.6)
        need = wm.time_needed_s()
        margin = wm.safety_margin_s
        verdict = "driver can handle it" if margin > 0 else "GUARDIAN MUST ACT"
        print(f"{state:12s}{wm.driver.reaction_time_s:>6.1f}s"
              f"{need - wm.driver.reaction_time_s:>7.1f}s"
              f"{need:>7.1f}s{margin:>8.1f}s{wm.risk_score:>7.0f}  {verdict}")

    print("\nMotorway following case: 90 km/h, lead car also at 88 km/h")
    print("(closing speed only 0.6 m/s -> no emergency for anyone)")
    print("-" * 78)
    for state in ("alert", "microsleep"):
        wm = scenario(state, speed_kmh=90.0, ttc=6.0, closing_mps=0.6)
        print(f"{state:12s}need {wm.time_needed_s():.1f}s  "
              f"margin {wm.safety_margin_s:+.1f}s  risk {wm.risk_score:.0f}")

    print("\nSame urban case on a WET road (friction 0.70):")
    print("-" * 78)
    for state in ("alert", "drowsy"):
        wm = scenario(state, speed_kmh=50.0, ttc=3.0, closing_mps=50.0 / 3.6,
                      surface="wet")
        print(f"{state:12s}need {wm.time_needed_s():.1f}s  "
              f"margin {wm.safety_margin_s:+.1f}s  risk {wm.risk_score:.0f}")

    print("\nVSS projection (what Guardian publishes to the CarSky KUKSA bus):")
    wm = WorldModel(
        frame_id=120, timestamp=6.0,
        scene=SceneState(min_ttc_s=1.8),
        driver=DriverState(state="drowsy", perclos=0.52, longest_closure_s=1.4,
                           explanation="eyes closed 52% of the last 10s"),
        vehicle=VehicleState(speed_kmh=48.0),
    )
    for path, value in wm.to_vss().items():
        print(f"  {path:42s} {value}")
    print("\n" + wm.summary())
