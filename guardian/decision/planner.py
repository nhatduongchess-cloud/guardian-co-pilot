"""
planner.py
==========

GUARDIAN CO-PILOT — V4 PREDICTION + PLANNING ENGINE

Implements the middle box of the Guardian architecture:

    World Model ──► Prediction Engine ──► Planning Engine ──► Candidate Action
                    (trajectory,           (behaviour +
                     risk, driver           motion planning,
                     behaviour)             cost function)

The Planning Engine PROPOSES. It has no authority: its output is a *candidate*
action that the Safety Kernel must verify before anything reaches the vehicle.
That separation is deliberate — a planner may be optimistic, tuned, or one day
learned; the kernel that guards the actuators must stay simple and auditable.

HOW PLANNING ACTUALLY WORKS HERE
--------------------------------
We do not write `if ttc < 1.2: brake()`. We enumerate candidate brake levels
and score each one with a cost function over three competing objectives, then
take the cheapest:

    safety   — does this action actually avoid the collision?
    comfort  — how unpleasant is the deceleration for the occupants?
    goal     — how much progress do we throw away by slowing down?

Braking hard is always safest and always worst for comfort and progress. The
cost weights encode Guardian's stance: safety dominates, but among equally safe
options we pick the gentlest. This is what stops the system from slamming the
brakes at every parked car — the false-alarm problem that makes drivers switch
ADAS off.

WHERE PERSONALISATION ENTERS
----------------------------
The Driver Behaviour Forecast asks one question: *will this driver react in
time on their own?* An alert driver with a positive safety margin will — so the
planner proposes a warning and leaves the driving to them. A microsleeping
driver will not, so the same scene produces an intervention. The scene did not
change; the human did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

from guardian.world_model.state import WorldModel


# ===========================================================================
# SECTION 1 — VOCABULARY
# ===========================================================================


class BehaviorMode(str, Enum):
    """What kind of manoeuvre the planner is proposing."""

    CRUISE = "cruise"                    # nothing to do
    INFORM = "inform"                    # driver-state notice, no hazard
    WARN = "warn"                        # hazard exists, driver should act
    ASSIST_BRAKE = "assist_brake"        # partial braking support
    EMERGENCY_BRAKE = "emergency_brake"  # full autonomous intervention


class WarningLevel(str, Enum):
    """How loudly the cockpit should speak (consumed by V2 HMI)."""

    NONE = "none"
    INFO = "info"
    CAUTION = "caution"
    CRITICAL = "critical"


#: Brake levels (percent) the planner is allowed to consider.
CANDIDATE_BRAKE_LEVELS: tuple[float, ...] = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)

#: How far ahead the planner acts. A hazard beyond this is re-assessed next
#: frame rather than braked for now.
PLANNING_HORIZON_S = 5.0

#: Cost weights. Safety outranks everything; comfort and progress break ties.
W_SAFETY = 100.0
W_COMFORT = 1.0
W_GOAL = 0.8


# ===========================================================================
# SECTION 2 — PREDICTION ENGINE
# ===========================================================================


@dataclass
class Prediction:
    """Short-horizon forecast used by the planner."""

    ttc_trend_s_per_s: float = 0.0   # <0 means the situation is deteriorating
    ttc_in_1s: float = math.inf      # trajectory forecast
    risk_trend: float = 0.0          # risk points per second
    driver_reacts_in_time: bool = True
    driver_reaction_probability: float = 1.0

    def describe(self) -> str:
        if not math.isfinite(self.ttc_in_1s):
            return "no obstacle on a collision course"
        trend = ("closing" if self.ttc_trend_s_per_s < -0.05
                 else "opening" if self.ttc_trend_s_per_s > 0.05 else "steady")
        return (f"TTC {self.ttc_in_1s:.1f}s in 1s ({trend}), "
                f"driver reacts p={self.driver_reaction_probability:.0%}")


class PredictionEngine:
    """Turns the current world state (plus a little history) into a forecast."""

    def __init__(self, horizon_s: float = 1.0) -> None:
        self.horizon_s = horizon_s

    def predict(
        self,
        world: WorldModel,
        previous: Optional[WorldModel] = None,
    ) -> Prediction:
        """
        Args:
            world:    current unified state.
            previous: the state one frame earlier, for trend estimation.
        """
        prediction = Prediction()

        # --- trajectory forecast -------------------------------------------
        ttc = world.scene.min_ttc_s
        if math.isfinite(ttc):
            if previous is not None and math.isfinite(previous.scene.min_ttc_s):
                dt = max(world.timestamp - previous.timestamp, 1e-3)
                prediction.ttc_trend_s_per_s = (
                    ttc - previous.scene.min_ttc_s
                ) / dt
            else:
                # With no history, assume the gap closes at the natural rate:
                # every second that passes removes a second of TTC.
                prediction.ttc_trend_s_per_s = -1.0

            prediction.ttc_in_1s = max(
                0.0, ttc + prediction.ttc_trend_s_per_s * self.horizon_s
            )

        # --- risk forecast --------------------------------------------------
        if previous is not None:
            dt = max(world.timestamp - previous.timestamp, 1e-3)
            prediction.risk_trend = (world.risk_score - previous.risk_score) / dt

        # --- driver behaviour forecast (the personalisation hook) -----------
        prediction.driver_reaction_probability = self._reaction_probability(world)
        prediction.driver_reacts_in_time = (
            prediction.driver_reaction_probability >= 0.5
        )
        return prediction

    @staticmethod
    def _reaction_probability(world: WorldModel) -> float:
        """
        Probability that the human resolves this alone, before Guardian acts.

        Two independent requirements, multiplied:
          - TIME: is there any margin left after their reaction time?
          - CAPABILITY: is this driver in a state to use that time?

        A microsleeping driver scores ~0 even with generous margin: they are
        not going to see it. An alert driver with no margin also scores low —
        no human is fast enough. Both failure modes call for intervention, and
        the planner does not need to know which one applies.
        """
        margin = world.safety_margin_s
        if not math.isfinite(margin):
            return 1.0  # nothing to react to

        # Time term: 0 at zero margin, saturating to 1 by ~2 s of slack.
        time_term = max(0.0, min(1.0, margin / 2.0))
        return float(time_term * world.driver.capability)


# ===========================================================================
# SECTION 3 — COST FUNCTION
# ===========================================================================


@dataclass
class ActionCost:
    """Cost breakdown of one candidate action. Lower is better."""

    safety: float = 0.0
    comfort: float = 0.0
    goal: float = 0.0

    @property
    def total(self) -> float:
        return W_SAFETY * self.safety + W_COMFORT * self.comfort + W_GOAL * self.goal

    def describe(self) -> str:
        return (f"safety {self.safety:.2f} | comfort {self.comfort:.2f} | "
                f"goal {self.goal:.2f} | total {self.total:.1f}")


def _achieved_decel(brake_pct: float, world: WorldModel) -> float:
    """Deceleration (m/s^2) delivered by a given brake pedal percentage."""
    from guardian.world_model.state import EMERGENCY_DECEL_MPS2

    return world.context.available_decel(EMERGENCY_DECEL_MPS2) * (brake_pct / 100.0)


def evaluate_action(
    brake_pct: float,
    world: WorldModel,
    prediction: Prediction,
) -> ActionCost:
    """
    Score one candidate brake level on the three competing objectives.

    SAFETY — will this braking level stop us in the distance available? We
    compare the distance the obstacle sits at against the distance we would
    still travel: (reaction distance) + (braking distance at this level). If
    the driver is forecast to react in time on their own, Guardian's own
    braking is not what saves us, so the safety cost of doing nothing drops.
    """
    from guardian.world_model.state import EMERGENCY_DECEL_MPS2

    cost = ActionCost()

    # --- comfort: harsh braking is unpleasant; penalise super-linearly ------
    cost.comfort = (brake_pct / 100.0) ** 2

    # --- goal: any braking costs progress ----------------------------------
    cost.goal = brake_pct / 100.0

    # --- safety -------------------------------------------------------------
    target = world.scene.critical_object
    if target is None or not math.isfinite(world.scene.min_ttc_s):
        # No hazard: braking has no safety benefit, only cost. This is what
        # keeps the planner from inventing phantom interventions.
        cost.safety = 0.0
        return cost

    # PLANNING HORIZON. A hazard eight seconds out does not need action on this
    # frame — we will see it again in 50 ms with better information. Without
    # this gate the planner brakes for every distant car, which is precisely
    # the false-alarm behaviour that makes drivers switch ADAS off.
    if world.scene.min_ttc_s > PLANNING_HORIZON_S:
        cost.safety = 0.0
        return cost

    closing = max(target.closing_speed_mps, 0.1)
    available = max(target.distance_m, 0.1)

    if brake_pct == 0.0:
        # "Do nothing" does NOT mean nobody brakes — it means GUARDIAN does not
        # brake and the human does. Whether that is safe depends entirely on
        # the driver, which is how driver state reaches the decision.
        if not prediction.driver_reacts_in_time:
            cost.safety = 5.0        # nobody brakes: collision
            return cost
        delay = world.driver.reaction_time_s
        decel = world.context.available_decel(EMERGENCY_DECEL_MPS2)
    else:
        delay = 0.2                  # system actuation latency
        decel = _achieved_decel(brake_pct, world)

    distance_needed = closing * delay + closing ** 2 / (2.0 * decel)
    shortfall = distance_needed - available

    if shortfall <= 0:
        cost.safety = 0.0                       # collision avoided
    else:
        # Normalised by the available distance so the penalty scales with how
        # badly we fall short rather than with absolute metres.
        cost.safety = float(min(5.0, shortfall / available))
    return cost


# ===========================================================================
# SECTION 4 — PLANNING ENGINE
# ===========================================================================


@dataclass
class CandidateAction:
    """What the planner proposes. Not yet authorised to reach the vehicle."""

    behavior: BehaviorMode
    brake_pct: float
    warning_level: WarningLevel
    hmi_message: str
    rationale: str
    cost: ActionCost = field(default_factory=ActionCost)
    prediction: Prediction = field(default_factory=Prediction)

    def describe(self) -> str:
        return (f"{self.behavior.value:16s} brake {self.brake_pct:5.1f}%  "
                f"[{self.warning_level.value}]  {self.rationale}")


class PlanningEngine:
    """Behaviour + motion planning driven by the cost function."""

    def __init__(
        self,
        brake_levels: Sequence[float] = CANDIDATE_BRAKE_LEVELS,
        prediction_engine: Optional[PredictionEngine] = None,
    ) -> None:
        self.brake_levels = tuple(brake_levels)
        self.prediction_engine = prediction_engine or PredictionEngine()

    def plan(
        self,
        world: WorldModel,
        previous: Optional[WorldModel] = None,
    ) -> CandidateAction:
        """Produce the cheapest acceptable action for this frame."""
        prediction = self.prediction_engine.predict(world, previous)

        # --- motion planning: score every candidate, keep the cheapest ------
        scored = [
            (evaluate_action(level, world, prediction), level)
            for level in self.brake_levels
        ]
        best_cost, best_brake = min(scored, key=lambda pair: pair[0].total)

        # --- behaviour planning: name the manoeuvre and voice the cockpit ---
        behavior, warning, message, rationale = self._classify(
            world, prediction, best_brake
        )

        return CandidateAction(
            behavior=behavior,
            brake_pct=best_brake,
            warning_level=warning,
            hmi_message=message,
            rationale=rationale,
            cost=best_cost,
            prediction=prediction,
        )

    # -- behaviour classification -------------------------------------------

    @staticmethod
    def _classify(
        world: WorldModel,
        prediction: Prediction,
        brake_pct: float,
    ) -> tuple[BehaviorMode, WarningLevel, str, str]:
        """Map the chosen brake level and context onto a named behaviour."""
        driver = world.driver
        hazard = math.isfinite(world.scene.min_ttc_s)

        if brake_pct >= 80.0:
            return (
                BehaviorMode.EMERGENCY_BRAKE, WarningLevel.CRITICAL,
                "Emergency braking - obstacle ahead",
                (f"TTC {world.scene.min_ttc_s:.1f}s vs {world.time_needed_s():.1f}s "
                 f"needed; driver ({driver.state}) cannot recover in time"),
            )

        if brake_pct > 0.0:
            return (
                BehaviorMode.ASSIST_BRAKE, WarningLevel.CRITICAL,
                "Braking assist active",
                (f"margin {world.safety_margin_s:+.1f}s; supporting a "
                 f"{driver.state} driver with {brake_pct:.0f}% brake"),
            )

        if hazard and not prediction.driver_reacts_in_time:
            return (
                BehaviorMode.WARN, WarningLevel.CRITICAL,
                "Obstacle ahead - brake now",
                (f"TTC {world.scene.min_ttc_s:.1f}s and driver reaction "
                 f"probability only {prediction.driver_reaction_probability:.0%}"),
            )

        if hazard:
            return (
                BehaviorMode.WARN, WarningLevel.CAUTION,
                "Obstacle ahead",
                (f"TTC {world.scene.min_ttc_s:.1f}s with {world.safety_margin_s:+.1f}s "
                 f"margin - {driver.state} driver expected to handle it"),
            )

        if driver.state != "alert":
            return (
                BehaviorMode.INFORM, WarningLevel.INFO,
                f"Driver state: {driver.state}",
                driver.explanation or f"driver monitored as {driver.state}",
            )

        return (
            BehaviorMode.CRUISE, WarningLevel.NONE, "",
            "road clear, driver alert",
        )


# ===========================================================================
# SECTION 5 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    from guardian.world_model.state import (
        DRIVER_REACTION_TIME_S, ContextState, DriverState, SceneState,
        TrackedObject, VehicleState,
    )

    def frame(state: str, speed_kmh: float, ttc: float,
              closing_mps: float, surface: str = "dry") -> WorldModel:
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

    planner = PlanningEngine()

    def intervention_threshold(state: str, speed_kmh: float,
                               surface: str = "dry") -> float:
        """Largest TTC at which Guardian still brakes for this driver."""
        closing = speed_kmh / 3.6
        ttc = 0.5
        threshold = 0.0
        while ttc <= PLANNING_HORIZON_S:
            if planner.plan(frame(state, speed_kmh, ttc, closing, surface)).brake_pct > 0:
                threshold = ttc
            ttc += 0.1
        return threshold

    print("THE HEADLINE RESULT — when does Guardian take over?")
    print("=" * 92)
    print("Sweeping TTC to find the intervention boundary. Same car, same road,")
    print("same stationary obstacle at 50 km/h — only the human differs.\n")
    print(f"  {'driver':12s}{'reaction':>10s}{'intervenes below':>19s}"
          f"{'extra warning vs alert':>26s}")
    print("  " + "-" * 76)
    base = intervention_threshold("alert", 50.0)
    for state in ("alert", "yawning", "distracted", "drowsy", "microsleep"):
        thr = intervention_threshold(state, 50.0)
        react = DRIVER_REACTION_TIME_S[state]
        extra = thr - base
        gain = "-" if state == "alert" else f"+{extra:.1f}s earlier"
        print(f"  {state:12s}{react:>9.1f}s{thr:>18.1f}s{gain:>26s}")
    print("\n  At 50 km/h, +1.0 s of warning is ~14 metres of extra stopping room.")

    print("\nDECISION AT TTC 4.0s (inside the discriminating band)")
    print("=" * 92)
    for state in ("alert", "yawning", "distracted", "drowsy", "microsleep"):
        action = planner.plan(frame(state, 50.0, 4.0, 50.0 / 3.6))
        print(f"  {state:11s} {action.describe()}")

    print("\nCOST BREAKDOWN for a drowsy driver at TTC 2.5s")
    print("=" * 92)
    world = frame("drowsy", 50.0, 2.5, 50.0 / 3.6)
    pred = planner.prediction_engine.predict(world)
    print(f"  forecast: {pred.describe()}\n")
    print(f"  {'brake':>7s}{'safety':>10s}{'comfort':>10s}{'goal':>8s}{'TOTAL':>10s}")
    print("  " + "-" * 46)
    for level in CANDIDATE_BRAKE_LEVELS:
        c = evaluate_action(level, world, pred)
        mark = "  <- chosen" if level == planner.plan(world).brake_pct else ""
        print(f"  {level:>6.0f}%{c.safety:>10.2f}{c.comfort:>10.2f}"
              f"{c.goal:>8.2f}{c.total:>10.1f}{mark}")

    print("\nNO FALSE ALARMS: motorway following, 90 km/h, closing 0.6 m/s")
    print("=" * 92)
    for state in ("alert", "microsleep"):
        action = planner.plan(frame(state, 90.0, 8.0, 0.6))
        print(f"  {state:11s} {action.describe()}")

    print("\nWET ROAD raises the bar for everyone (50 km/h, TTC 2.5s)")
    print("=" * 92)
    for state in ("alert", "drowsy"):
        action = planner.plan(frame(state, 50.0, 2.5, 50.0 / 3.6, surface="wet"))
        print(f"  {state:11s} {action.describe()}")
