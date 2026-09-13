"""
explainer.py
============

THE SLOW PATH: WHY THE CAR DID THAT, IN A LANGUAGE THE DRIVER TRUSTS.

The Fast Path decides in under 100 ms and its output is a brake percentage. This
is the other half of the design: the sentence the driver actually hears.

It matters because of the failure mode the project exists to fix. A car that
brakes or beeps without saying why loses trust, and a feature the driver does
not trust gets switched off - at which point its safety value is zero. So the
explanation is not a log line dressed up for a demo; it is the product.

WHAT THIS IS NOT
----------------
`VerifiedCommand.audit_trail()` already exists and is excellent - for an
engineer. It names checks, statuses and clamped values. A driver at 50 km/h
needs one sentence with the cause in it, not an audit. Both are kept: this
module reads the same command and speaks to the other audience.

DETERMINISTIC BY DEFAULT
------------------------
Every explanation here is assembled from templates in `vocabulary.py`. No model
runs, nothing is sampled, the same inputs always produce the same sentence, and
it costs microseconds against a 2-second budget. That is a deliberate choice for
a safety message: a language model that is usually right is a poor fit for the
one sentence that has to be right.

A generative narrator can still be plugged in for richer phrasing - pass
`narrator=` - and the personalization engine's Phi-3 runtime is the intended
implementation. It is optional, it is never on the safety path, and if it fails
the templates answer anyway.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from guardian.explain import vocabulary as V

#: The proposal's KPI for the Slow Path. Templates land ~4 orders of magnitude
#: under this; the budget exists for the day a narrator is attached.
SLOW_PATH_BUDGET_MS = 2000.0

#: Below this TTC a number helps; above it, quoting "TTC 214 s" is noise.
_TTC_WORTH_SAYING_S = 12.0

#: A blink is not evidence. Only a closure long enough to be a lapse of
#: attention is worth telling the driver about.
_WORTH_SAYING_CLOSURE_S = 1.0

#: Matches RuleThresholds.phone_rate — below this the monitor itself does not
#: consider the driver distracted, so it is not a reason for anything.
_WORTH_SAYING_PHONE_RATE = 0.15


@dataclass(frozen=True)
class Factor:
    """One contributing reason. `kind` lets a UI group or colour them."""

    kind: str  # scene | driver | context | vehicle | kernel
    text: str


@dataclass(frozen=True)
class Explanation:
    """What the driver is told, plus enough structure for an HMI to lay it out."""

    lang: str
    headline: str
    factors: tuple[Factor, ...]
    action: str
    severity: str
    latency_ms: float
    #: True when a narrator produced the headline instead of the templates.
    narrated: bool = False

    @property
    def within_budget(self) -> bool:
        return self.latency_ms <= SLOW_PATH_BUDGET_MS

    def to_text(self) -> str:
        lines = [self.headline]
        lines += [f"  - {f.text}" for f in self.factors]
        return "\n".join(lines)


def _object_noun(object_class: str, lang: str) -> str:
    return V.lookup(
        V.OBJECT, lang, (object_class or "").lower(),
        V.PHRASE[lang]["unknown_object"],
    )


def _driver_word(state: str, lang: str) -> str:
    return V.lookup(V.DRIVER_STATE, lang, (state or "alert").lower(), state or "alert")


def _action_word(behavior: str, lang: str) -> str:
    return V.lookup(V.ACTION, lang, str(behavior), str(behavior))


def _check_key(name: object) -> str:
    """
    Normalise a kernel check name into a vocabulary key.

    The kernel numbers its checks for the audit trail ("4. Comfort &
    Drivability"); the ordinal is a presentation detail, not part of the name.
    """
    return re.sub(r"^\s*\d+[.)]\s*", "", str(name or "")).strip().lower()


class GuardianExplainer:
    """
    Turns one (WorldModel, VerifiedCommand) pair into one Explanation.

    Stateless and pure apart from the clock, so it is safe to reuse across a
    whole trip and trivial to test.
    """

    def __init__(
        self,
        lang: str = V.VI,
        narrator: Optional[Callable[[Explanation], str]] = None,
    ) -> None:
        if lang not in V.LANGUAGES:
            raise ValueError(f"unsupported language {lang!r}; expected one of {V.LANGUAGES}")
        self.lang = lang
        self._narrator = narrator

    # -- factors ---------------------------------------------------------

    def _scene_factor(self, world) -> Optional[Factor]:
        target = world.scene.critical_object
        if target is None:
            return None
        noun = _object_noun(target.object_class, self.lang)
        distance = V.number(target.distance_m, 1, self.lang)
        ttc = target.ttc_s
        if ttc is not None and ttc == ttc and ttc < _TTC_WORTH_SAYING_S:  # NaN-safe
            text = V.PHRASE[self.lang]["scene_factor"].format(
                noun, distance, V.number(ttc, 1, self.lang)
            )
        else:
            text = V.PHRASE[self.lang]["scene_factor_no_ttc"].format(noun, distance)
        return Factor("scene", text)

    def _driver_evidence(self, driver) -> str:
        """
        Rebuild the monitor's evidence from its NUMBERS, in this language.

        `DriverState.explanation` is an English engineering sentence written for
        an audit log. Interpolating it into a Vietnamese message produced
        half-translated output ("Tài xế buồn ngủ - drowsy: average eye closure
        0.26 ..."), which is exactly the kind of thing a driver stops trusting.
        The numbers carry the same information and translate cleanly.
        """
        P = V.PHRASE[self.lang]
        parts: list[str] = []

        closure = getattr(driver, "eye_closure", None)
        perclos = getattr(driver, "perclos", None)
        if closure is not None and perclos is not None:
            parts.append(P["driver_evidence_eyes"].format(
                V.number(float(closure), 2, self.lang),
                V.number(100.0 * float(perclos), 0, self.lang),
            ))

        longest = getattr(driver, "longest_closure_s", 0.0) or 0.0
        if float(longest) >= _WORTH_SAYING_CLOSURE_S:
            parts.append(P["driver_evidence_closure"].format(
                V.number(float(longest), 1, self.lang)
            ))

        phone = getattr(driver, "phone_rate", 0.0) or 0.0
        if float(phone) >= _WORTH_SAYING_PHONE_RATE:
            parts.append(P["driver_evidence_phone"].format(
                V.number(100.0 * float(phone), 0, self.lang)
            ))

        return ", ".join(parts)

    def _driver_factors(self, world) -> list[Factor]:
        driver = world.driver
        word = _driver_word(driver.state, self.lang)
        reason = self._driver_evidence(driver)
        if reason:
            text = V.PHRASE[self.lang]["driver_factor_reason"].format(word, reason)
        else:
            text = V.PHRASE[self.lang]["driver_factor"].format(word)
        factors = [Factor("driver", text)]

        # Only worth saying when the driver is NOT the baseline: an alert driver
        # needing 0.8 s is not information.
        if str(driver.state).lower() != "alert":
            reaction = V.number(driver.reaction_time_s, 1, self.lang)
            factors.append(
                Factor("driver", V.PHRASE[self.lang]["driver_reaction"].format(reaction))
            )
        return factors

    def _context_factor(self, world) -> Optional[Factor]:
        ctx = world.context
        wet = str(ctx.surface).lower() == "wet"
        night = bool(ctx.is_night)
        if wet and night:
            return Factor("context", V.PHRASE[self.lang]["context_wet_night"])
        if wet:
            return Factor("context", V.PHRASE[self.lang]["context_wet"])
        if night:
            return Factor("context", V.PHRASE[self.lang]["context_night"])
        return None

    def _kernel_factors(self, command) -> list[Factor]:
        """
        Surface only the checks that CHANGED something.

        A kernel that passed twelve checks has said nothing the driver needs; a
        kernel that vetoed or escalated has said everything.
        """
        out: list[Factor] = []
        for check in getattr(command, "checks", ()) or ():
            status = str(getattr(check.status, "value", check.status)).lower()
            key = {
                "vetoed": "kernel_vetoed",
                "escalated": "kernel_escalated",
                "dampened": "kernel_dampened",
            }.get(status)
            if not key:
                continue
            # `check.message` is free-form English written for the audit trail
            # ("rate limited to 15%/frame"). The engineer already has it there;
            # the driver gets the localised name of the step instead.
            label = V.CHECK.get(self.lang, V.CHECK[V.EN]).get(
                _check_key(getattr(check, "name", ""))
            )
            if label:
                out.append(Factor("kernel", V.PHRASE[self.lang][key].format(label)))
            else:
                out.append(Factor("kernel", V.PHRASE[self.lang][key + "_plain"]))

        if str(getattr(command.source, "value", command.source)).lower() == "failsafe":
            out.append(Factor("kernel", V.PHRASE[self.lang]["failsafe"]))
        return out

    # -- headline --------------------------------------------------------

    def _headline(self, world, command, factors: Sequence[Factor]) -> str:
        phrases = V.PHRASE[self.lang]
        action = _action_word(
            str(getattr(command.behavior, "value", command.behavior)), self.lang
        )

        # The primary cause: the hazard if there is one, else the driver.
        scene = next((f for f in factors if f.kind == "scene"), None)
        driver = next((f for f in factors if f.kind == "driver"), None)
        primary = scene.text if scene else (driver.text if driver else phrases["no_reason"])

        if command.is_intervention:
            head = phrases["headline_intervention"].format(
                action, V.number(command.brake_pct, 0, self.lang), primary
            )
        elif str(getattr(command.warning_level, "value", command.warning_level)) != "none":
            head = phrases["headline_warning"].format(action, primary)
        else:
            return phrases["headline_normal"].format(
                _driver_word(world.driver.state, self.lang)
            )

        # Guardian's actual thesis: the threshold moved because of the driver.
        # Say so, but only when it is true and only once.
        if scene is not None and str(world.driver.state).lower() != "alert":
            head += " " + phrases["earlier_because_driver"].format(
                _driver_word(world.driver.state, self.lang)
            )
        return head

    # -- entry point -----------------------------------------------------

    def explain(self, world, command) -> Explanation:
        started = time.perf_counter()

        factors: list[Factor] = []
        scene = self._scene_factor(world)
        if scene:
            factors.append(scene)
        factors.extend(self._driver_factors(world))
        context = self._context_factor(world)
        if context:
            factors.append(context)
        factors.extend(self._kernel_factors(command))

        headline = self._headline(world, command, factors)
        narrated = False
        severity = str(getattr(command.warning_level, "value", command.warning_level))
        action = _action_word(
            str(getattr(command.behavior, "value", command.behavior)), self.lang
        )

        explanation = Explanation(
            lang=self.lang,
            headline=headline,
            factors=tuple(factors),
            action=action,
            severity=severity,
            latency_ms=0.0,
        )

        # Optional generative phrasing. Never on the safety path: if it raises,
        # the deterministic headline is already in hand and simply stands.
        if self._narrator is not None:
            try:
                narrated_text = self._narrator(explanation)
                if narrated_text and narrated_text.strip():
                    headline = narrated_text.strip()
                    narrated = True
            except Exception:  # noqa: BLE001 - a narrator must not break safety output
                narrated = False

        latency_ms = (time.perf_counter() - started) * 1000.0
        return Explanation(
            lang=self.lang,
            headline=headline,
            factors=tuple(factors),
            action=action,
            severity=severity,
            latency_ms=latency_ms,
            narrated=narrated,
        )


def explain(world, command, lang: str = V.VI) -> Explanation:
    """Convenience wrapper for one-off use."""
    return GuardianExplainer(lang=lang).explain(world, command)
