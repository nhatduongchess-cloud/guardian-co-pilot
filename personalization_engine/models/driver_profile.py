"""
driver_profile.py
==================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: MODELS (the shared vocabulary of the whole system)

This file defines the EXACT shape of every piece of driver data in Guardian.
Every other layer (repository, service, API) and every other vertical
(cockpit, perception, safety kernel) depends on these shapes.

We use Pydantic v2 because it gives us THREE things for free:
    1. Type safety      -> wrong types are rejected automatically.
    2. Validation       -> illegal values (e.g. seat_position = 999) are blocked.
    3. JSON conversion  -> objects turn into JSON and back with one call.

Because this is a "paste-and-run" file, it has ZERO external
dependencies beyond `pydantic`. Nothing here can crash at import time.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
# `datetime` -> used to timestamp WHEN a driver became active.
from datetime import datetime, timezone

# `Enum` -> lets us define a fixed set of allowed text values.
#           This prevents typos like "hihg" instead of "high".
from enum import Enum

# Pydantic pieces:
#   BaseModel -> the parent class that gives validation + JSON powers.
#   Field     -> lets us attach rules (min, max, description) to a field.
#   ConfigDict-> model-level configuration (Pydantic v2 style).
from pydantic import BaseModel, Field, ConfigDict


# ===========================================================================
# SECTION 1 — ENUMS (fixed, safe vocabularies)
# ===========================================================================
# WHY ENUMS?
# A plain string field could receive ANY text ("blue", "banana", "HIGH ").
# By using an Enum, only the exact listed values are legal. This kills a
# whole category of bugs BEFORE the code ever runs on later.
# Each Enum inherits from `str` so the value is still a normal string in JSON.
# ---------------------------------------------------------------------------


class ExperienceLevel(str, Enum):
    """How experienced the driver is. Drives how careful Guardian should be."""

    NOVICE = "novice"            # new driver -> Guardian is more protective
    INTERMEDIATE = "intermediate"
    EXPERIENCED = "experienced"  # veteran   -> Guardian is less intrusive


class UITheme(str, Enum):
    """Which look-and-feel the Digital Cockpit (Vertical 2) should render."""

    MINIMAL = "minimal"  # clean, few elements — for confident drivers
    GUIDED = "guided"    # extra hints & prompts — for new drivers
    SPORT = "sport"      # performance-focused dashboard


class DrivingProfile(str, Enum):
    """Preferred behaviour of the powertrain / driving feel."""

    ECO = "eco"          # save fuel/battery
    COMFORT = "comfort"  # smooth and gentle
    SPORT = "sport"      # responsive and aggressive


class WarningSensitivity(str, Enum):
    """
    HOW EARLY safety warnings fire. THIS IS A SAFETY INPUT.
    It is read by Vertical 4 (Safety Kernel) to shift its decision thresholds.
      - HIGH   -> warn earlier (good for novices / nervous drivers)
      - LOW    -> warn later  (fewer beeps for veterans)
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ===========================================================================
# SECTION 2 — PREFERENCES (what a single driver likes)
# ===========================================================================


class Preferences(BaseModel):
    """
    The set of adjustable comfort + safety settings for ONE driver.

    Every numeric field has a hard min/max so an impossible value
    (e.g. a seat position of 500) can never enter the system.
    """

    # `model_config` = Pydantic v2 way to configure the model.
    # `extra="forbid"` -> if incoming JSON has an unknown field, REJECT it.
    #    This protects us from silent typos in teammate-provided data.
    model_config = ConfigDict(extra="forbid")

    # ge = "greater or equal", le = "less or equal" -> the legal range.
    seat_position: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Seat distance/recline as a percentage (0=full front, 100=full back).",
    )

    climate_temperature: float = Field(
        default=22.0,
        ge=16.0,
        le=30.0,
        description="Preferred cabin temperature in Celsius.",
    )

    mirror_angle: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Mirror tilt as a percentage (0=down, 100=up).",
    )

    # These four use Enums, so only valid options are accepted.
    ui_theme: UITheme = Field(
        default=UITheme.MINIMAL,
        description="Which cockpit theme to render (Vertical 2).",
    )

    driving_profile: DrivingProfile = Field(
        default=DrivingProfile.COMFORT,
        description="Preferred powertrain behaviour.",
    )

    warning_sensitivity: WarningSensitivity = Field(
        default=WarningSensitivity.MEDIUM,
        description="SAFETY INPUT: how early warnings fire (read by Vertical 4).",
    )


# ===========================================================================
# SECTION 3 — DRIVER PROFILE (a full person)
# ===========================================================================


class DriverProfile(BaseModel):
    """
    A complete, stored record of ONE human driver.

    This is what the repository saves/loads and what the API returns
    when you ask "who is driver_001?".
    """

    model_config = ConfigDict(extra="forbid")

    # `...` (Ellipsis) as the default means "this field is REQUIRED".
    # If it is missing, Pydantic raises a clear validation error.
    driver_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier, e.g. 'driver_001'.",
    )

    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable name, e.g. 'Alice'.",
    )

    experience: ExperienceLevel = Field(
        default=ExperienceLevel.INTERMEDIATE,
        description="Skill level of the driver.",
    )

    # A nested model. If `preferences` is omitted, we build a default
    # Preferences() object using `default_factory` (a function that
    # produces the default). We use a factory, NOT a plain default,
    # so every driver gets their OWN fresh Preferences object.
    preferences: Preferences = Field(
        default_factory=Preferences,
        description="This driver's comfort + safety settings.",
    )

    def activate(self) -> "ActiveDriverProfile":
        """
        Convert this stored profile into the LIVE object that the rest of
        Guardian consumes. It is the same data plus a timestamp marking
        the exact moment this driver took control.

        Returns:
            ActiveDriverProfile: the published "who is driving now" object.
        """
        return ActiveDriverProfile(
            driver_id=self.driver_id,
            name=self.name,
            experience=self.experience,
            preferences=self.preferences,
            activated_at=datetime.now(timezone.utc),
        )


# ===========================================================================
# SECTION 4 — ACTIVE DRIVER PROFILE (the output contract of Vertical 1)
# ===========================================================================


class ActiveDriverProfile(BaseModel):
    """
    THE PUBLISHED CONTRACT of Vertical 1.

    This is the single object that flows OUT of the Personalization Engine
    and INTO Verticals 2, 3, and 4. It is a DriverProfile plus the moment
    it became active. Downstream verticals depend ONLY on this shape.
    """

    model_config = ConfigDict(extra="forbid")

    driver_id: str
    name: str
    experience: ExperienceLevel
    preferences: Preferences

    # Stamped automatically when the driver is activated.
    activated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC time the driver became the active driver.",
    )


# ===========================================================================
# SECTION 5 — SELF-TEST (runs only if you execute THIS file directly)
# ===========================================================================
# `python driver_profile.py` will run the block below.
# When this file is IMPORTED by other modules, this block is SKIPPED.
# This lets you sanity-check the file on its own, with zero setup.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Build a driver using only required fields; the rest fill in as defaults.
    alice = DriverProfile(driver_id="driver_001", name="Alice")
    print("Default driver created:")
    print(alice.model_dump_json(indent=2))

    # 2) Activate her -> produce the published contract object.
    active = alice.activate()
    print("\nActive profile (the object other verticals receive):")
    print(active.model_dump_json(indent=2))
