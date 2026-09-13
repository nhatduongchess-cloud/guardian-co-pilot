"""
test_driver_profile.py
=======================

Unit tests for VERTICAL 1 -> MODELS layer (driver_profile.py).

WHY THIS FILE EXISTS
--------------------
These tests are our "safety net". They automatically prove that:
  - good data is accepted,
  - bad data is REJECTED (not silently swallowed),
  - activate() builds the correct published contract.

If a teammate accidentally breaks the model on later, running
`pytest` turns RED immediately and tells us exactly what broke.

HOW TO RUN
----------
From the folder `personalization_engine/`:
    pip install pytest pydantic
    pytest -v

We run from `personalization_engine/` so that `from models...` resolves.
"""

# `pytest` gives us the test runner + `pytest.raises` to assert that
# illegal input DOES raise an error.
import pytest

# Pydantic raises `ValidationError` whenever input breaks a rule.
from pydantic import ValidationError

# The things we are testing. Importing from the package `models`
# (this works because models/__init__.py re-exports them).
from models.driver_profile import (
    ActiveDriverProfile,
    DriverProfile,
    DrivingProfile,
    ExperienceLevel,
    Preferences,
    UITheme,
    WarningSensitivity,
)


# ===========================================================================
# GROUP 1 — DEFAULTS: a minimal driver fills in sensible defaults
# ===========================================================================


def test_driver_requires_only_id_and_name():
    """A driver built with only the required fields must succeed."""
    driver = DriverProfile(driver_id="driver_001", name="Alice")

    assert driver.driver_id == "driver_001"
    assert driver.name == "Alice"
    # experience defaults to INTERMEDIATE
    assert driver.experience == ExperienceLevel.INTERMEDIATE
    # preferences object is auto-created
    assert isinstance(driver.preferences, Preferences)


def test_preferences_have_expected_defaults():
    """Each preference field must default to the value we designed."""
    prefs = Preferences()

    assert prefs.seat_position == 50
    assert prefs.climate_temperature == 22.0
    assert prefs.mirror_angle == 50
    assert prefs.ui_theme == UITheme.MINIMAL
    assert prefs.driving_profile == DrivingProfile.COMFORT
    assert prefs.warning_sensitivity == WarningSensitivity.MEDIUM


# ===========================================================================
# GROUP 2 — REQUIRED FIELDS: missing required data must be rejected
# ===========================================================================


def test_missing_driver_id_raises():
    """driver_id is required (declared with '...'); omitting it must fail."""
    with pytest.raises(ValidationError):
        DriverProfile(name="Alice")  # no driver_id


def test_empty_driver_id_raises():
    """driver_id has min_length=1; an empty string must fail."""
    with pytest.raises(ValidationError):
        DriverProfile(driver_id="", name="Alice")


def test_missing_name_raises():
    """name is required; omitting it must fail."""
    with pytest.raises(ValidationError):
        DriverProfile(driver_id="driver_001")  # no name


# ===========================================================================
# GROUP 3 — NUMERIC RANGE: values outside min/max must be rejected
# ===========================================================================


def test_seat_position_above_max_raises():
    """seat_position is capped at 100."""
    with pytest.raises(ValidationError):
        Preferences(seat_position=101)


def test_seat_position_below_min_raises():
    """seat_position cannot be negative."""
    with pytest.raises(ValidationError):
        Preferences(seat_position=-1)


def test_climate_temperature_out_of_range_raises():
    """climate_temperature must stay within 16.0 .. 30.0."""
    with pytest.raises(ValidationError):
        Preferences(climate_temperature=45.0)


def test_seat_position_boundaries_are_allowed():
    """The exact min and max values (0 and 100) must be accepted."""
    assert Preferences(seat_position=0).seat_position == 0
    assert Preferences(seat_position=100).seat_position == 100


# ===========================================================================
# GROUP 4 — ENUM SAFETY: invalid text choices must be rejected
# ===========================================================================


def test_invalid_ui_theme_raises():
    """'rainbow' is not a valid UITheme."""
    with pytest.raises(ValidationError):
        Preferences(ui_theme="rainbow")


def test_invalid_warning_sensitivity_raises():
    """A typo like 'hihg' must be rejected (this is a SAFETY field)."""
    with pytest.raises(ValidationError):
        Preferences(warning_sensitivity="hihg")


def test_valid_enum_values_are_accepted():
    """Passing the raw string of a valid enum must work and coerce to enum."""
    prefs = Preferences(ui_theme="guided", warning_sensitivity="high")
    assert prefs.ui_theme == UITheme.GUIDED
    assert prefs.warning_sensitivity == WarningSensitivity.HIGH


# ===========================================================================
# GROUP 5 — EXTRA FIELDS: unknown keys must be rejected (extra="forbid")
# ===========================================================================


def test_unknown_preference_field_raises():
    """A typo'd key like 'climate_temp' must be rejected, not ignored."""
    with pytest.raises(ValidationError):
        Preferences(climate_temp=21.0)  # correct name is climate_temperature


# ===========================================================================
# GROUP 6 — ACTIVATE(): produces the published contract correctly
# ===========================================================================


def test_activate_returns_active_profile():
    """activate() must return an ActiveDriverProfile with matching data."""
    driver = DriverProfile(
        driver_id="driver_002",
        name="Bob",
        experience=ExperienceLevel.EXPERIENCED,
    )

    active = driver.activate()

    assert isinstance(active, ActiveDriverProfile)
    assert active.driver_id == "driver_002"
    assert active.name == "Bob"
    assert active.experience == ExperienceLevel.EXPERIENCED
    # preferences carry over unchanged
    assert active.preferences == driver.preferences


def test_activate_stamps_a_timestamp():
    """activate() must attach a UTC activated_at timestamp."""
    driver = DriverProfile(driver_id="driver_003", name="Cara")
    active = driver.activate()

    # timezone-aware means tzinfo is present (UTC), not None.
    assert active.activated_at.tzinfo is not None


# ===========================================================================
# GROUP 7 — ISOLATION: each driver gets its OWN preferences object
# ===========================================================================


def test_preferences_are_not_shared_between_drivers():
    """
    Thanks to default_factory, changing one driver's preferences must NOT
    affect another driver's. This guards against a nasty shared-state bug.
    """
    driver_a = DriverProfile(driver_id="a", name="A")
    driver_b = DriverProfile(driver_id="b", name="B")

    # Mutate A's seat position.
    driver_a.preferences.seat_position = 10

    # B must remain untouched at its default.
    assert driver_b.preferences.seat_position == 50


# ===========================================================================
# GROUP 8 — JSON ROUND-TRIP: object -> JSON -> object stays identical
# ===========================================================================


def test_json_round_trip_preserves_data():
    """
    Serialize to JSON and rebuild. The result must equal the original.
    This proves the object travels safely across the network to other
    verticals (which is exactly how they will receive it).
    """
    original = DriverProfile(
        driver_id="driver_004",
        name="Dana",
        experience=ExperienceLevel.NOVICE,
        preferences=Preferences(
            seat_position=30,
            ui_theme=UITheme.GUIDED,
            warning_sensitivity=WarningSensitivity.HIGH,
        ),
    )

    as_json = original.model_dump_json()          # object -> JSON string
    rebuilt = DriverProfile.model_validate_json(as_json)  # JSON -> object

    assert rebuilt == original
