"""
personalization_service.py
==========================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: SERVICE (the business-logic "brain")

This layer sits BETWEEN the API (HTTP) and the REPOSITORY (storage). It:
  - orchestrates work (fetch from repo, transform, return),
  - enforces BUSINESS RULES that belong to neither HTTP nor storage,
  - produces the ActiveDriverProfile that flows to Verticals 2, 3, 4.

TWO design ideas make this file important:

1) DEPENDENCY INJECTION
   The service receives a ProfileRepository from OUTSIDE (constructor arg).
   It depends on the ABSTRACT ProfileRepository, never a concrete class.
   -> We can pass a JSON repo, a SQLite repo, or a fake repo in tests,
      without changing a single line here.

2) SAFETY POLICY (personalization as a SAFETY input)
   A novice driver must NEVER end up with warning sensitivity below MEDIUM,
   even if their saved preference says LOW. Safety overrides comfort.
   This rule lives here because it is a DOMAIN decision, not storage or HTTP.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from models.driver_profile import (
    ActiveDriverProfile,
    DriverProfile,
    ExperienceLevel,
    WarningSensitivity,
)
from repository.profile_repository import ProfileRepository


# ===========================================================================
# SECTION 1 — SERVICE-LEVEL ERRORS
# ===========================================================================
# These are "business" errors. The API layer will later translate them into
# proper HTTP status codes (e.g. DriverNotFoundError -> HTTP 404).
# ---------------------------------------------------------------------------


class ServiceError(Exception):
    """Base class for all business-logic errors in this service."""


class DriverNotFoundError(ServiceError):
    """Raised when a requested driver_id does not exist in storage."""


# ===========================================================================
# SECTION 2 — THE SERVICE
# ===========================================================================


class PersonalizationService:
    """
    The brain of Vertical 1.

    It never touches the disk directly and never speaks HTTP. It only:
      - calls the repository to load/save data,
      - applies business rules (the safety policy),
      - returns clean model objects.
    """

    # A single source of truth for the minimum safety level of novices.
    # Keeping it as a class constant makes the rule easy to find and tune.
    _NOVICE_MIN_SENSITIVITY = WarningSensitivity.MEDIUM

    # Order the sensitivity levels from lowest to highest so we can compare
    # "is X lower than Y?" using their positions in this list.
    _SENSITIVITY_ORDER = [
        WarningSensitivity.LOW,
        WarningSensitivity.MEDIUM,
        WarningSensitivity.HIGH,
    ]

    def __init__(self, repository: ProfileRepository) -> None:
        """
        Args:
            repository: ANY object honouring the ProfileRepository contract.
                        This is DEPENDENCY INJECTION — we receive it, we do
                        not build it ourselves.
        """
        self._repository = repository

    # -- read operations -----------------------------------------------------

    def get_driver(self, driver_id: str) -> DriverProfile:
        """
        Fetch one driver by id.

        Raises:
            DriverNotFoundError: if no driver has this id.
        """
        profile = self._repository.get_by_id(driver_id)
        if profile is None:
            # Turn the repository's "None" into a clear business error.
            raise DriverNotFoundError(f"No driver found with id '{driver_id}'.")
        return profile

    def list_drivers(self) -> list[DriverProfile]:
        """Return all known drivers (may be an empty list)."""
        return self._repository.list_all()

    # -- write operations ----------------------------------------------------

    def create_or_update_driver(self, profile: DriverProfile) -> DriverProfile:
        """
        Save a new driver or update an existing one.

        We apply the safety policy BEFORE saving, so bad safety settings can
        never even be persisted.
        """
        safe_profile = self._apply_safety_policy(profile)
        return self._repository.save(safe_profile)

    # -- the star: activation ------------------------------------------------

    def activate_driver(self, driver_id: str) -> ActiveDriverProfile:
        """
        Make a stored driver the ACTIVE driver and publish their live profile.

        Steps:
          1. Load the stored driver (or fail with DriverNotFoundError).
          2. Apply the safety policy (enforce novice minimum sensitivity).
          3. Convert to the published ActiveDriverProfile (adds a timestamp).

        Returns:
            ActiveDriverProfile: the object Verticals 2/3/4 will consume.
        """
        profile = self.get_driver(driver_id)          # step 1
        safe_profile = self._apply_safety_policy(profile)  # step 2
        return safe_profile.activate()                # step 3

    # -- business rules (private) --------------------------------------------

    def _apply_safety_policy(self, profile: DriverProfile) -> DriverProfile:
        """
        Enforce Guardian's safety rules on a profile.

        RULE 1 — Novice minimum sensitivity:
            A novice driver's warning sensitivity is raised to at least
            MEDIUM. Safety overrides personal comfort preference.

        We return a NEW profile (a copy) rather than mutating the input,
        so callers never get surprised by hidden side effects.
        """
        # If the driver is not a novice, no change is needed.
        if profile.experience != ExperienceLevel.NOVICE:
            return profile

        current = profile.preferences.warning_sensitivity

        # Compare positions in the ordered list. A smaller index = lower level.
        if self._is_lower(current, self._NOVICE_MIN_SENSITIVITY):
            # Make a deep copy so we never mutate the caller's object,
            # then bump the sensitivity up to the enforced minimum.
            adjusted = profile.model_copy(deep=True)
            adjusted.preferences.warning_sensitivity = self._NOVICE_MIN_SENSITIVITY
            return adjusted

        # Already at or above the minimum -> leave it untouched.
        return profile

    def _is_lower(self, a: WarningSensitivity, b: WarningSensitivity) -> bool:
        """Return True if sensitivity `a` is a lower level than `b`."""
        return self._SENSITIVITY_ORDER.index(a) < self._SENSITIVITY_ORDER.index(b)


# ===========================================================================
# SECTION 3 — SELF-TEST (runs only when executed directly)
# ===========================================================================
if __name__ == "__main__":
    # We build an IN-MEMORY fake repository so this demo needs no files.
    # Notice how easily we can inject a different repo — that is DI at work.
    class _InMemoryRepo(ProfileRepository):
        def __init__(self) -> None:
            self._data: dict[str, DriverProfile] = {}

        def get_by_id(self, driver_id):
            return self._data.get(driver_id)

        def list_all(self):
            return list(self._data.values())

        def save(self, profile):
            self._data[profile.driver_id] = profile
            return profile

        def exists(self, driver_id):
            return driver_id in self._data

    service = PersonalizationService(_InMemoryRepo())

    # A novice who WANTS low sensitivity...
    novice = DriverProfile(driver_id="driver_001", name="Teen", experience="novice")
    novice.preferences.warning_sensitivity = WarningSensitivity.LOW
    service.create_or_update_driver(novice)

    active = service.activate_driver("driver_001")
    print("Novice requested LOW sensitivity.")
    print("After safety policy ->", active.preferences.warning_sensitivity.value)
    # Expected: "medium"  (safety overrode the LOW preference)
