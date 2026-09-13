"""
test_personalization_service.py
===============================

Unit tests for VERTICAL 1 -> SERVICE layer (personalization_service.py).

WHY THIS FILE EXISTS
--------------------
The service holds Guardian's BUSINESS RULES — most importantly the safety
policy that raises a novice driver's warning sensitivity. A bug here is a
SAFETY bug, so we test every branch of that rule.

KEY TECHNIQUE: a FAKE repository
--------------------------------
Instead of touching the disk, we inject a tiny in-memory repository that
honours the ProfileRepository contract. This:
  - makes tests fast and isolated (no files),
  - proves Dependency Injection works (the service accepts ANY repo).

HOW TO RUN (from personalization_engine/):
    pytest -v
"""

from typing import Optional

import pytest

from models.driver_profile import (
    ActiveDriverProfile,
    DriverProfile,
    ExperienceLevel,
    WarningSensitivity,
)
from repository.profile_repository import ProfileRepository
from services.personalization_service import (
    DriverNotFoundError,
    PersonalizationService,
)


# ---------------------------------------------------------------------------
# A FAKE repository used only in tests. It stores drivers in a plain dict.
# It fulfils the ProfileRepository contract, so the service cannot tell it
# apart from the real JSON one — that is exactly the power of the interface.
# ---------------------------------------------------------------------------
class FakeRepo(ProfileRepository):
    def __init__(self) -> None:
        self.data: dict[str, DriverProfile] = {}

    def get_by_id(self, driver_id: str) -> Optional[DriverProfile]:
        return self.data.get(driver_id)

    def list_all(self) -> list[DriverProfile]:
        return list(self.data.values())

    def save(self, profile: DriverProfile) -> DriverProfile:
        self.data[profile.driver_id] = profile
        return profile

    def exists(self, driver_id: str) -> bool:
        return driver_id in self.data


# ---------------------------------------------------------------------------
# A pytest FIXTURE: gives every test a fresh service wired to a fresh FakeRepo.
# `service.repo` style access is available via the returned tuple.
# ---------------------------------------------------------------------------
@pytest.fixture
def service_and_repo():
    repo = FakeRepo()
    service = PersonalizationService(repo)
    return service, repo


# ===========================================================================
# GROUP 1 — READ operations
# ===========================================================================


def test_get_driver_returns_saved_driver(service_and_repo):
    service, repo = service_and_repo
    repo.save(DriverProfile(driver_id="driver_001", name="Alice"))

    result = service.get_driver("driver_001")
    assert result.name == "Alice"


def test_get_driver_missing_raises(service_and_repo):
    service, _ = service_and_repo
    with pytest.raises(DriverNotFoundError):
        service.get_driver("ghost")


def test_list_drivers_empty(service_and_repo):
    service, _ = service_and_repo
    assert service.list_drivers() == []


def test_list_drivers_returns_all(service_and_repo):
    service, repo = service_and_repo
    repo.save(DriverProfile(driver_id="driver_001", name="Alice"))
    repo.save(DriverProfile(driver_id="driver_002", name="Bob"))

    ids = {d.driver_id for d in service.list_drivers()}
    assert ids == {"driver_001", "driver_002"}


# ===========================================================================
# GROUP 2 — WRITE operations
# ===========================================================================


def test_create_or_update_persists(service_and_repo):
    service, repo = service_and_repo
    service.create_or_update_driver(DriverProfile(driver_id="driver_001", name="Alice"))

    assert repo.exists("driver_001") is True


# ===========================================================================
# GROUP 3 — SAFETY POLICY (the important part)
# ===========================================================================


def test_novice_low_is_raised_to_medium_on_save(service_and_repo):
    """A novice asking for LOW must be stored as MEDIUM."""
    service, repo = service_and_repo

    novice = DriverProfile(driver_id="driver_001", name="Teen", experience="novice")
    novice.preferences.warning_sensitivity = WarningSensitivity.LOW
    service.create_or_update_driver(novice)

    stored = repo.get_by_id("driver_001")
    assert stored.preferences.warning_sensitivity == WarningSensitivity.MEDIUM


def test_novice_high_is_left_untouched(service_and_repo):
    """A novice already at HIGH must NOT be lowered."""
    service, repo = service_and_repo

    novice = DriverProfile(driver_id="driver_001", name="Teen", experience="novice")
    novice.preferences.warning_sensitivity = WarningSensitivity.HIGH
    service.create_or_update_driver(novice)

    stored = repo.get_by_id("driver_001")
    assert stored.preferences.warning_sensitivity == WarningSensitivity.HIGH


def test_experienced_low_stays_low(service_and_repo):
    """An experienced driver may keep LOW sensitivity; policy does not apply."""
    service, repo = service_and_repo

    veteran = DriverProfile(
        driver_id="driver_001", name="Dad", experience="experienced"
    )
    veteran.preferences.warning_sensitivity = WarningSensitivity.LOW
    service.create_or_update_driver(veteran)

    stored = repo.get_by_id("driver_001")
    assert stored.preferences.warning_sensitivity == WarningSensitivity.LOW


def test_safety_policy_does_not_mutate_caller_object(service_and_repo):
    """
    The original object passed by the caller must be untouched; the policy
    works on a COPY. This guards against hidden side effects.
    """
    service, _ = service_and_repo

    novice = DriverProfile(driver_id="driver_001", name="Teen", experience="novice")
    novice.preferences.warning_sensitivity = WarningSensitivity.LOW

    service.create_or_update_driver(novice)

    # The caller's own object must STILL say LOW (only the stored copy changed).
    assert novice.preferences.warning_sensitivity == WarningSensitivity.LOW


# ===========================================================================
# GROUP 4 — ACTIVATION (the published contract)
# ===========================================================================


def test_activate_returns_active_profile_with_timestamp(service_and_repo):
    service, repo = service_and_repo
    repo.save(DriverProfile(driver_id="driver_001", name="Alice"))

    active = service.activate_driver("driver_001")

    assert isinstance(active, ActiveDriverProfile)
    assert active.driver_id == "driver_001"
    assert active.activated_at.tzinfo is not None  # timezone-aware UTC stamp


def test_activate_applies_safety_policy(service_and_repo):
    """
    Even if a novice with LOW was somehow stored directly (bypassing save),
    activation must still enforce the safety minimum in the published object.
    """
    service, repo = service_and_repo

    # Inject directly into the fake repo, bypassing the service's save().
    novice = DriverProfile(driver_id="driver_001", name="Teen", experience="novice")
    novice.preferences.warning_sensitivity = WarningSensitivity.LOW
    repo.data["driver_001"] = novice

    active = service.activate_driver("driver_001")
    assert active.preferences.warning_sensitivity == WarningSensitivity.MEDIUM


def test_activate_missing_driver_raises(service_and_repo):
    service, _ = service_and_repo
    with pytest.raises(DriverNotFoundError):
        service.activate_driver("ghost")
