"""
test_client.py
==============

Tests for VERTICAL 1 -> CLIENT layer (personalization_client.py).

WHY THIS FILE EXISTS
--------------------
The client is the CROSS-VERTICAL bridge V4 will depend on. We must prove:
  - it parses HTTP/JSON responses into real objects,
  - it maps a 404 to a clean DriverNotFoundError,
  - it survives a DOWN service without crashing the caller (V4's loop).

TWO TESTING TECHNIQUES
----------------------
1) Inject a FastAPI TestClient as the client's http_client
   -> the client talks to the real app IN MEMORY (no network port).
   We also override get_service so it uses a FakeRepo (no real data file).

2) httpx.MockTransport that RAISES a ConnectError
   -> lets us simulate "V1 is down" deterministically, to test the
      graceful-failure path.

HOW TO RUN (from personalization_engine/):
    pytest -v
"""

from typing import Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from main import app
from api.routes import get_service
from models.driver_profile import DriverProfile
from repository.profile_repository import ProfileRepository
from services.personalization_service import PersonalizationService
from client.personalization_client import (
    DriverNotFoundError,
    PersonalizationClient,
    PersonalizationUnavailableError,
)


# ---------------------------------------------------------------------------
# Fake in-memory repository (isolates tests from the real drivers.json).
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
# FIXTURE: a PersonalizationClient wired to the real app (in memory) whose
# service is backed by a fresh FakeRepo. Returns (client, repo).
# ---------------------------------------------------------------------------
@pytest.fixture
def client_and_repo():
    repo = FakeRepo()

    def override_get_service() -> PersonalizationService:
        return PersonalizationService(repo)

    app.dependency_overrides[get_service] = override_get_service

    # TestClient runs the ASGI app synchronously, in memory.
    test_http = TestClient(app)
    client = PersonalizationClient(http_client=test_http)

    yield client, repo

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# FIXTURE: a client pointed at a "dead" service. Its MockTransport raises a
# ConnectError on every request, simulating V1 being down.
# ---------------------------------------------------------------------------
@pytest.fixture
def dead_client():
    def _raise(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    dead_http = httpx.Client(
        base_url="http://dead", transport=httpx.MockTransport(_raise)
    )
    return PersonalizationClient(http_client=dead_http)


# ===========================================================================
# GROUP 1 — HAPPY PATH (service is up)
# ===========================================================================


def test_health_true_when_up(client_and_repo):
    client, _ = client_and_repo
    assert client.health() is True


def test_list_drivers_empty(client_and_repo):
    client, _ = client_and_repo
    assert client.list_drivers() == []


def test_create_and_list(client_and_repo):
    client, repo = client_and_repo
    client.create_or_update_driver(DriverProfile(driver_id="driver_001", name="Alice"))

    drivers = client.list_drivers()
    assert len(drivers) == 1
    assert drivers[0].name == "Alice"
    # Confirm it really reached the (fake) storage.
    assert repo.exists("driver_001") is True


def test_create_returns_parsed_object(client_and_repo):
    client, _ = client_and_repo
    result = client.create_or_update_driver(
        DriverProfile(driver_id="driver_001", name="Alice")
    )
    # The client must return a real DriverProfile, not raw JSON.
    assert isinstance(result, DriverProfile)
    assert result.driver_id == "driver_001"


def test_get_driver_found(client_and_repo):
    client, repo = client_and_repo
    repo.save(DriverProfile(driver_id="driver_001", name="Alice"))

    driver = client.get_driver("driver_001")
    assert isinstance(driver, DriverProfile)
    assert driver.name == "Alice"


def test_get_driver_missing_raises_not_found(client_and_repo):
    client, _ = client_and_repo
    with pytest.raises(DriverNotFoundError):
        client.get_driver("ghost")


# ===========================================================================
# GROUP 2 — ACTIVATION (the cross-vertical star)
# ===========================================================================


def test_get_active_driver_returns_active_profile(client_and_repo):
    client, repo = client_and_repo
    repo.save(DriverProfile(driver_id="driver_001", name="Alice"))

    active = client.get_active_driver("driver_001")
    assert active.driver_id == "driver_001"
    # The published contract carries the activation timestamp.
    assert active.activated_at is not None


def test_get_active_driver_enforces_safety(client_and_repo):
    """A novice with LOW must come back as MEDIUM through the client."""
    client, repo = client_and_repo
    novice = DriverProfile(driver_id="driver_003", name="Lan", experience="novice")
    novice.preferences.warning_sensitivity = "low"
    repo.data["driver_003"] = novice

    active = client.get_active_driver("driver_003")
    assert active.preferences.warning_sensitivity.value == "medium"


def test_get_active_driver_missing_raises_not_found(client_and_repo):
    client, _ = client_and_repo
    with pytest.raises(DriverNotFoundError):
        client.get_active_driver("ghost")


# ===========================================================================
# GROUP 3 — GRACEFUL FAILURE (service is down)
# ===========================================================================


def test_health_false_when_service_down(dead_client):
    """health() must return False (NOT raise) when V1 is unreachable."""
    assert dead_client.health() is False


def test_calls_raise_unavailable_when_service_down(dead_client):
    """Data calls must raise a clean PersonalizationUnavailableError."""
    with pytest.raises(PersonalizationUnavailableError):
        dead_client.get_driver("driver_001")

    with pytest.raises(PersonalizationUnavailableError):
        dead_client.get_active_driver("driver_001")
