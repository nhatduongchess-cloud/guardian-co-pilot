"""
test_api.py
===========

Integration tests for VERTICAL 1 -> API layer (routes.py + main.py).

WHY THIS FILE EXISTS
--------------------
These are the HIGHEST-level tests: they exercise the whole stack
(HTTP -> route -> service -> repository -> model) exactly as the vehicle
and other verticals will. If these pass, Vertical 1 truly works end to end.

KEY TOOLS
---------
1) TestClient
   A fake in-memory browser from FastAPI. It calls endpoints directly,
   WITHOUT opening a network port or running Uvicorn. Fast and reliable.

2) app.dependency_overrides
   We swap the real `get_service` for one backed by an in-memory FakeRepo.
   This keeps tests isolated — they never read or write the real
   mock_data/drivers.json file.

HOW TO RUN (from personalization_engine/):
    pip install fastapi httpx pytest
    pytest -v
"""

from typing import Optional

import pytest
from fastapi.testclient import TestClient

from main import app
from models.driver_profile import DriverProfile
from repository.profile_repository import ProfileRepository
from services.personalization_service import PersonalizationService
from api.routes import get_service


# ---------------------------------------------------------------------------
# A fake in-memory repository (same idea as in the service tests).
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
# FIXTURE: a TestClient whose service uses a FRESH FakeRepo per test.
# We override get_service, yield the client + repo, then clean the override.
# ---------------------------------------------------------------------------
@pytest.fixture
def client_and_repo():
    repo = FakeRepo()

    # This replacement provider ignores the real JSON repo and uses our fake.
    def override_get_service() -> PersonalizationService:
        return PersonalizationService(repo)

    # Install the override so every endpoint gets the fake-backed service.
    app.dependency_overrides[get_service] = override_get_service

    client = TestClient(app)
    yield client, repo

    # Always clean up so overrides never leak into other tests.
    app.dependency_overrides.clear()


# ===========================================================================
# GROUP 1 — HEALTH
# ===========================================================================


def test_health_returns_ok(client_and_repo):
    client, _ = client_and_repo
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ===========================================================================
# GROUP 2 — LIST + GET
# ===========================================================================


def test_list_drivers_empty(client_and_repo):
    client, _ = client_and_repo
    response = client.get("/api/v1/drivers")

    assert response.status_code == 200
    assert response.json() == []


def test_get_driver_found(client_and_repo):
    client, repo = client_and_repo
    repo.save(DriverProfile(driver_id="driver_001", name="Alice"))

    response = client.get("/api/v1/drivers/driver_001")

    assert response.status_code == 200
    assert response.json()["name"] == "Alice"


def test_get_driver_not_found_returns_404(client_and_repo):
    client, _ = client_and_repo
    response = client.get("/api/v1/drivers/ghost")

    assert response.status_code == 404


# ===========================================================================
# GROUP 3 — CREATE (POST /drivers)
# ===========================================================================


def test_create_driver_returns_201(client_and_repo):
    client, repo = client_and_repo
    body = {"driver_id": "driver_001", "name": "Alice"}

    response = client.post("/api/v1/drivers", json=body)

    assert response.status_code == 201
    assert response.json()["driver_id"] == "driver_001"
    assert repo.exists("driver_001") is True


def test_create_novice_low_is_stored_as_medium(client_and_repo):
    """The safety policy must apply through the full HTTP stack."""
    client, _ = client_and_repo
    body = {
        "driver_id": "driver_003",
        "name": "Lan",
        "experience": "novice",
        "preferences": {"warning_sensitivity": "low"},
    }

    response = client.post("/api/v1/drivers", json=body)

    assert response.status_code == 201
    # low was requested, medium must come back.
    assert response.json()["preferences"]["warning_sensitivity"] == "medium"


# ===========================================================================
# GROUP 4 — VALIDATION (FastAPI returns 422 for bad input)
# ===========================================================================


def test_create_with_missing_name_returns_422(client_and_repo):
    client, _ = client_and_repo
    response = client.post("/api/v1/drivers", json={"driver_id": "x"})  # no name

    assert response.status_code == 422


def test_create_with_invalid_enum_returns_422(client_and_repo):
    client, _ = client_and_repo
    body = {
        "driver_id": "x",
        "name": "X",
        "preferences": {"ui_theme": "rainbow"},  # not a valid theme
    }
    response = client.post("/api/v1/drivers", json=body)

    assert response.status_code == 422


def test_create_with_out_of_range_seat_returns_422(client_and_repo):
    client, _ = client_and_repo
    body = {
        "driver_id": "x",
        "name": "X",
        "preferences": {"seat_position": 999},  # above max 100
    }
    response = client.post("/api/v1/drivers", json=body)

    assert response.status_code == 422


# ===========================================================================
# GROUP 5 — ACTIVATE (the star endpoint)
# ===========================================================================


def test_activate_returns_active_profile(client_and_repo):
    client, repo = client_and_repo
    repo.save(DriverProfile(driver_id="driver_001", name="Alice"))

    response = client.post("/api/v1/drivers/driver_001/activate")

    assert response.status_code == 200
    data = response.json()
    assert data["driver_id"] == "driver_001"
    # The published contract must include the activation timestamp.
    assert "activated_at" in data


def test_activate_novice_enforces_safety(client_and_repo):
    client, repo = client_and_repo
    novice = DriverProfile(
        driver_id="driver_003", name="Lan", experience="novice"
    )
    novice.preferences.warning_sensitivity = "low"
    repo.data["driver_003"] = novice  # inject raw, bypassing save()

    response = client.post("/api/v1/drivers/driver_003/activate")

    assert response.status_code == 200
    assert response.json()["preferences"]["warning_sensitivity"] == "medium"


def test_activate_missing_driver_returns_404(client_and_repo):
    client, _ = client_and_repo
    response = client.post("/api/v1/drivers/ghost/activate")

    assert response.status_code == 404
