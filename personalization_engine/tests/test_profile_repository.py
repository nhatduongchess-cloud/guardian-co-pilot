"""
test_profile_repository.py
==========================

Unit tests for VERTICAL 1 -> REPOSITORY layer (profile_repository.py).

WHY THIS FILE EXISTS
--------------------
The repository is the ONLY code that touches the disk, so it is the most
likely place for real-world failures (missing files, corrupt JSON, bad
records). These tests prove it behaves correctly in ALL of those cases.

KEY TOOL: `tmp_path`
--------------------
`tmp_path` is a built-in pytest fixture. For each test, pytest creates a
brand-new empty temporary folder and passes its Path in. It is deleted
automatically afterwards. This means every test is isolated and NEVER
touches our real mock_data — no cleanup code needed.

HOW TO RUN (from personalization_engine/):
    pytest -v
"""

import json

import pytest

from models.driver_profile import DriverProfile, WarningSensitivity
from repository.profile_repository import (
    CorruptStorageError,
    JsonProfileRepository,
)


# ---------------------------------------------------------------------------
# A small helper so each test can build a repo pointing at a fresh temp file.
# ---------------------------------------------------------------------------
def _make_repo(tmp_path) -> JsonProfileRepository:
    """Return a repository backed by a fresh JSON file inside tmp_path."""
    return JsonProfileRepository(tmp_path / "drivers.json")


# ===========================================================================
# GROUP 1 — EMPTY / MISSING FILE behaves as empty storage (no crash)
# ===========================================================================


def test_missing_file_is_treated_as_empty(tmp_path):
    """Before any save, the file does not exist; the repo must look empty."""
    repo = _make_repo(tmp_path)

    assert repo.list_all() == []
    assert repo.exists("driver_001") is False
    assert repo.get_by_id("driver_001") is None


# ===========================================================================
# GROUP 2 — SAVE + READ BACK
# ===========================================================================


def test_save_then_get_by_id(tmp_path):
    """A saved driver can be fetched back with identical data."""
    repo = _make_repo(tmp_path)
    repo.save(DriverProfile(driver_id="driver_001", name="Alice"))

    fetched = repo.get_by_id("driver_001")
    assert fetched is not None
    assert fetched.name == "Alice"


def test_save_multiple_and_list_all(tmp_path):
    """Saving several drivers must make all of them appear in list_all()."""
    repo = _make_repo(tmp_path)
    repo.save(DriverProfile(driver_id="driver_001", name="Alice"))
    repo.save(DriverProfile(driver_id="driver_002", name="Bob"))

    ids = {d.driver_id for d in repo.list_all()}
    assert ids == {"driver_001", "driver_002"}


def test_exists_reflects_saved_state(tmp_path):
    """exists() must be False before save and True after."""
    repo = _make_repo(tmp_path)
    assert repo.exists("driver_001") is False

    repo.save(DriverProfile(driver_id="driver_001", name="Alice"))
    assert repo.exists("driver_001") is True


# ===========================================================================
# GROUP 3 — UPSERT: saving an existing id overwrites, never duplicates
# ===========================================================================


def test_save_same_id_overwrites(tmp_path):
    """Saving the same driver_id twice updates it and keeps a single record."""
    repo = _make_repo(tmp_path)
    repo.save(DriverProfile(driver_id="driver_001", name="Alice"))

    # Save again with a different name -> should REPLACE, not duplicate.
    repo.save(DriverProfile(driver_id="driver_001", name="Alice Smith"))

    all_drivers = repo.list_all()
    assert len(all_drivers) == 1
    assert all_drivers[0].name == "Alice Smith"


# ===========================================================================
# GROUP 4 — PERSISTENCE: data survives across repository instances
# ===========================================================================


def test_data_persists_across_new_instances(tmp_path):
    """
    A second repository pointing at the same file must see the data the
    first one saved. This proves data really reached the disk.
    """
    path = tmp_path / "drivers.json"

    repo_writer = JsonProfileRepository(path)
    repo_writer.save(DriverProfile(driver_id="driver_001", name="Alice"))

    # Brand-new object, same file.
    repo_reader = JsonProfileRepository(path)
    assert repo_reader.exists("driver_001") is True
    assert repo_reader.get_by_id("driver_001").name == "Alice"


def test_saved_enum_round_trips_through_disk(tmp_path):
    """A nested enum (warning_sensitivity) must survive the JSON write/read."""
    repo = _make_repo(tmp_path)
    driver = DriverProfile(driver_id="driver_001", name="Alice")
    driver.preferences.warning_sensitivity = WarningSensitivity.HIGH
    repo.save(driver)

    reloaded = JsonProfileRepository(tmp_path / "drivers.json").get_by_id("driver_001")
    assert reloaded.preferences.warning_sensitivity == WarningSensitivity.HIGH


# ===========================================================================
# GROUP 5 — AUTO-CREATE parent folder
# ===========================================================================


def test_save_creates_missing_parent_folder(tmp_path):
    """
    If the target file lives in a folder that does not exist yet, save()
    must create the folder rather than crash.
    """
    nested = tmp_path / "deep" / "nested" / "drivers.json"
    repo = JsonProfileRepository(nested)

    repo.save(DriverProfile(driver_id="driver_001", name="Alice"))

    assert nested.exists()  # the file (and its folders) were created


# ===========================================================================
# GROUP 6 — CORRUPT STORAGE: broken files raise a CLEAR named error
# ===========================================================================


def test_invalid_json_raises_corrupt_error(tmp_path):
    """A file containing non-JSON text must raise CorruptStorageError."""
    path = tmp_path / "drivers.json"
    path.write_text("this is not json {{{", encoding="utf-8")

    repo = JsonProfileRepository(path)
    with pytest.raises(CorruptStorageError):
        repo.list_all()


def test_json_that_is_not_a_list_raises(tmp_path):
    """The top-level JSON must be a list; a dict must be rejected clearly."""
    path = tmp_path / "drivers.json"
    path.write_text(json.dumps({"driver_id": "x"}), encoding="utf-8")

    repo = JsonProfileRepository(path)
    with pytest.raises(CorruptStorageError):
        repo.list_all()


def test_invalid_record_shape_raises(tmp_path):
    """
    A record missing required fields (no 'name') must raise CorruptStorageError
    when the repo tries to validate it into a DriverProfile.
    """
    path = tmp_path / "drivers.json"
    # Valid JSON list, but the driver object is missing the required 'name'.
    path.write_text(json.dumps([{"driver_id": "driver_001"}]), encoding="utf-8")

    repo = JsonProfileRepository(path)
    with pytest.raises(CorruptStorageError):
        repo.list_all()
