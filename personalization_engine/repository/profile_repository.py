"""
profile_repository.py
=====================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: REPOSITORY (the ONLY layer that touches storage)

This file implements the "Repository Pattern". The service layer will ask
this repository for driver data WITHOUT knowing (or caring) whether that
data lives in a JSON file, a SQLite database, or the cloud.

We ship TWO things:
    1. ProfileRepository       -> an ABSTRACT interface (the contract).
    2. JsonProfileRepository   -> a CONCRETE implementation backed by a
                                  simple JSON file (fast to use under time pressure).

Later we can add `SqliteProfileRepository(ProfileRepository)` and swap it in
WITHOUT changing the service, the API, or the models. That is the whole point.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
# `abc` -> tools to declare an ABSTRACT base class (an interface that
#          cannot be used directly; it only defines required methods).
from abc import ABC, abstractmethod

# `json` -> read/write JSON text.
import json

# `pathlib.Path` -> a modern, cross-platform way to handle file paths.
#                   Works the same on Windows and Linux (important for the
#                   judging machine, which may be Linux).
from pathlib import Path

# `Optional[X]` means "an X, or None". `list`/`dict` are built-in generics.
from typing import Optional

# Pydantic error type -> raised if a stored record has a broken shape.
from pydantic import ValidationError

# Our data model — the vocabulary this repository reads/writes.
from models.driver_profile import DriverProfile


# ===========================================================================
# SECTION 1 — CUSTOM ERRORS (clear, named failures)
# ===========================================================================
# Named exceptions make debugging obvious. When something fails, the error
# name alone tells us WHAT went wrong — no guessing on later.
# ---------------------------------------------------------------------------


class RepositoryError(Exception):
    """Base error for any storage problem in this module."""


class CorruptStorageError(RepositoryError):
    """Raised when the backing file exists but is not valid/expected data."""


# ===========================================================================
# SECTION 2 — THE ABSTRACT INTERFACE (the contract every repo must honour)
# ===========================================================================


class ProfileRepository(ABC):
    """
    Abstract contract for driver-profile storage.

    Any concrete storage (JSON, SQLite, ...) MUST provide these four methods.
    The service layer depends on THIS type, never on a concrete class.
    """

    @abstractmethod
    def get_by_id(self, driver_id: str) -> Optional[DriverProfile]:
        """Return the driver with this id, or None if not found."""
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> list[DriverProfile]:
        """Return every stored driver profile."""
        raise NotImplementedError

    @abstractmethod
    def save(self, profile: DriverProfile) -> DriverProfile:
        """Insert or update a profile, then return the saved profile."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, driver_id: str) -> bool:
        """Return True if a driver with this id exists."""
        raise NotImplementedError


# ===========================================================================
# SECTION 3 — JSON IMPLEMENTATION (what we use today)
# ===========================================================================


class JsonProfileRepository(ProfileRepository):
    """
    Stores all driver profiles in ONE JSON file on disk.

    File format (a JSON list of driver objects):
        [
            {"driver_id": "driver_001", "name": "Alice", ...},
            {"driver_id": "driver_002", "name": "Bob",   ...}
        ]

    Design choices for safety:
      - If the file does NOT exist yet, we treat storage as EMPTY (no crash).
      - If the file exists but is broken, we raise CorruptStorageError with a
        clear message instead of a confusing low-level traceback.
      - We re-read the file on each call. Simpler = fewer surprises. For a
        project dataset (a handful of drivers) this is more than fast enough.
    """

    def __init__(self, file_path: str | Path) -> None:
        """
        Args:
            file_path: path to the JSON file that holds the drivers.
                       e.g. "mock_data/drivers.json"
        """
        # Convert whatever we were given into a Path object once, up front.
        self._file_path: Path = Path(file_path)

    # -- internal helpers ----------------------------------------------------

    def _read_raw(self) -> list[dict]:
        """
        Read and parse the JSON file into a list of raw dictionaries.

        Returns an empty list if the file does not exist yet.
        Raises CorruptStorageError if the file exists but is unreadable
        or is not a JSON list.
        """
        # No file yet -> behave as empty storage (a normal first-run state).
        if not self._file_path.exists():
            return []

        try:
            # Read the whole file as UTF-8 text, then parse JSON.
            text = self._file_path.read_text(encoding="utf-8")
            data = json.loads(text)
        except (OSError, json.JSONDecodeError) as exc:
            # OSError    -> file could not be read (permissions, etc.)
            # JSONDecodeError -> the text was not valid JSON.
            raise CorruptStorageError(
                f"Could not read/parse storage file: {self._file_path} ({exc})"
            ) from exc

        # We REQUIRE the top-level JSON to be a list of driver objects.
        if not isinstance(data, list):
            raise CorruptStorageError(
                f"Storage file must contain a JSON list, got {type(data).__name__}: "
                f"{self._file_path}"
            )

        return data

    def _load_index(self) -> dict[str, DriverProfile]:
        """
        Build an in-memory dictionary { driver_id: DriverProfile } from disk.

        Each raw dict is validated through DriverProfile, so a malformed
        record is caught here with a clear error.
        """
        index: dict[str, DriverProfile] = {}

        for raw in self._read_raw():
            try:
                # Validate + convert the raw dict into a real DriverProfile.
                profile = DriverProfile.model_validate(raw)
            except ValidationError as exc:
                raise CorruptStorageError(
                    f"A stored driver record is invalid in {self._file_path}: {exc}"
                ) from exc

            # Key the profile by its id for O(1) lookups.
            index[profile.driver_id] = profile

        return index

    def _write_index(self, index: dict[str, DriverProfile]) -> None:
        """
        Persist the in-memory dictionary back to the JSON file on disk.

        We create the parent folder if it does not exist, so a fresh
        checkout never fails on 'folder not found'.
        """
        # Ensure the containing directory exists (e.g. create "mock_data/").
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert each DriverProfile back into a plain dict (JSON-ready).
        # `mode="json"` makes sure enums/datetimes become JSON-safe values.
        records = [profile.model_dump(mode="json") for profile in index.values()]

        # Write pretty-printed JSON so the file stays human-readable/editable.
        self._file_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # -- public interface (the contract methods) -----------------------------

    def get_by_id(self, driver_id: str) -> Optional[DriverProfile]:
        """Return the driver with this id, or None if not present."""
        # `.get()` on a dict returns None when the key is missing — exactly
        # the behaviour our interface promises.
        return self._load_index().get(driver_id)

    def list_all(self) -> list[DriverProfile]:
        """Return every stored driver profile as a list."""
        return list(self._load_index().values())

    def save(self, profile: DriverProfile) -> DriverProfile:
        """
        Insert a new driver or overwrite an existing one (an "upsert"),
        then persist the whole set back to disk.

        Returns:
            The saved DriverProfile (unchanged, for caller convenience).
        """
        index = self._load_index()      # load current state
        index[profile.driver_id] = profile  # insert or replace
        self._write_index(index)        # persist back to disk
        return profile

    def exists(self, driver_id: str) -> bool:
        """Return True if a driver with this id is stored."""
        return driver_id in self._load_index()


# ===========================================================================
# SECTION 4 — SELF-TEST (runs only when this file is executed directly)
# ===========================================================================
if __name__ == "__main__":
    import tempfile

    # Use a throwaway temp file so we never touch real data during this demo.
    tmp = Path(tempfile.gettempdir()) / "guardian_repo_demo.json"
    if tmp.exists():
        tmp.unlink()

    repo = JsonProfileRepository(tmp)

    print("exists before save:", repo.exists("driver_001"))   # False
    print("list before save:  ", repo.list_all())             # []

    repo.save(DriverProfile(driver_id="driver_001", name="Alice"))
    repo.save(DriverProfile(driver_id="driver_002", name="Bob"))

    print("exists after save: ", repo.exists("driver_001"))   # True
    print("count after save:  ", len(repo.list_all()))        # 2
    print("fetch driver_002:  ", repo.get_by_id("driver_002").name)  # Bob
    print("fetch missing:     ", repo.get_by_id("nope"))      # None
    print(f"\nDemo file written to: {tmp}")
