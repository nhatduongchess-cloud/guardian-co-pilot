"""
repository package — the ONLY layer that touches storage.

Re-exports let other layers write:
    from repository import JsonProfileRepository, ProfileRepository
"""

from repository.profile_repository import (
    CorruptStorageError,
    JsonProfileRepository,
    ProfileRepository,
    RepositoryError,
)

__all__ = [
    "CorruptStorageError",
    "JsonProfileRepository",
    "ProfileRepository",
    "RepositoryError",
]
