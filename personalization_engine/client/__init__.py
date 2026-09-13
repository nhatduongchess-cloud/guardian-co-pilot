"""
client package — the cross-vertical "plug" for the Personalization Engine.

Re-exports let Vertical 4 write:
    from client import PersonalizationClient
"""

from client.personalization_client import (
    DriverNotFoundError,
    PersonalizationClient,
    PersonalizationClientError,
    PersonalizationUnavailableError,
)

__all__ = [
    "DriverNotFoundError",
    "PersonalizationClient",
    "PersonalizationClientError",
    "PersonalizationUnavailableError",
]
