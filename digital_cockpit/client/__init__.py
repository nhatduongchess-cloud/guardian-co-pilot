"""
client package — the cross-vertical "plug" for the Digital Cockpit.

Re-exports let Vertical 4 write:
    from client import CockpitClient
"""

from client.cockpit_client import (
    CockpitClient,
    CockpitClientError,
    CockpitUnavailableError,
)

__all__ = [
    "CockpitClient",
    "CockpitClientError",
    "CockpitUnavailableError",
]
