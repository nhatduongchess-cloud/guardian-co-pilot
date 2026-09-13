"""
services package — the in-memory business logic of the Digital Cockpit.

Re-export lets other layers write:
    from services import CockpitService
"""

from services.cockpit_service import CockpitService

__all__ = ["CockpitService"]
