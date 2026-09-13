"""
api package — the HTTP layer of the Digital Cockpit.

Re-exports let main.py write:
    from api import router
"""

from api.routes import get_cockpit_service, router

__all__ = ["get_cockpit_service", "router"]
