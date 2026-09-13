"""
api package — the HTTP layer of the Personalization Engine.

Re-exports let main.py write:
    from api import router
"""

from api.routes import get_repository, get_service, router

__all__ = [
    "get_repository",
    "get_service",
    "router",
]
