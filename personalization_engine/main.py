"""
main.py
=======

VERTICAL 1 — PERSONALIZATION ENGINE
The ENTRY POINT (the "power switch" of the whole vertical).

Running this file boots the entire Personalization Engine as a live web
service. It does three small jobs and nothing else (Single Responsibility):

    1. Create the FastAPI application object.
    2. Enable CORS so other verticals / browsers may call us.
    3. Plug in the router from api/routes.py.

We keep main.py deliberately TINY. All the real logic lives in the layers
below (routes -> service -> repository -> models). This file is just wiring.

HOW TO RUN
----------
From the folder `personalization_engine/`:

    pip install fastapi "uvicorn[standard]"
    python main.py

Then open in a browser:
    http://127.0.0.1:8000/docs      <- interactive API documentation
    http://127.0.0.1:8000/api/v1/health
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
# `FastAPI`             -> the application class (the ASGI app).
# `CORSMiddleware`      -> lets other origins (e.g. the cockpit UI) call us.
# `RedirectResponse`    -> used to send "/" visitors straight to the docs.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# `uvicorn` -> the web server that actually listens on the network port.
import uvicorn

# Our routers:
#   routes.router        -> driver identification + comfort/UI preferences
#   memory_routes.router -> telemetry ingest, Learning Loop, advisory
#                           thresholds, Vietnamese explanations, consent (§2.4)
from api.routes import router
from api.memory_routes import router as memory_router


# ===========================================================================
# SECTION 1 — CREATE THE APPLICATION
# ===========================================================================
# The title/description/version show up on the auto-generated /docs page and
# make the API look professional when demoing to a reviewer.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Guardian — Personalization Engine (Vertical 1)",
    description=(
        "Vertical 1 of the Guardian AI Operating System. It (1) identifies the "
        "active driver and publishes their comfort/safety preferences, (2) learns "
        "each driver's behaviour over trips via a COLD_START→WARMING→PERSONALIZED "
        "state machine, (3) suggests advisory, safety-floored thresholds to the V4 "
        "Safety Kernel, and (4) produces grounded Vietnamese explanations for the "
        "V2 cockpit. ADVISORY ONLY — V1 never commands the vehicle."
    ),
    version="2.0.0",
)


# ===========================================================================
# SECTION 2 — ENABLE CORS
# ===========================================================================
# For this project we allow every origin so any vertical or browser tab can
# call us without friction. In a real product we would list exact origins.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow any origin (development-friendly)
    allow_credentials=True,
    allow_methods=["*"],      # allow GET, POST, ...
    allow_headers=["*"],      # allow any request headers
)


# ===========================================================================
# SECTION 3 — PLUG IN THE ROUTER
# ===========================================================================
# This attaches every endpoint (all prefixed /api/v1) to the application.
# ---------------------------------------------------------------------------
app.include_router(router)          # driver identification + preferences
app.include_router(memory_router)   # driver memory, thresholds, explanations


# ===========================================================================
# SECTION 4 — A FRIENDLY ROOT REDIRECT
# ===========================================================================
# Visiting the bare "/" sends the user to the interactive docs. Small touch,
# but it makes the service pleasant to open during a live demo.
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the homepage to the interactive API docs."""
    return RedirectResponse(url="/docs")


# ===========================================================================
# SECTION 5 — RUN THE SERVER (only when executed directly)
# ===========================================================================
# `uvicorn.run` starts the network server. `reload=True` auto-restarts the
# server whenever we edit a file — handy while developing, and safe to keep
# here. host 127.0.0.1 = local machine only.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",       # "module:variable" — where to find the app
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
