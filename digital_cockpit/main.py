"""
main.py
=======

VERTICAL 2 — DIGITAL COCKPIT
The ENTRY POINT (the "power switch" of the cockpit service).

Guardian runs as FOUR SEPARATE services (like real automotive ECUs), each on
its own port. This is Vertical 2; it listens on port 8001.

    Port map:
        8000  Personalization Engine (V1)
        8001  Digital Cockpit        (V2)   <-- this service
        8002  Perception Engine       (V3)
        8003  Decision + Safety Kernel(V4, MAIN)

Like V1's main.py, this file is deliberately tiny: create app, enable CORS,
plug in the router. All logic lives in the layers below.

HOW TO RUN
----------
From the folder `digital_cockpit/`:
    pip install fastapi "uvicorn[standard]"
    python main.py

Then open:
    http://127.0.0.1:8001/docs
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

import uvicorn

from api.routes import router


# ===========================================================================
# SECTION 1 — CREATE THE APPLICATION
# ===========================================================================
app = FastAPI(
    title="Guardian - Digital Cockpit",
    description=(
        "Vertical 2 of the Guardian AI Operating System. "
        "Holds the live in-car display state (driver, theme, telemetry, "
        "alerts). Vertical 4 pushes updates; the screen reads the state."
    ),
    version="1.0.0",
)


# ===========================================================================
# SECTION 2 — ENABLE CORS
# ===========================================================================
# Allow any origin so the browser-based display (and other verticals) can
# call this service without cross-origin friction during this project.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# SECTION 3 — PLUG IN THE ROUTER
# ===========================================================================
app.include_router(router)


# ===========================================================================
# SECTION 4 — FRIENDLY ROOT REDIRECT
# ===========================================================================
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send the homepage straight to the interactive API docs."""
    return RedirectResponse(url="/docs")


# ===========================================================================
# SECTION 5 — RUN THE SERVER (only when executed directly)
# ===========================================================================
# Port 8001 keeps this service separate from V1 (8000). host 127.0.0.1 means
# local machine only; use "0.0.0.0" to allow other machines to reach it.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )
