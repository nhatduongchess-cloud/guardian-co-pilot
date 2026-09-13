"""
cockpit_client.py
=================

VERTICAL 2 — DIGITAL COCKPIT
Layer: CLIENT (the cross-vertical "plug" V4 uses to DRIVE the screen)

Where PersonalizationClient (V1) mostly PULLS data ("who is driving?"), this
client mostly PUSHES commands ("show this alert", "update the speed"). V4's
control loop holds a CockpitClient and calls these clean Python methods; all
HTTP/JSON details are hidden inside.

This is the ADAPTER pattern again, pointed the other way: it adapts V4's
plain-Python intentions into V2's HTTP world.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
from typing import Optional

import httpx

from models.cockpit_state import (
    Alert,
    AlertSeverity,
    CockpitState,
    CockpitTheme,
    GearPosition,
)


# ===========================================================================
# SECTION 1 — CLIENT-SIDE ERRORS
# ===========================================================================


class CockpitClientError(Exception):
    """Base error for any problem talking to the Digital Cockpit."""


class CockpitUnavailableError(CockpitClientError):
    """The cockpit service could not be reached (down, wrong URL, timeout)."""


# ===========================================================================
# SECTION 2 — THE CLIENT
# ===========================================================================


class CockpitClient:
    """
    A thin, safe wrapper around Vertical 2's REST API.

    Usage (as V4 will use it):
        cockpit = CockpitClient("http://127.0.0.1:8001")
        cockpit.set_driver("Lan", theme="guided")
        cockpit.update_telemetry(speed_kmh=42.0, gear="D")
        cockpit.show_alert("collision_front", "critical",
                           "Obstacle ahead", source="safety_kernel")
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        timeout: float = 5.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """
        Args:
            base_url:    where Vertical 2 runs (production default: 8001).
            timeout:     seconds before giving up on a request.
            http_client: OPTIONAL. Inject a pre-built client (e.g. FastAPI's
                         TestClient) so tests run in memory without a network
                         port. When injected, base_url/timeout are ignored.
        """
        self._base_url = base_url.rstrip("/")
        self._api_prefix = "/api/v1"

        if http_client is not None:
            self._http = http_client
            self._owns_http = False  # caller owns it -> we must not close it
        else:
            self._http = httpx.Client(base_url=self._base_url, timeout=timeout)
            self._owns_http = True

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client, only if WE created it."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "CockpitClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- internal helper -----------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """Perform a request, converting network failures into a clean error."""
        url = f"{self._api_prefix}{path}"
        try:
            return self._http.request(method, url, **kwargs)
        except httpx.RequestError as exc:
            raise CockpitUnavailableError(
                f"Could not reach Digital Cockpit at {self._base_url}: {exc}"
            ) from exc

    def _raise_for_unexpected(
        self, response: httpx.Response, allowed: tuple[int, ...] = (200,)
    ) -> None:
        """Raise a clear error if the status code is not one we expected."""
        if response.status_code not in allowed:
            raise CockpitClientError(
                f"Unexpected response {response.status_code} from "
                f"{response.request.url}: {response.text}"
            )

    # -- public methods ------------------------------------------------------

    def health(self) -> bool:
        """Return True if the cockpit is alive. Never raises for a down service."""
        try:
            response = self._request("GET", "/health")
        except CockpitUnavailableError:
            return False
        return response.status_code == 200

    def get_state(self) -> CockpitState:
        """Fetch the full current cockpit state as a validated object."""
        response = self._request("GET", "/cockpit/state")
        self._raise_for_unexpected(response)
        return CockpitState.model_validate(response.json())

    def set_driver(
        self,
        driver_name: str,
        theme: CockpitTheme | str = CockpitTheme.MINIMAL,
    ) -> CockpitState:
        """
        Configure the cockpit for the active driver.

        `theme` accepts the raw ui_theme string from V1 (e.g. "guided"); we
        send its value and the service validates it.
        """
        # Normalise the theme to its string value for the JSON body.
        theme_value = theme.value if isinstance(theme, CockpitTheme) else theme

        response = self._request(
            "POST",
            "/cockpit/driver",
            json={"driver_name": driver_name, "theme": theme_value},
        )
        self._raise_for_unexpected(response)
        return CockpitState.model_validate(response.json())

    def update_telemetry(
        self,
        speed_kmh: Optional[float] = None,
        gear: Optional[GearPosition | str] = None,
    ) -> CockpitState:
        """Partially update live numbers (speed, gear)."""
        body: dict = {}
        if speed_kmh is not None:
            body["speed_kmh"] = speed_kmh
        if gear is not None:
            body["gear"] = gear.value if isinstance(gear, GearPosition) else gear

        response = self._request("POST", "/cockpit/telemetry", json=body)
        self._raise_for_unexpected(response)
        return CockpitState.model_validate(response.json())

    def show_alert(
        self,
        alert_id: str,
        severity: AlertSeverity | str,
        message: str,
        source: str,
    ) -> CockpitState:
        """
        Push (or update) an alert on the screen.

        Convenience wrapper: builds the Alert for you so V4 does not need to
        import the Alert model. Reusing the same alert_id UPDATES the existing
        alert instead of duplicating it.
        """
        # Build + validate the Alert locally before sending.
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            message=message,
            source=source,
        )
        response = self._request(
            "POST", "/cockpit/alerts", json=alert.model_dump(mode="json")
        )
        self._raise_for_unexpected(response)
        return CockpitState.model_validate(response.json())

    def dismiss_alert(self, alert_id: str) -> bool:
        """
        Remove an alert (the hazard has cleared).

        Returns:
            True  if the alert existed and was removed,
            False if there was no such alert (service returned 404).

        We treat "already gone" as a non-error: for a control loop, dismissing
        a hazard that is no longer shown is perfectly fine.
        """
        response = self._request("POST", f"/cockpit/alerts/{alert_id}/dismiss")

        if response.status_code == 404:
            return False
        self._raise_for_unexpected(response)
        return True


# ===========================================================================
# SECTION 3 — SELF-TEST (talks to a REAL running cockpit if one is up)
# ===========================================================================
# Start the cockpit first (python main.py), then run this file.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    client = CockpitClient("http://127.0.0.1:8001")

    if not client.health():
        print("Digital Cockpit is NOT running.")
        print("Start it first:  python main.py")
    else:
        print("Cockpit is healthy. Simulating what V4 would push:")
        client.set_driver("Lan (Teen)", theme="guided")
        client.update_telemetry(speed_kmh=42.0, gear="D")
        client.show_alert("eco", "info", "Eco mode on", source="cockpit")
        client.show_alert(
            "collision_front", "critical", "Obstacle ahead - brake!",
            source="safety_kernel",
        )

        state = client.get_state()
        print(f"\nDriver: {state.driver_name} | Theme: {state.theme.value} | "
              f"Speed: {state.telemetry.speed_kmh}")
        print("Alerts (critical first):")
        for a in state.alerts:
            print(f"  [{a.severity.value:8s}] {a.message}")

        removed = client.dismiss_alert("collision_front")
        print(f"\nDismissed collision_front: {removed}")

    client.close()
