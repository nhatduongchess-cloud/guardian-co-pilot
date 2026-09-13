"""
personalization_client.py
==========================

VERTICAL 1 — PERSONALIZATION ENGINE
Layer: CLIENT (the "plug" other verticals use to call us)

This is the CROSS-VERTICAL bridge. Vertical 4 (the central brain) will hold a
PersonalizationClient inside its control loop and call clean Python methods
like `get_active_driver(...)`. All the messy HTTP details (URLs, JSON parsing,
error codes, timeouts) are hidden INSIDE this class.

This is the ADAPTER pattern: it adapts V1's HTTP/JSON world into V4's
plain-Python world.

WHY THIS MATTERS
-------------------------------
It proves Guardian is ONE integrated system, not four isolated apps. V4 can
consume V1 without knowing anything about FastAPI, routes, or JSON.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
# `httpx` -> a modern HTTP client library (like `requests`, but newer and
#            also usable in tests WITHOUT opening a real network port).
import httpx

# We reuse V1's data models so responses come back as real, validated objects.
# (Later, when we build V4, these shared shapes may move into a common
#  `contracts` package so both verticals import the SAME classes.)
from models.driver_profile import ActiveDriverProfile, DriverProfile
from models.driver_memory import DriverMemory
from models.explanation import ExplanationResponse, PersonalizedThreshold
from models.telemetry import EventType, TelemetryEvent


# ===========================================================================
# SECTION 1 — CLIENT-SIDE ERRORS
# ===========================================================================
# The caller (V4) should not see raw httpx errors. We translate every failure
# into these clear, named exceptions instead.
# ---------------------------------------------------------------------------


class PersonalizationClientError(Exception):
    """Base error for any problem talking to the Personalization Engine."""


class PersonalizationUnavailableError(PersonalizationClientError):
    """The service could not be reached (down, wrong URL, or timed out)."""


class DriverNotFoundError(PersonalizationClientError):
    """The requested driver id does not exist (HTTP 404 from the service)."""


# ===========================================================================
# SECTION 2 — THE CLIENT
# ===========================================================================


class PersonalizationClient:
    """
    A thin, safe wrapper around Vertical 1's REST API.

    Usage (as V4 will use it):
        client = PersonalizationClient("http://127.0.0.1:8000")
        profile = client.get_active_driver("driver_003")
        sensitivity = profile.preferences.warning_sensitivity  # ready to use

    Always close it when done, or use it as a context manager:
        with PersonalizationClient(url) as client:
            ...
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 5.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """
        Args:
            base_url:    where Vertical 1 is running (used in production).
            timeout:     seconds to wait before giving up on a request.
            http_client: OPTIONAL. Lets tests inject a pre-built httpx.Client
                         (e.g. FastAPI's TestClient) that runs the app IN
                         MEMORY, without opening a real network port. When we
                         inject one, `base_url`/`timeout` are ignored because
                         the injected client already carries its own settings.
        """
        # Remove any trailing slash so we can safely append paths.
        self._base_url = base_url.rstrip("/")

        # The versioned API prefix, matching the router in api/routes.py.
        self._api_prefix = "/api/v1"

        if http_client is not None:
            # Test / injected mode: use the client we were handed. We do NOT
            # own it, so we must not close it (the caller manages its life).
            self._http = http_client
            self._owns_http = False
        else:
            # Production mode: build our own reusable client (connection
            # pooling makes repeated calls in V4's control loop faster).
            self._http = httpx.Client(base_url=self._base_url, timeout=timeout)
            self._owns_http = True

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client, but only if WE created it."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "PersonalizationClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- internal helper -----------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        """
        Perform an HTTP request and convert low-level network failures into
        our clean PersonalizationUnavailableError.

        We do NOT check status codes here — each public method decides how to
        interpret 404 vs 200 for itself.
        """
        url = f"{self._api_prefix}{path}"
        try:
            return self._http.request(method, url, **kwargs)
        except httpx.RequestError as exc:
            # Covers connection refused, DNS failure, timeout, etc.
            raise PersonalizationUnavailableError(
                f"Could not reach Personalization Engine at {self._base_url}: {exc}"
            ) from exc

    # -- public methods ------------------------------------------------------

    def health(self) -> bool:
        """
        Return True if the Personalization Engine is alive and healthy.

        Never raises for a 'down' service — it simply returns False, which is
        exactly what a control loop wants when deciding whether to use V1.
        """
        try:
            response = self._request("GET", "/health")
        except PersonalizationUnavailableError:
            return False
        return response.status_code == 200

    def list_drivers(self) -> list[DriverProfile]:
        """Fetch all known drivers as validated DriverProfile objects."""
        response = self._request("GET", "/drivers")
        self._raise_for_unexpected(response)
        # Parse each JSON object in the list into a DriverProfile.
        return [DriverProfile.model_validate(item) for item in response.json()]

    def get_driver(self, driver_id: str) -> DriverProfile:
        """
        Fetch one driver by id.

        Raises:
            DriverNotFoundError: if the service returns 404.
        """
        response = self._request("GET", f"/drivers/{driver_id}")

        if response.status_code == 404:
            raise DriverNotFoundError(f"No driver with id '{driver_id}'.")
        self._raise_for_unexpected(response)

        return DriverProfile.model_validate(response.json())

    def create_or_update_driver(self, profile: DriverProfile) -> DriverProfile:
        """
        Create or update a driver on the service.

        We send the profile as JSON; the service applies its safety policy and
        returns the stored profile (which we parse back into an object).
        """
        response = self._request(
            "POST",
            "/drivers",
            # `mode="json"` makes enums/datetimes JSON-safe before sending.
            json=profile.model_dump(mode="json"),
        )
        self._raise_for_unexpected(response, allowed=(200, 201))
        return DriverProfile.model_validate(response.json())

    def get_active_driver(self, driver_id: str) -> ActiveDriverProfile:
        """
        THE CROSS-VERTICAL STAR METHOD.

        Activate a driver and return the published ActiveDriverProfile — the
        exact object Vertical 4 needs to configure the vehicle and set safety
        thresholds. The service's safety policy has already been applied.

        Raises:
            DriverNotFoundError: if the driver does not exist.
        """
        response = self._request("POST", f"/drivers/{driver_id}/activate")

        if response.status_code == 404:
            raise DriverNotFoundError(f"No driver with id '{driver_id}'.")
        self._raise_for_unexpected(response)

        return ActiveDriverProfile.model_validate(response.json())

    # =======================================================================
    # DRIVER MEMORY / REASONING (the §2.4 API surface)
    # =======================================================================
    # These wrap the memory_routes endpoints so V2/V4 can consume the Learning
    # Loop, advisory thresholds and Vietnamese explanations as plain Python.
    # -----------------------------------------------------------------------

    def ingest_telemetry(self, event: TelemetryEvent) -> TelemetryEvent:
        """Push one TelemetryEvent into the feature store. Returns it with its id."""
        response = self._request(
            "POST", "/telemetry", json=event.model_dump(mode="json")
        )
        self._raise_for_unexpected(response, allowed=(200, 201))
        return TelemetryEvent.model_validate(response.json())

    def end_trip(self, driver_id: str, trip_id: str) -> DriverMemory:
        """Trigger the Learning Loop for a finished trip; return updated memory."""
        response = self._request(
            "POST", f"/drivers/{driver_id}/trips/{trip_id}/end"
        )
        if response.status_code == 404:
            raise DriverNotFoundError(f"No events for trip '{trip_id}'.")
        self._raise_for_unexpected(response)
        return DriverMemory.model_validate(response.json())

    def get_personalized_threshold(
        self, driver_id: str, event_type: EventType | str
    ) -> PersonalizedThreshold:
        """
        THE METHOD V4 CALLS. Returns the advisory, safety-floored threshold.
        It is a suggestion only — V4 decides whether to use it (Constraint #1).
        """
        et = event_type.value if isinstance(event_type, EventType) else event_type
        response = self._request(
            "GET", f"/drivers/{driver_id}/threshold", params={"event_type": et}
        )
        self._raise_for_unexpected(response)
        return PersonalizedThreshold.model_validate(response.json())

    def get_explanation(self, event_id: str) -> ExplanationResponse:
        """THE METHOD V2 CALLS. Fetch the Vietnamese explanation for an event."""
        response = self._request("GET", f"/explanations/{event_id}")
        if response.status_code == 404:
            raise PersonalizationClientError(f"No event with id '{event_id}'.")
        self._raise_for_unexpected(response)
        return ExplanationResponse.model_validate(response.json())

    def explain_event(self, event: TelemetryEvent) -> ExplanationResponse:
        """Explain a live event object directly (no prior ingestion needed)."""
        response = self._request(
            "POST", "/explain", json=event.model_dump(mode="json")
        )
        self._raise_for_unexpected(response)
        return ExplanationResponse.model_validate(response.json())

    def get_memory(self, driver_id: str) -> DriverMemory:
        """Transparency: show everything Guardian has learned about a driver."""
        response = self._request("GET", f"/drivers/{driver_id}/memory")
        self._raise_for_unexpected(response)
        return DriverMemory.model_validate(response.json())

    def reset_memory(self, driver_id: str) -> DriverMemory:
        """Consent: wipe a driver's learned data (keeps consent + safety floor)."""
        response = self._request("DELETE", f"/drivers/{driver_id}/memory")
        self._raise_for_unexpected(response)
        return DriverMemory.model_validate(response.json())

    def set_consent(self, driver_id: str, enabled: bool) -> DriverMemory:
        """Consent: turn personalisation on/off for a driver."""
        response = self._request(
            "PUT", f"/drivers/{driver_id}/consent", params={"enabled": enabled}
        )
        self._raise_for_unexpected(response)
        return DriverMemory.model_validate(response.json())

    def reasoning_status(self) -> dict:
        """Report whether the real Phi-3 model or the fallback is active."""
        response = self._request("GET", "/reasoning/status")
        self._raise_for_unexpected(response)
        return response.json()

    # -- internal helper -----------------------------------------------------

    def _raise_for_unexpected(
        self, response: httpx.Response, allowed: tuple[int, ...] = (200,)
    ) -> None:
        """Raise a clear error if the status code is not one we expected."""
        if response.status_code not in allowed:
            raise PersonalizationClientError(
                f"Unexpected response {response.status_code} from "
                f"{response.request.url}: {response.text}"
            )


# ===========================================================================
# SECTION 3 — SELF-TEST (talks to a REAL running server if one is up)
# ===========================================================================
# Start the server first (python main.py), then run this file to see the
# client fetch live data. If no server is running, it reports that cleanly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    client = PersonalizationClient("http://127.0.0.1:8000")

    if not client.health():
        print("Personalization Engine is NOT running.")
        print("Start it first:  python main.py")
    else:
        print("Service is healthy. Fetching drivers...")
        for driver in client.list_drivers():
            print(f"  {driver.driver_id}: {driver.name}")

        print("\nActivating driver_003 (novice) via the client:")
        active = client.get_active_driver("driver_003")
        print(f"  name           : {active.name}")
        print(f"  warning level  : {active.preferences.warning_sensitivity.value}")
        print(f"  activated_at   : {active.activated_at}")

    client.close()
