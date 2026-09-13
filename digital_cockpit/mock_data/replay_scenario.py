"""
replay_scenario.py
==================

A small DEMO TOOL for Vertical 2 (Digital Cockpit).

It reads a scenario file (cockpit_scenario.json) and PUSHES each step into a
running cockpit using CockpitClient, respecting each step's timestamp. This is
exactly what Vertical 4 will do live during a drive — here we replay a canned
script so a demo tells a clear story.

HOW TO RUN
----------
1. Start the cockpit in one terminal (from digital_cockpit/):
       python main.py
2. In another terminal (from digital_cockpit/):
       python mock_data/replay_scenario.py
   Add --fast to skip the waits (instant replay for a quick check):
       python mock_data/replay_scenario.py --fast

Open http://127.0.0.1:8001/api/v1/cockpit/state while it runs to watch the
screen state change.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import argparse
import json
import time
from pathlib import Path

from client.cockpit_client import CockpitClient, CockpitUnavailableError


# The scenario file lives next to this script.
_SCENARIO_PATH = Path(__file__).parent / "cockpit_scenario.json"


def _dispatch(client: CockpitClient, step: dict) -> str:
    """
    Execute one scenario step against the cockpit and return a short summary
    line to print. We read the 'action' field and call the matching client
    method.
    """
    action = step["action"]

    if action == "set_driver":
        client.set_driver(step["driver_name"], theme=step["theme"])
        return f"driver = {step['driver_name']} (theme {step['theme']})"

    if action == "telemetry":
        client.update_telemetry(
            speed_kmh=step.get("speed_kmh"),
            gear=step.get("gear"),
        )
        return f"telemetry speed={step.get('speed_kmh')} gear={step.get('gear')}"

    if action == "alert":
        client.show_alert(
            alert_id=step["alert_id"],
            severity=step["severity"],
            message=step["message"],
            source=step["source"],
        )
        return f"ALERT [{step['severity']}] {step['message']}"

    if action == "dismiss":
        removed = client.dismiss_alert(step["alert_id"])
        return f"dismiss {step['alert_id']} -> {removed}"

    # Unknown action -> fail loudly rather than silently skipping.
    raise ValueError(f"Unknown scenario action: {action!r}")


def replay(url: str, fast: bool) -> None:
    """Load the scenario and push every step, honouring the timeline."""
    scenario = json.loads(_SCENARIO_PATH.read_text(encoding="utf-8"))
    steps = scenario["steps"]

    client = CockpitClient(url)

    # Fail early with a friendly message if the cockpit is not up.
    if not client.health():
        print(f"Cockpit not reachable at {url}. Start it first: python main.py")
        client.close()
        return

    print(f"Replaying '{scenario['name']}' ({len(steps)} steps)"
          f"{' [FAST]' if fast else ''}\n")

    previous_t = 0.0
    try:
        for step in steps:
            # Wait the gap between this step and the previous one (unless fast).
            if not fast:
                gap = step["t"] - previous_t
                if gap > 0:
                    time.sleep(gap)
            previous_t = step["t"]

            summary = _dispatch(client, step)
            print(f"  t={step['t']:>4.1f}s  {summary}")
    except CockpitUnavailableError as exc:
        print(f"\nLost connection to cockpit: {exc}")
    finally:
        client.close()

    print("\nDone. Final state:")
    # Reconnect briefly to print the final screen (best-effort).
    try:
        with CockpitClient(url) as c:
            state = c.get_state()
            print(f"  driver={state.driver_name} theme={state.theme.value} "
                  f"speed={state.telemetry.speed_kmh} gear={state.telemetry.gear.value}")
            for a in state.alerts:
                print(f"  [{a.severity.value:8s}] {a.message}")
    except CockpitUnavailableError:
        pass


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a cockpit scenario.")
    parser.add_argument(
        "--url", default="http://127.0.0.1:8001",
        help="Base URL of the running cockpit service.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip the timeline waits and push all steps instantly.",
    )
    args = parser.parse_args()

    replay(url=args.url, fast=args.fast)
