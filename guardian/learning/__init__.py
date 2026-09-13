"""
Guardian's learning loop: the threshold moves toward the person driving.

Bounded by design - personalisation tunes WHEN to warn, within a hard safety
floor and ceiling it can never cross, and has no route to the brake command.
"""

from guardian.learning.loop import (
    DEFAULT_PERCLOS_THRESHOLD,
    DriverProfile,
    LearningLoop,
    SafetyBounds,
    TripOutcome,
    summarise,
)

__all__ = [
    "DEFAULT_PERCLOS_THRESHOLD",
    "DriverProfile",
    "LearningLoop",
    "SafetyBounds",
    "TripOutcome",
    "summarise",
]
