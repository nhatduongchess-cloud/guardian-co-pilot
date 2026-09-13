"""
models package — the shared vocabulary of the Personalization Engine.

Re-exporting here lets other layers write:
    from models import DriverProfile, TelemetryEvent, DriverMemory
instead of the longer per-module paths.

The vocabulary is grouped into three contracts:
    * driver_profile.py  -> WHO the driver is + comfort/UI preferences
    * telemetry.py       -> INPUT events from V3/V4 (§3.1)
    * driver_memory.py   -> LEARNED behaviour + state machine (§3.2)
    * explanation.py     -> reasoning + threshold contracts (§3.3, §3.4)
"""

from models.driver_profile import (
    ActiveDriverProfile,
    DriverProfile,
    DrivingProfile,
    ExperienceLevel,
    Preferences,
    UITheme,
    WarningSensitivity,
)
from models.telemetry import (
    EventSource,
    EventType,
    KernelAction,
    SafetyKernelDecision,
    ScalarFeatures,
    TelemetryEvent,
)
from models.driver_memory import (
    Baseline,
    Consent,
    DriverMemory,
    MemorySnippet,
    PersonalizationState,
    SafetyFloor,
)
from models.explanation import (
    ExplanationContext,
    ExplanationResponse,
    PersonalizedThreshold,
)

# `__all__` declares the official public names of this package.
__all__ = [
    # driver_profile
    "ActiveDriverProfile",
    "DriverProfile",
    "DrivingProfile",
    "ExperienceLevel",
    "Preferences",
    "UITheme",
    "WarningSensitivity",
    # telemetry (§3.1)
    "EventSource",
    "EventType",
    "KernelAction",
    "SafetyKernelDecision",
    "ScalarFeatures",
    "TelemetryEvent",
    # driver_memory (§3.2)
    "Baseline",
    "Consent",
    "DriverMemory",
    "MemorySnippet",
    "PersonalizationState",
    "SafetyFloor",
    # explanation (§3.3, §3.4)
    "ExplanationContext",
    "ExplanationResponse",
    "PersonalizedThreshold",
]
