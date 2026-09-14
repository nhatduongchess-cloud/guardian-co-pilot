"""
guardian.drowsiness
===================

Does the shipped driver-state engine actually catch a sleepy driver?

This package answers that question against labelled footage and draws the
answer, including the parts that do not flatter it. It runs from cached
feature CSVs with pandas and numpy only - no simulator, no deep-learning
stack, no private toolkit.
"""

from guardian.drowsiness.detect import (  # noqa: F401
    DEFAULT_FPS,
    Summary,
    Transition,
    TripResult,
    evaluate_trips,
    summarise,
)
from guardian.drowsiness.timeline import render_timeline  # noqa: F401

__all__ = [
    "DEFAULT_FPS",
    "Summary",
    "Transition",
    "TripResult",
    "evaluate_trips",
    "summarise",
    "render_timeline",
]
