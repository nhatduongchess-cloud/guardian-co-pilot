"""
labels.py
=========

THE LABEL CONTRACT, on its own.

These five names and their fixed ordering are the only thing the rule engine,
the CNN ablation and the evaluation reports have to agree on. They used to live
in `dataset.py`, which imports torch, OpenCV and the organiser's private trip
toolkit - so importing five strings pulled in a deep-learning stack and a
dependency most people running this repo do not have.

Keeping the contract here is what lets the drowsiness demo run from a clean
checkout with pandas and nothing else.
"""

from __future__ import annotations

#: Fixed ordering. Index positions are written into saved models and reports,
#: so append to the end - never reorder.
CLASS_NAMES: tuple[str, ...] = (
    "alert",
    "drowsy",
    "yawning",
    "distracted",
    "microsleep",
)

CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)

#: The states that mean "this driver is not fit to be driving right now".
#: `distracted` is deliberately excluded: eyes-off-road is a different failure
#: from falling asleep, and conflating them hides which one the system caught.
IMPAIRED_STATES: frozenset[str] = frozenset({"drowsy", "yawning", "microsleep"})
