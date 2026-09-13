"""
Guardian's Slow Path: the sentence the driver hears.

The Fast Path decides in milliseconds and outputs a brake percentage. This
package turns that decision into a reason, in Vietnamese by default, because a
warning the driver does not understand is a warning they switch off.
"""

from guardian.explain.explainer import (
    SLOW_PATH_BUDGET_MS,
    Explanation,
    Factor,
    GuardianExplainer,
    explain,
)
from guardian.explain.vocabulary import EN, LANGUAGES, VI

__all__ = [
    "EN",
    "Explanation",
    "Factor",
    "GuardianExplainer",
    "LANGUAGES",
    "SLOW_PATH_BUDGET_MS",
    "VI",
    "explain",
]
