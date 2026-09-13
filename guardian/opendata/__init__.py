"""
Guardian on open data.

Checks the shipped driver-state thresholds against public datasets containing
many more faces than the six the project was tuned on. Code only: datasets are
fetched at run time and never redistributed here.
"""

from .evaluate import FrameRecord, Report, Thresholds, evaluate, roc_auc, summarise
from .sources import REGISTRY, OpenDataset, get

__all__ = [
    "FrameRecord",
    "OpenDataset",
    "REGISTRY",
    "Report",
    "Thresholds",
    "evaluate",
    "get",
    "roc_auc",
    "summarise",
]
