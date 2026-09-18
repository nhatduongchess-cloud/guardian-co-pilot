"""
Guardian on open data.

Checks the shipped driver-state thresholds against public datasets containing
many more faces than the six the project was tuned on. Code only: datasets are
fetched at run time and never redistributed here.
"""

from . import rldd, temporal
from .evaluate import FrameRecord, Report, Thresholds, evaluate, roc_auc, summarise
from .rldd import Fold, load_folds
from .sources import REGISTRY, OpenDataset, get
from .temporal import BlinkRule, TemporalReport, cross_validate, reduce_windows

__all__ = [
    "BlinkRule",
    "Fold",
    "FrameRecord",
    "OpenDataset",
    "REGISTRY",
    "Report",
    "TemporalReport",
    "Thresholds",
    "cross_validate",
    "evaluate",
    "get",
    "load_folds",
    "reduce_windows",
    "rldd",
    "roc_auc",
    "summarise",
    "temporal",
]
