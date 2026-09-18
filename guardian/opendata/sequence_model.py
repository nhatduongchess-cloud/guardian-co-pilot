"""
sequence_model.py
=================

THE LEARNED MODEL, GIVEN EVERY ADVANTAGE THE RULE DOES NOT HAVE.

`temporal.py` already compares four thresholds against gradient boosting and
logistic regression on sixty drivers. Both of those fitted models see each
window as a *bag* of numbers: boosting gets the five summary statistics, and
logistic regression gets the window flattened, which technically contains the
order but cannot use it, because a linear map over 120 positions has no notion
that position 7 comes before position 8.

Drowsiness is not a bag. It is a trajectory - blinks lengthen, the lid slows,
the gaps between them stretch. If ordering carries anything at all, a recurrent
model should find it, and none of the predictors so far could have. This module
gives that model its turn.

WHAT IS DIFFERENT HERE, AND WHY IT IS FAIR
------------------------------------------
Two things, and both cost the network rather than help it:

1.  **It gets a validation set, and pays for it.** A neural network needs to be
    told when to stop; four thresholds do not. That validation set has to be
    driver-disjoint from both the training and the test drivers, or early
    stopping quietly tunes on strangers. The release makes this possible
    exactly once: each fold's training array turns out to be precisely the
    union of the other four folds' test arrays, so one of those four can be
    lifted out whole as a validation group of unseen drivers. The network
    therefore trains on roughly three folds where boosting trained on four.
    That is a real handicap, it is the honest one, and it is reported.

2.  **It gets class weights.** The data is 19% alert, 40% low vigilant, 41%
    drowsy, and the metric is macro-F1. An unweighted cross-entropy optimises
    something else and would lose to the rule for a reason that has nothing to
    do with representation. Inverse-frequency weights let it optimise roughly
    what it is scored on.

Everything else is deliberately ordinary: one GRU layer, a concatenation of the
final hidden state with the mean over real blinks, dropout, a linear head.
About fifteen thousand parameters. The point is not to win with architecture -
it is to find out whether ordering carries signal that thresholds are throwing
away.

SCALING
-------
Robust, and fitted on training rows only. The blink frequency feature runs from
-108 to +705 with a standard deviation of 44; feeding that into a GRU alongside
amplitudes in the range of one guarantees the frequency channel dominates every
gate for the first several epochs. Median and IQR, then a hard clip at five
scaled units, keeps the heavy tail from setting the scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from . import rldd
from .rldd import Fold

#: Anything past this many scaled units is an outlier, not information. Clipping
#: rather than dropping keeps the row and its label.
CLIP = 5.0

DEFAULT_HIDDEN = 48
DEFAULT_EPOCHS = 60
DEFAULT_PATIENCE = 10
DEFAULT_BATCH = 128
DEFAULT_LR = 3e-3


@dataclass(frozen=True)
class TrainingConfig:
    hidden: int = DEFAULT_HIDDEN
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    batch_size: int = DEFAULT_BATCH
    lr: float = DEFAULT_LR
    dropout: float = 0.2
    weight_decay: float = 1e-4


# ---------------------------------------------------------------------------
# Scaling.
# ---------------------------------------------------------------------------

class RobustScaler:
    """Median and IQR per feature, computed over real blinks only."""

    def __init__(self) -> None:
        self.median: Optional[np.ndarray] = None
        self.scale: Optional[np.ndarray] = None

    def fit(self, windows: np.ndarray) -> "RobustScaler":
        mask = rldd.padding_mask(windows)
        flat = windows[mask]  # (n_real_blinks, 4)
        self.median = np.median(flat, axis=0)
        q75, q25 = np.percentile(flat, [75, 25], axis=0)
        iqr = q75 - q25
        # A feature with no spread would divide by zero; leave it alone instead.
        self.scale = np.where(iqr > 0, iqr, 1.0)
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        if self.median is None or self.scale is None:
            raise RuntimeError("scaler used before fit()")
        mask = rldd.padding_mask(windows)
        out = np.clip((windows - self.median) / self.scale, -CLIP, CLIP)
        # Padding stays exactly zero after scaling, so the mask the model is
        # handed keeps meaning what it meant before.
        out[~mask] = 0.0
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# The model.
# ---------------------------------------------------------------------------

def _build_model(config: TrainingConfig, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)

    class BlinkGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=len(rldd.BLINK_FEATURES),
                hidden_size=config.hidden,
                batch_first=True,
            )
            self.dropout = nn.Dropout(config.dropout)
            self.head = nn.Linear(config.hidden * 2, 3)

        def forward(self, x, mask):
            # x: (B, 30, 4)  mask: (B, 30) True where a real blink sits.
            out, _ = self.gru(x)
            lengths = mask.sum(dim=1).clamp(min=1)
            # The last REAL step, not the last slot: padding is leading, so for a
            # short sequence the final slot is real, but taking it blindly would
            # break the moment padding ever moved to the tail.
            idx = (mask.float().cumsum(dim=1) * mask.float()).argmax(dim=1)
            last = out[torch.arange(len(out)), idx]
            mean = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
            return self.head(self.dropout(torch.cat([last, mean], dim=1)))

    return BlinkGRU()


# ---------------------------------------------------------------------------
# The driver-disjoint validation split.
# ---------------------------------------------------------------------------

def split_validation_group(
    fold: Fold, folds: Sequence[Fold], validation_index: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Lift one other fold's drivers out of this fold's training array.

    Returns (train_x, train_y, val_x, val_y, validation_fold_index).

    The membership test is by exact window signature rather than by index,
    because nothing in the release promises an ordering and a silently wrong
    offset here would put the same drivers on both sides of early stopping -
    the exact failure this function exists to prevent.
    """
    others = [f for f in folds if f.index != fold.index]
    if not others:
        raise ValueError("need at least two folds to hold a validation group out")
    chosen = next(
        (f for f in others if f.index == validation_index),
        others[fold.index % len(others)],
    )
    val_signatures = {row.tobytes() for row in chosen.test_windows.reshape(chosen.n_test, -1)}

    flat = fold.train_windows.reshape(fold.n_train, -1)
    is_val = np.array([row.tobytes() in val_signatures for row in flat])
    if not is_val.any():
        raise ValueError(
            f"fold {fold.index}: no training row belongs to fold {chosen.index}'s "
            "test set, so a driver-disjoint validation split cannot be made"
        )
    return (
        fold.train_windows[~is_val],
        fold.train_labels[~is_val],
        fold.train_windows[is_val],
        fold.train_labels[is_val],
        chosen.index,
    )


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------

def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from .temporal import macro_f1

    return macro_f1(y_true, y_pred)


@dataclass
class TrainedFold:
    """One fitted network plus the paperwork needed to trust its number."""

    fold: int
    validation_fold: int
    n_train: int
    n_validation: int
    best_epoch: int
    best_validation_macro_f1: float
    epochs_run: int
    scaler: RobustScaler
    model: object

    def predict(self, windows: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(self.scaler.transform(windows))
        mask = torch.from_numpy(rldd.padding_mask(windows))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x, mask)
        classes = np.array([0, 5, 10])
        return classes[logits.argmax(dim=1).numpy()]


def train_fold(
    fold: Fold,
    folds: Sequence[Fold],
    config: Optional[TrainingConfig] = None,
    seed: int = 42,
    verbose: bool = True,
) -> TrainedFold:
    """Fit one network. The held-out fold is never touched, not even to peek."""
    import torch
    from torch import nn

    config = config or TrainingConfig()
    train_x, train_y, val_x, val_y, val_index = split_validation_group(fold, folds)

    scaler = RobustScaler().fit(train_x)
    xt = torch.from_numpy(scaler.transform(train_x))
    mt = torch.from_numpy(rldd.padding_mask(train_x))
    yt = torch.from_numpy((train_y // 5).astype(np.int64))
    xv = torch.from_numpy(scaler.transform(val_x))
    mv = torch.from_numpy(rldd.padding_mask(val_x))

    counts = np.bincount((train_y // 5).astype(int), minlength=3).astype(float)
    weights = torch.from_numpy((counts.sum() / np.maximum(counts, 1)).astype(np.float32))

    model = _build_model(config, seed)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    generator = torch.Generator().manual_seed(seed)
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(xt[batch], mt[batch]), yt[batch])
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            predicted = model(xv, mv).argmax(dim=1).numpy() * 5
        score = _macro_f1(val_y, predicted)

        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(
            f"[sequence] fold {fold.index}: trained on {len(train_x)} windows, "
            f"validated on fold {val_index} ({len(val_x)} windows), "
            f"best epoch {best_epoch} at val macro-F1 {best_score:.4f}"
        )

    return TrainedFold(
        fold=fold.index,
        validation_fold=val_index,
        n_train=len(train_x),
        n_validation=len(val_x),
        best_epoch=best_epoch,
        best_validation_macro_f1=round(best_score, 4),
        epochs_run=epoch,
        scaler=scaler,
        model=model,
    )


def available() -> bool:
    """True when torch is installed. The rest of the package must not need it."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True
