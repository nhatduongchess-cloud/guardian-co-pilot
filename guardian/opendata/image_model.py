"""
image_model.py
==============

THE EXPERIMENT THAT IS DESIGNED TO LOOK GOOD AND THEN FAIL.

Everything else in `guardian/opendata/` is careful about who the drivers are.
This module deliberately is not, because the public image datasets make care
impossible: `akahana/Driver-Drowsiness-Dataset` and
`n7i5x9/driver-drowsiness-dataset` ship **no participant id**. Their train/test
splits are splits of *frames*, so consecutive frames of the same face - often
milliseconds apart - land on both sides. A model trained that way is being
asked to recognise faces it has already seen, and it will score beautifully.

That number is not a lie, but it answers a question nobody has: *given a frame
from a session I already trained on, can I classify it?* Guardian needs the
other question: *given a driver I have never met, can I classify them?*

So this module runs the flattering experiment on purpose, and then runs the
only cross-population check the data allows: **train on one upload, test on the
other.** Different collection, different faces, same task. The gap between
those two numbers is the whole output. If in-dataset accuracy is high and
cross-dataset accuracy collapses toward chance, the model learned the people.

Every number this module emits is tagged `random_frame_split`. Nothing here is
comparable to the subject-disjoint results in `temporal.py`, and the report
says so at the top rather than in a footnote.

WHERE THIS RUNS
---------------
Not in the cloud container: Hugging Face is unreachable from there. This runs
on a machine that can reach the Hub and preferably has a GPU.

    python -m guardian.opendata.image_model --epochs 3

Weights are not committed; nothing downloaded here is redistributed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import sources

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "opendata"

IMAGE_SIZE = 160  # the crops are 227x227; 160 keeps the face and halves the work
BATCH_SIZE = 64
DEFAULT_EPOCHS = 3
DEFAULT_LR = 3e-4
#: Guardian's vocabulary, as integers for the loss.
CLASS_ORDER = ("alert", "drowsy")


@dataclass
class SplitScore:
    name: str
    n: int
    accuracy: float
    balanced_accuracy: float
    drowsy_recall: float
    alert_recall: float
    auc: Optional[float]


@dataclass
class ImageReport:
    trained_on: dict
    protocol: str
    in_dataset: dict
    cross_dataset: Optional[dict]
    collapse: Optional[dict]
    epochs: int
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_markdown(self) -> str:
        lines = [
            "# A face model on public driver images, and what it is worth",
            "",
            f"_Generated {self.generated_at}._",
            "",
            "> **Read this before the numbers.** The public image datasets carry no",
            "> participant id, so the split below is a split of **frames, not",
            "> drivers** - consecutive frames of the same face sit on both sides of",
            "> it. The in-dataset score is therefore not a cross-driver score and is",
            "> **not comparable** to anything in `rldd_report.md`. It is reported so",
            "> that the second number has something to fall from.",
            "",
            "## What was trained",
            "",
            f"- **Dataset**: `{self.trained_on['repo_id']}`",
            f"- **Licence**: {self.trained_on['licence']}",
            f"- **Protocol**: {self.protocol}",
            f"- **Epochs**: {self.epochs}",
            "",
            "## Result",
            "",
            "| split | n | accuracy | balanced acc. | drowsy recall | alert recall | AUC |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in [self.in_dataset] + ([self.cross_dataset] if self.cross_dataset else []):
            lines.append(
                f"| {row['name']} | {row['n']} | {row['accuracy']} "
                f"| {row['balanced_accuracy']} | {row['drowsy_recall']} "
                f"| {row['alert_recall']} | {row['auc']} |"
            )
        if self.collapse:
            lines += [
                "",
                "## The gap",
                "",
                f"Balanced accuracy falls **{self.collapse['balanced_accuracy_drop']:+}** "
                f"when the same weights meet a different upload of the same task "
                f"({self.in_dataset['balanced_accuracy']} to "
                f"{self.cross_dataset['balanced_accuracy']}), against a chance floor of "
                "0.5.",
                "",
                self.collapse["verdict"],
            ]
        lines.append("")
        return "\n".join(lines)


def _load_split(source, split: str, limit: Optional[int], seed: int):
    """Pull a split into memory as (uint8 images, int labels)."""
    from datasets import load_dataset

    data = load_dataset(source.repo_id, split=split)
    if limit is not None and limit < len(data):
        data = data.shuffle(seed=seed).select(range(limit))

    from PIL import Image

    images, labels = [], []
    for row in data:
        mapped = source.normalise(row[source.label_column])
        if mapped is None:
            continue
        image = row[source.image_column]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        images.append(np.asarray(image, dtype=np.uint8))
        labels.append(CLASS_ORDER.index(mapped))
    return np.stack(images), np.array(labels, dtype=np.int64)


def _to_tensor(images: np.ndarray, device):
    import torch

    # ImageNet statistics, because the backbone is an ImageNet backbone.
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float() / 255.0
    return (x - mean) / std


def _score(name: str, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> SplitScore:
    from .temporal import roc_auc

    drowsy = CLASS_ORDER.index("drowsy")
    alert = CLASS_ORDER.index("alert")
    recall_d = float((y_pred[y_true == drowsy] == drowsy).mean()) if (y_true == drowsy).any() else 0.0
    recall_a = float((y_pred[y_true == alert] == alert).mean()) if (y_true == alert).any() else 0.0
    auc = roc_auc(scores[y_true == drowsy], scores[y_true == alert])
    return SplitScore(
        name=name,
        n=int(len(y_true)),
        accuracy=round(float((y_true == y_pred).mean()), 4),
        # Balanced accuracy, because these uploads are not 50/50 and accuracy
        # alone would reward guessing the bigger class.
        balanced_accuracy=round((recall_d + recall_a) / 2, 4),
        drowsy_recall=round(recall_d, 4),
        alert_recall=round(recall_a, 4),
        auc=round(auc, 4) if auc is not None else None,
    )


def _predict(model, images: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    preds, scores = [], []
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            logits = model(_to_tensor(images[start : start + BATCH_SIZE], device))
            probability = torch.softmax(logits, dim=1)[:, CLASS_ORDER.index("drowsy")]
            preds.append(logits.argmax(dim=1).cpu().numpy())
            scores.append(probability.cpu().numpy())
    return np.concatenate(preds), np.concatenate(scores)


def run(
    train_key: str = "ddd",
    cross_key: Optional[str] = "n7",
    epochs: int = DEFAULT_EPOCHS,
    limit_train: Optional[int] = 12000,
    limit_eval: Optional[int] = 3000,
    seed: int = 42,
    out_dir: Optional[Path] = None,
) -> ImageReport:
    import torch
    from torch import nn
    from torchvision import models

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[image] device={device}")

    source = sources.get(train_key)
    print(f"[image] loading {source.repo_id} (licence: {source.licence})")
    train_x, train_y = _load_split(source, "train", limit_train, seed)
    test_x, test_y = _load_split(source, "test", limit_eval, seed)
    print(f"[image] train {train_x.shape} test {test_x.shape}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_ORDER))
    model = model.to(device)

    counts = np.bincount(train_y, minlength=len(CLASS_ORDER)).astype(float)
    weights = torch.tensor(
        (counts.sum() / np.maximum(counts, 1)).astype(np.float32), device=device
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=1e-4)

    generator = torch.Generator().manual_seed(seed)
    labels = torch.from_numpy(train_y).to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        running = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE].numpy()
            optimiser.zero_grad()
            loss = loss_fn(model(_to_tensor(train_x[batch], device)), labels[batch])
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(batch)
        print(f"[image] epoch {epoch}/{epochs} loss {running / len(order):.4f}")

    pred, score = _predict(model, test_x, device)
    in_dataset = _score(f"{train_key} test (random-frame split)", test_y, pred, score)
    print(f"[image] in-dataset balanced accuracy {in_dataset.balanced_accuracy}")

    cross = collapse = None
    if cross_key:
        other = sources.get(cross_key)
        print(f"[image] cross-dataset check on {other.repo_id}")
        cross_x, cross_y = _load_split(other, other.split, limit_eval, seed)
        cpred, cscore = _predict(model, cross_x, device)
        cross = _score(f"{cross_key} (never trained on)", cross_y, cpred, cscore)
        drop = round(cross.balanced_accuracy - in_dataset.balanced_accuracy, 4)
        collapse = {
            "balanced_accuracy_drop": drop,
            "verdict": (
                "The weights do not transfer. Most of the in-dataset score was "
                "the model recognising sessions it had already seen, which is "
                "exactly the failure this project documented on six drivers - "
                "reproduced here with thousands of faces, because more faces "
                "without a driver-aware split does not fix it."
                if cross.balanced_accuracy < in_dataset.balanced_accuracy - 0.15
                else "The weights hold up better than expected across uploads. "
                "That is worth noting, but it is still not a cross-driver "
                "result: neither upload identifies its participants, so the "
                "two may overlap in people for all anyone can tell."
            ),
        }
        print(f"[image] cross-dataset balanced accuracy {cross.balanced_accuracy} ({drop:+})")

    report = ImageReport(
        trained_on={
            "repo_id": source.repo_id,
            "licence": source.licence,
            "provenance": source.provenance,
        },
        protocol=(
            "random_frame_split - the dataset ships no participant id, so frames "
            "of the same face appear in both train and test. NOT a cross-driver "
            "protocol and not comparable to rldd_report.md."
        ),
        in_dataset=asdict(in_dataset),
        cross_dataset=asdict(cross) if cross else None,
        collapse=collapse,
        epochs=epochs,
    )

    out_dir = Path(out_dir or DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "image_model_report.json").write_text(report.to_json(), encoding="utf-8")
    (out_dir / "image_model_report.md").write_text(report.to_markdown(), encoding="utf-8")
    print(f"[image] wrote {out_dir / 'image_model_report.md'}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Face model on public driver images.")
    parser.add_argument("--train-key", default="ddd")
    parser.add_argument("--cross-key", default="n7")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--limit-train", type=int, default=12000)
    parser.add_argument("--limit-eval", type=int, default=3000)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run(
        args.train_key,
        args.cross_key or None,
        args.epochs,
        args.limit_train,
        args.limit_eval,
        out_dir=args.out_dir,
    )
    print()
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
