"""
model.py
========

GUARDIAN CO-PILOT — CHALLENGE 2 (Driver Intelligence)
Model definition: an ImageNet-pretrained backbone plus a 5-class head.

WHY A SEPARATE FILE
-------------------
Training loops and model architecture change for different reasons. Keeping
them apart means we can swap ResNet18 -> MobileNetV3 with one argument and
never touch the training code (Single Responsibility, again).

BACKBONE CHOICE (a judge WILL ask this)
---------------------------------------
Measured on an RTX 4070 Laptop, batch of 32 at 224x224, with our 5-class head:

  resnet18            11.18M params   ~1990 img/s   strong + fastest here [default]
  mobilenet_v3_large   2.98M params    ~530 img/s   light, for on-vehicle edge
  mobilenet_v3_small   0.93M params    ~890 img/s   smallest, latency demos
  efficientnet_b0      ~4.0M params        —        accuracy/size balance

Counter-intuitive but important: MobileNet has 12x fewer parameters yet runs
SLOWER on this GPU. Depthwise convolutions are optimised for phone CPUs/NPUs
and are memory-bandwidth-bound on a GPU. "Fewer parameters" does not mean
"faster" — it depends on the hardware. We use resnet18 for the project (GPU)
and cite MobileNet for the edge-deployment story, stating both honestly.

All of these come from torchvision under a **BSD-3 licence** — permissive and
commercially safe. That matters: YOLOv8 (used later in Challenge 1) is AGPL-3.0,
so Challenge 2's stack stays clean for any real deployment story.

CHECKPOINTS CARRY THEIR LABELS
------------------------------
A model that outputs index 3 is useless if we forget index 3 meant
"distracted". So `save_checkpoint` stores CLASS_NAMES and the architecture
alongside the weights, and `load_checkpoint` restores them together. This
removes a whole family of silent, score-destroying bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from guardian.challenge2.dataset import CLASS_NAMES, NUM_CLASSES


# ===========================================================================
# SECTION 1 — SUPPORTED ARCHITECTURES
# ===========================================================================
# Each entry knows how to build the backbone and how many features its final
# layer produces, so the head can be sized correctly.
# ---------------------------------------------------------------------------
SUPPORTED_ARCHS: tuple[str, ...] = (
    "resnet18",
    "resnet34",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "efficientnet_b0",
)


def _build_backbone(arch: str, pretrained: bool) -> tuple[nn.Module, int]:
    """
    Create a backbone with its classifier removed.

    Returns:
        (backbone, n_features) — the feature extractor and its output width.
    """
    from torchvision import models

    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unsupported arch {arch!r}. Choose from {SUPPORTED_ARCHS}.")

    # `weights="DEFAULT"` pulls the best available ImageNet weights; None means
    # random init (useful for ablation: "does pretraining actually help?").
    weights = "DEFAULT" if pretrained else None

    if arch.startswith("resnet"):
        net = getattr(models, arch)(weights=weights)
        n_features = net.fc.in_features
        net.fc = nn.Identity()          # strip the ImageNet classifier
        return net, n_features

    if arch.startswith("mobilenet_v3"):
        net = getattr(models, arch)(weights=weights)
        # MobileNetV3's classifier is a small MLP; its first Linear tells us
        # the feature width coming out of the pooled backbone.
        n_features = net.classifier[0].in_features
        net.classifier = nn.Identity()
        return net, n_features

    # efficientnet_b0
    net = models.efficientnet_b0(weights=weights)
    n_features = net.classifier[1].in_features
    net.classifier = nn.Identity()
    return net, n_features


# ===========================================================================
# SECTION 2 — THE MODEL
# ===========================================================================


class DriverStateNet(nn.Module):
    """
    Backbone + classification head for the 5 driver states.

    Optionally adds a second head regressing `alertness` (0..1) from the SAME
    features. Multi-task learning can regularise the classifier — but it needs
    alertness targets from the dataset, so it is OFF by default until the data
    layer supplies them.
    """

    def __init__(
        self,
        arch: str = "resnet18",
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.2,
        alertness_head: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.arch = arch
        self.num_classes = num_classes
        self.has_alertness_head = alertness_head

        self.backbone, n_features = _build_backbone(arch, pretrained)

        # Optionally freeze the feature extractor (train the head only).
        # Useful when data is tiny; here we usually fine-tune everything.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Dropout before the classifier fights overfitting — important given
        # our small, highly-correlated dataset (600 near-identical frames per
        # trip at 20 FPS).
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, num_classes),
        )

        # Auxiliary regression head (sigmoid keeps it in 0..1).
        self.alertness = (
            nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, 1), nn.Sigmoid())
            if alertness_head
            else None
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: image batch (B, 3, H, W), already normalised.

        Returns:
            dict with 'logits' (B, num_classes) and, if enabled,
            'alertness' (B,). Returning a dict keeps the API stable when heads
            are added later — callers index by name, not position.
        """
        features = self.backbone(x)
        out: dict[str, torch.Tensor] = {"logits": self.classifier(features)}
        if self.alertness is not None:
            out["alertness"] = self.alertness(features).squeeze(-1)
        return out

    @torch.no_grad()
    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return predicted class indices for a batch."""
        self.eval()
        return self.forward(x)["logits"].argmax(dim=1)


# ===========================================================================
# SECTION 3 — FACTORY + DEVICE HELPERS
# ===========================================================================


def build_model(
    arch: str = "resnet18",
    pretrained: bool = True,
    dropout: float = 0.2,
    alertness_head: bool = False,
    freeze_backbone: bool = False,
) -> DriverStateNet:
    """Create a DriverStateNet. One place to change architecture."""
    return DriverStateNet(
        arch=arch,
        num_classes=NUM_CLASSES,
        pretrained=pretrained,
        dropout=dropout,
        alertness_head=alertness_head,
        freeze_backbone=freeze_backbone,
    )


def get_device() -> torch.device:
    """Return CUDA when available, else CPU. Single place to decide."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters — handy for the 'edge-ready' argument in the pitch."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


# ===========================================================================
# SECTION 4 — CHECKPOINTS (weights travel WITH their label ordering)
# ===========================================================================


@dataclass
class CheckpointMeta:
    """Everything needed to rebuild and correctly interpret a model."""

    arch: str
    class_names: tuple[str, ...]
    img_size: int
    alertness_head: bool
    notes: str = ""


def save_checkpoint(
    model: DriverStateNet,
    path: str | Path,
    img_size: int = 224,
    notes: str = "",
) -> Path:
    """Save weights plus the metadata required to use them safely."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {
                "arch": model.arch,
                # Freeze the label order INTO the artefact.
                "class_names": list(CLASS_NAMES),
                "img_size": img_size,
                "alertness_head": model.has_alertness_head,
                "notes": notes,
            },
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path, device: Optional[torch.device] = None
) -> tuple[DriverStateNet, CheckpointMeta]:
    """
    Rebuild a model from a checkpoint and verify its labels match ours.

    Raises:
        ValueError: if the checkpoint's class ordering differs from the current
                    CLASS_NAMES — that mismatch would silently mislabel every
                    prediction, so we refuse to continue.
    """
    path = Path(path)
    device = device or get_device()

    # weights_only=True is the safe default for loading untrusted files; our
    # payload is plain tensors + a small dict, so it works here.
    blob = torch.load(path, map_location=device, weights_only=True)
    meta_dict = blob["meta"]

    saved_classes = tuple(meta_dict["class_names"])
    if saved_classes != CLASS_NAMES:
        raise ValueError(
            f"Checkpoint class order {saved_classes} does not match current "
            f"{CLASS_NAMES}. Refusing to load — predictions would be mislabelled."
        )

    model = build_model(
        arch=meta_dict["arch"],
        pretrained=False,  # weights come from the checkpoint
        alertness_head=meta_dict.get("alertness_head", False),
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()

    meta = CheckpointMeta(
        arch=meta_dict["arch"],
        class_names=saved_classes,
        img_size=int(meta_dict.get("img_size", 224)),
        alertness_head=bool(meta_dict.get("alertness_head", False)),
        notes=meta_dict.get("notes", ""),
    )
    return model, meta


# ===========================================================================
# SECTION 5 — SELF-TEST
# ===========================================================================
if __name__ == "__main__":
    import time

    device = get_device()
    print(f"Device: {device}")

    for arch in ("resnet18", "mobilenet_v3_large", "mobilenet_v3_small"):
        model = build_model(arch=arch, pretrained=True).to(device)
        n_params = count_parameters(model)

        # Time a forward pass on a realistic batch.
        batch = torch.randn(32, 3, 224, 224, device=device)
        model.eval()
        with torch.no_grad():
            model(batch)  # warm-up (CUDA kernels compile on first call)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0

        fps = 32 / dt
        print(f"  {arch:20s} params={n_params/1e6:5.2f}M  "
              f"logits={tuple(out['logits'].shape)}  "
              f"batch32={dt*1000:6.1f}ms  ({fps:,.0f} img/s)")

    # Round-trip a checkpoint to prove labels travel with the weights.
    print("\nCheckpoint round-trip:")
    model = build_model(arch="resnet18", pretrained=False)
    ckpt = save_checkpoint(model, "artifacts/_selftest.pt", notes="self-test")
    restored, meta = load_checkpoint(ckpt)
    print(f"  saved + restored: arch={meta.arch} classes={meta.class_names}")
    Path(ckpt).unlink()  # tidy up
    print("  temp checkpoint removed")
