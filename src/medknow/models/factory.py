"""ResNet-18 model factory for MedKnow V1.

The architecture matches the manuscript exactly: ImageNet-pretrained ResNet-18
with the convolutional backbone frozen, the final residual block (layer4) and
the classification head fine-tuned, and a dropout layer inside the head (used
for MC Dropout at inference).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models


def list_available_models() -> dict[str, dict[str, str]]:
    """Return the architectures supported by the formal ``medknow`` package."""
    return {
        "resnet18": {
            "name": "ResNet-18",
            "input_size": "224",
            "description": "ImageNet-pretrained ResNet-18 with a dropout head",
        }
    }


def create_resnet18(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """Create the manuscript ResNet-18 classifier.

    Args:
        num_classes: Number of output classes (2 for pneumonia/normal).
        pretrained: Load ImageNet weights (requires network on first call).
        freeze_backbone: Freeze all parameters except layer4 and the head.
        dropout_rate: Dropout probability in the classification head.

    Returns:
        The model (not moved to a device, not in eval mode).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def get_target_layer(model: nn.Module) -> nn.Module:
    """Grad-CAM target layer for ResNet-18 (last block of layer4)."""
    return model.layer4[-1]


def enable_dropout(model: nn.Module) -> None:
    """Put dropout modules in train mode for MC Dropout inference."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


def load_trained_model(
    model_path: str | Path,
    num_classes: int = 2,
    device: torch.device | None = None,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """Load a trained checkpoint into an eval-mode ResNet-18.

    Accepts both raw state dicts and checkpoint dicts containing a
    ``model_state_dict`` key (the format saved by ``scripts/train.py`` and by
    ``medknow.training.trainer``).

    Args:
        model_path: Path to the checkpoint (.pth).
        num_classes: Number of output classes.
        device: Target device (defaults to CPU).
        dropout_rate: Head dropout rate used when building the architecture.

    Returns:
        The loaded model in eval mode on ``device``.

    Raises:
        FileNotFoundError: if the checkpoint does not exist.
        RuntimeError: if the state dict does not match the architecture.
    """
    device = device or torch.device("cpu")
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = create_resnet18(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
        dropout_rate=dropout_rate,
    )
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model
