"""Smoke test for the MedKnow training loop (CPU, tiny synthetic data)."""

import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medknow.config import load_config
from medknow.datasets.chest_xray import (
    base_transform,
    build_internal_dataset,
    make_loader,
)
from medknow.models.factory import create_resnet18
from medknow.training.trainer import train_model


def _make_tiny_dataset(root: Path, n: int = 8, size: int = 32) -> None:
    for cls in ("NORMAL", "PNEUMONIA"):
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.new("RGB", (size, size), ((i * 40) % 255, 60, 120)).save(
                d / f"{cls}_{i}.png"
            )


def test_train_smoke(tmp_path):
    split = tmp_path / "split"
    _make_tiny_dataset(split / "train", n=8)
    _make_tiny_dataset(split / "val", n=4)

    cfg = load_config(None)
    cfg["training"] = {
        "seed": 42,
        "epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 4,
        "use_amp": False,
        "early_stopping": {
            "enabled": True,
            "patience": 5,
            "monitor": "val_loss",
            "min_delta": 0.0,
        },
        "lr_scheduler": {
            "enabled": True,
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 3,
            "min_lr": 1e-6,
        },
    }

    train_loader = make_loader(
        build_internal_dataset(split, "train", base_transform(32)),
        batch_size=4, shuffle=True, num_workers=0,
    )
    val_loader = make_loader(
        build_internal_dataset(split, "val", base_transform(32)),
        batch_size=4, shuffle=False, num_workers=0,
    )
    model = create_resnet18(
        num_classes=2, pretrained=False, freeze_backbone=False, dropout_rate=0.3
    )
    ckpt = tmp_path / "best.pth"

    summary = train_model(
        model, train_loader, val_loader, cfg, ckpt, torch.device("cpu")
    )

    assert ckpt.exists()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in state
    assert "best_val_loss" in summary
    assert len(summary["history"]) == 1
