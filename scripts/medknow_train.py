#!/usr/bin/env python3
"""MedKnow — train the baseline ResNet-18 classifier.

Usage:
    python scripts/medknow_train.py --config configs/baseline.yaml --seed 42
    python scripts/medknow_train.py --seed 2024 --epochs 5 --no_amp

The training protocol matches the manuscript: frozen backbone + layer4/head
fine-tuning, AdamW, early stopping on validation loss, AMP on CUDA, and a
patient-level split loaded from ``data/split_patient/``.
"""

import argparse
import json
import logging

logger = logging.getLogger(__name__)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from medknow.config import get, load_config
from medknow.datasets.chest_xray import (
    base_transform,
    build_internal_dataset,
    make_loader,
    train_transform,
)
from medknow.models.factory import create_resnet18
from medknow.training.trainer import train_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the MedKnow baseline ResNet-18")
    ap.add_argument("--config", default="configs/baseline.yaml", help="YAML config")
    ap.add_argument("--seed", type=int, default=None, help="override training.seed")
    ap.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    ap.add_argument(
        "--data_dir", default=None,
        help="patient-level split dir (default: data/split_patient)",
    )
    ap.add_argument(
        "--out", default=None,
        help="checkpoint path (default: checkpoints/medknow_seed_{seed}.pth)",
    )
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.no_amp:
        cfg["training"]["use_amp"] = False

    seed = int(get(cfg, "training.seed", 42))
    split_dir = args.data_dir or get(cfg, "paths.split_dir")
    image_size = int(get(cfg, "data.image_size"))
    batch_size = int(get(cfg, "data.batch_size"))
    num_workers = int(get(cfg, "data.num_workers"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = args.out or str(
        Path(get(cfg, "paths.checkpoint_dir")) / f"medknow_seed_{seed}.pth"
    )

    train_tf = train_transform(
        image_size=image_size,
        rotation_deg=float(get(cfg, "training.augmentation.random_rotation_deg", 10)),
        brightness=float(get(cfg, "training.augmentation.brightness", 0.2)),
        contrast=float(get(cfg, "training.augmentation.contrast", 0.1)),
        affine_translation_frac=float(
            get(cfg, "training.augmentation.affine_translation_frac", 0.05)
        ),
    )
    val_tf = base_transform(image_size)

    train_loader = make_loader(
        build_internal_dataset(split_dir, "train", train_tf),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = make_loader(
        build_internal_dataset(split_dir, "val", val_tf),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    model = create_resnet18(
        num_classes=int(get(cfg, "model.num_classes", 2)),
        pretrained=bool(get(cfg, "model.pretrained", True)),
        freeze_backbone=bool(get(cfg, "model.freeze_backbone", True)),
        dropout_rate=float(get(cfg, "model.dropout_rate", 0.3)),
    )

    summary = train_model(
        model, train_loader, val_loader, cfg, out_path, device
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
