"""Reproducible training for the MedKnow baseline model.

Training settings mirror the manuscript: AdamW (lr 1e-4, weight decay 1e-5),
batch size 16, early stopping on validation loss (patience 5, min delta 1e-3),
ReduceLROnPlateau (factor 0.5, patience 3), and automatic mixed precision on
CUDA. Random seeds are set before training for reproducibility.
"""

from __future__ import annotations

import logging
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from medknow.config import get

logger = logging.getLogger(__name__)


def _checkpoint_metadata(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Build portable metadata so a checkpoint is not detached from its protocol."""
    return {
        "checkpoint_format": 1,
        "model_name": get(config, "model.name", "resnet18"),
        "num_classes": int(get(config, "model.num_classes", 2)),
        "class_names": list(get(config, "data.class_names", ["NORMAL", "PNEUMONIA"])),
        "input_size": int(get(config, "data.image_size", 224)),
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "dropout_rate": float(get(config, "model.dropout_rate", 0.3)),
        "seed": int(seed),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "config": {
            "model": get(config, "model"),
            "training": get(config, "training"),
            "data": get(config, "data"),
        },
    }


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
) -> tuple[float, float]:
    """Run one training epoch; returns (loss, accuracy)."""
    model.train()
    running_loss, total, correct = 0.0, 0, 0
    for images, labels in loader:
        device = next(model.parameters()).device
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=use_amp
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += int((outputs.argmax(1) == labels).sum().item())
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Evaluate the model; returns (loss, accuracy)."""
    model.eval()
    running_loss, total, correct = 0.0, 0, 0
    for images, labels in loader:
        device = next(model.parameters()).device
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        correct += int((outputs.argmax(1) == labels).sum().item())
    return running_loss / max(total, 1), correct / max(total, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    out_path: str | Path,
    device: torch.device,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    """Train ``model`` with manuscript settings, saving the best checkpoint.

    Args:
        model: The model to train (see ``medknow.models.factory``).
        train_loader: Training data.
        val_loader: Validation data (early-stopping monitor).
        config: Loaded medknow config dict (``training.*`` keys are used).
        out_path: Where to save the best checkpoint (dict with
            ``model_state_dict``, ``optimizer_state_dict``, ``epoch``,
            ``best_val_loss``, and a config snapshot).
        device: Device to train on.
        resume: Optional checkpoint to resume from.

    Returns:
        A summary dict with ``best_val_loss``, ``best_epoch``, and ``history``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seed = int(get(config, "training.seed", 42))
    epochs = int(get(config, "training.epochs", 25))
    learning_rate = float(get(config, "training.learning_rate", 1e-4))
    weight_decay = float(get(config, "training.weight_decay", 1e-5))
    use_amp = bool(get(config, "training.use_amp", True)) and device.type == "cuda"
    patience = int(get(config, "training.early_stopping.patience", 5))
    min_delta = float(get(config, "training.early_stopping.min_delta", 1e-3))

    set_seed(seed)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []

    if resume is not None:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        logger.info("Resumed from %s at epoch %d", resume, start_epoch)

    epochs_without_improvement = 0
    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = _train_epoch(
            model, train_loader, criterion, optimizer, scaler, use_amp
        )
        val_loss, val_acc = _eval_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })
        logger.info(
            "epoch %d/%d  train_loss %.4f (acc %.4f)  val_loss %.4f (acc %.4f)",
            epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "metadata": _checkpoint_metadata(config, seed),
                },
                out_path,
            )
            logger.info("Saved best checkpoint to %s", out_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint": str(out_path),
    }
