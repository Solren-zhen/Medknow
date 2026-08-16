"""Inference helpers: single-pass prediction and MC Dropout uncertainty."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from medknow.models.factory import enable_dropout


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    x: torch.Tensor,
    temperature: float = 1.0,
) -> np.ndarray:
    """Single-pass softmax probabilities for one input; shape ``(C,)``."""
    model.eval()
    logits = model(x) / temperature
    return F.softmax(logits, dim=1)[0].cpu().numpy()


@torch.no_grad()
def predict_batch_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a dataloader in eval mode; returns ``(labels, probs)`` arrays."""
    model.eval()
    device = next(model.parameters()).device
    all_labels: list = []
    all_probs: list = []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images) / temperature
        all_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


@torch.no_grad()
def predict_mc_dropout(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_samples: int = 30,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """MC Dropout: ``n_samples`` stochastic passes -> (mean probs, std probs).

    Batch normalization stays in eval mode so that only dropout contributes to
    the variance (matching the manuscript).
    """
    model.eval()
    enable_dropout(model)
    samples = []
    for _ in range(n_samples):
        logits = model(x) / temperature
        samples.append(F.softmax(logits, dim=1)[0].cpu().numpy())
    model.eval()
    samples = np.stack(samples)  # (N, C)
    return samples.mean(axis=0), samples.std(axis=0)
