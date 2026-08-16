"""Temperature scaling for model calibration.

Temperature scaling fits a single scalar ``T`` that divides the logits to
minimize negative log-likelihood on a validation set. It is a post-hoc
calibration method: it fixes *in-domain* miscalibration only and is **not** a
domain-adaptation method (the manuscript shows external ECE stays ~0.36 with
the internally fitted T = 1.67).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize_scalar


@torch.no_grad()
def collect_logits(model, loader, device: torch.device | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run a loader and return ``(labels, logits)`` from an eval-mode model."""
    device = device or next(model.parameters()).device
    model = model.to(device)
    model.eval()
    labels_list, logits_list = [], []
    for images, labels in loader:
        images = images.to(device)
        logits_list.append(model(images).cpu().numpy())
        labels_list.append(labels.numpy())
    return np.concatenate(labels_list), np.concatenate(logits_list)


def _nll(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    scaled = logits / temperature
    max_logit = scaled.max(axis=1, keepdims=True)
    log_probs = scaled - max_logit - np.log(
        np.exp(scaled - max_logit).sum(axis=1, keepdims=True)
    )
    return float(-np.mean(log_probs[np.arange(len(labels)), labels]))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> float:
    """Fit the temperature that minimizes NLL on ``(logits, labels)``."""
    result = minimize_scalar(
        _nll,
        bounds=bounds,
        method="bounded",
        args=(np.asarray(logits), np.asarray(labels)),
    )
    return float(result.x)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled softmax probabilities."""
    scaled = np.asarray(logits) / temperature
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)
