"""MC Dropout uncertainty (manuscript primary estimator)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from medknow.models.factory import enable_dropout
from medknow.uncertainty.base import UncertaintyResult


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def _forward_probs(model, images: torch.Tensor, temperature: float) -> np.ndarray:
    logits = model(images) / temperature
    return F.softmax(logits, dim=1).cpu().numpy()


def estimate_mc_dropout(
    model,
    loader,
    n_samples: int = 30,
    temperature: float = 1.0,
    device: torch.device | None = None,
    seed: int | None = None,
) -> UncertaintyResult:
    """Estimate uncertainty with ``n_samples`` stochastic forward passes.

    Batch normalization stays in eval mode so that only dropout contributes to
    the variance (matching the manuscript). The score is the standard deviation
    of the pneumonia probability across passes.
    """
    device = device or _device_of(model)
    model = model.to(device)
    model.eval()
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    enable_dropout(model)

    labels_list, mean_list, std_list = [], [], []
    for images, labels in loader:
        images = images.to(device)
        samples = np.stack(
            [_forward_probs(model, images, temperature) for _ in range(n_samples)]
        )  # (N_samples, B, C)
        mean_list.append(samples.mean(axis=0))
        std_list.append(samples.std(axis=0))
        labels_list.append(labels.numpy())

    model.eval()
    labels = np.concatenate(labels_list)
    probs = np.concatenate(mean_list)
    std_full = np.concatenate(std_list)
    scores = std_full[:, 1]  # std of P(pneumonia)
    return UncertaintyResult(
        labels=labels,
        probs=probs,
        preds=probs.argmax(axis=1),
        scores=scores,
        method="mc_dropout",
        extra={"std_full": std_full},
    )
