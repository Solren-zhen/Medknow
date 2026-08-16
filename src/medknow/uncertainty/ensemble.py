"""Deep ensemble uncertainty (implemented; results NOT yet verified)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from medknow.models.factory import load_trained_model
from medknow.training.inference import predict_batch_probs
from medknow.uncertainty.base import UncertaintyResult


def estimate_ensemble(
    model_paths: list[str | Path],
    loader,
    temperature: float = 1.0,
    device: torch.device | None = None,
    num_classes: int = 2,
) -> UncertaintyResult:
    """Ensemble by averaging member softmax probabilities.

    The score is the across-member standard deviation of the pneumonia
    probability. **No ensemble results are claimed yet**: this is the code
    path for a planned experiment (marked PENDING in results).
    """
    member_probs = []
    labels = None
    for path in model_paths:
        model = load_trained_model(path, num_classes=num_classes, device=device)
        labels, probs = predict_batch_probs(model, loader, temperature=temperature)
        member_probs.append(probs)
    probs = np.mean(member_probs, axis=0)
    scores = np.std([p[:, 1] for p in member_probs], axis=0)
    return UncertaintyResult(
        labels=labels,
        probs=probs,
        preds=probs.argmax(axis=1),
        scores=scores,
        method="ensemble",
        extra={"n_models": len(member_probs)},
    )
