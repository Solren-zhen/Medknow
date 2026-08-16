"""Maximum softmax probability and entropy baselines."""

from __future__ import annotations

import numpy as np
import torch

from medknow.training.inference import predict_batch_probs
from medknow.uncertainty.base import UncertaintyResult


def estimate_msp(
    model,
    loader,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> UncertaintyResult:
    """Plain-confidence baseline.

    Score = 1 - max probability, i.e. distance to the decision boundary for a
    binary task (higher = less confident = referred first). This is the
    manuscript's "plain-confidence referral" signal.
    """
    labels, probs = predict_batch_probs(model, loader, temperature=temperature)
    scores = 1.0 - probs.max(axis=1)
    return UncertaintyResult(
        labels=labels,
        probs=probs,
        preds=probs.argmax(axis=1),
        scores=scores,
        method="msp",
    )


def estimate_entropy(
    model,
    loader,
    temperature: float = 1.0,
    device: torch.device | None = None,
) -> UncertaintyResult:
    """Predictive entropy baseline (not in the manuscript; PENDING results)."""
    labels, probs = predict_batch_probs(model, loader, temperature=temperature)
    probs = np.clip(probs, 1e-12, 1.0)
    scores = -np.sum(probs * np.log(probs), axis=1)
    return UncertaintyResult(
        labels=labels,
        probs=probs,
        preds=probs.argmax(axis=1),
        scores=scores,
        method="entropy",
    )
