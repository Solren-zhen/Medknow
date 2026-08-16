"""Random referral baseline (matched-rate control)."""

from __future__ import annotations

import numpy as np

from medknow.training.inference import predict_batch_probs
from medknow.uncertainty.base import UncertaintyResult


def estimate_random(
    model,
    loader,
    temperature: float = 1.0,
    seed: int | None = None,
    device=None,
) -> UncertaintyResult:
    """Random uncertainty scores over the model's real predictions.

    The predictions are real model outputs, but the referral ranking is random,
    providing the matched-rate control that isolates whether the uncertainty
    signal itself carries information about correctness.
    """
    labels, probs = predict_batch_probs(model, loader, temperature=temperature)
    rng = np.random.default_rng(seed)
    scores = rng.random(len(labels))
    return UncertaintyResult(
        labels=labels,
        probs=probs,
        preds=probs.argmax(axis=1),
        scores=scores,
        method="random",
    )
