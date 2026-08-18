"""Unified uncertainty estimation interface.

Every method returns an :class:`UncertaintyResult` with the same structure:
per-sample labels, mean probabilities, argmax predictions, and an uncertainty
score where *higher = more uncertain* (and therefore referred first).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class UncertaintyResult:
    """Output of any uncertainty estimator.

    Attributes:
        labels: Ground-truth labels, shape ``(N,)``.
        probs: Mean predicted probabilities, shape ``(N, C)``.
        preds: Argmax predictions, shape ``(N,)``.
        scores: Uncertainty scores, shape ``(N,)``; higher = more uncertain.
        method: Name of the estimator.
        extra: Optional estimator-specific data (e.g. full std arrays).
    """

    labels: np.ndarray
    probs: np.ndarray
    preds: np.ndarray
    scores: np.ndarray
    method: str
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels)
        self.probs = np.asarray(self.probs)
        self.preds = np.asarray(self.preds)
        self.scores = np.asarray(self.scores)


def estimate_uncertainty(model, loader, method: str = "mc_dropout", **kwargs) -> UncertaintyResult:
    """Dispatch to a named uncertainty estimator.

    Only the keyword arguments accepted by the selected method are forwarded
    (``n_samples``/``seed`` for MC Dropout and random; ``temperature`` for all;
    ``model_paths`` for ensembles).

    Args:
        model: A PyTorch model (ignored by ``random``).
        loader: A DataLoader yielding ``(images, labels)``.
        method: One of ``mc_dropout``, ``msp``, ``entropy``, ``random``,
            ``ensemble``.
        **kwargs: Method-specific options.
    """
    temperature = kwargs.pop("temperature", 1.0)
    device = kwargs.pop("device", None)
    seed = kwargs.pop("seed", None)
    n_samples = kwargs.pop("n_samples", 30)

    if method == "mc_dropout":
        from medknow.uncertainty.mc_dropout import estimate_mc_dropout
        return estimate_mc_dropout(
            model, loader, n_samples=n_samples, temperature=temperature,
            device=device, seed=seed,
        )
    if method == "msp":
        from medknow.uncertainty.msp import estimate_msp
        return estimate_msp(model, loader, temperature=temperature, device=device)
    if method == "entropy":
        from medknow.uncertainty.msp import estimate_entropy
        return estimate_entropy(model, loader, temperature=temperature, device=device)
    if method == "random":
        from medknow.uncertainty.random import estimate_random
        return estimate_random(
            model, loader, temperature=temperature, seed=seed, device=device,
        )
    if method == "conformal":
        from medknow.calibration.temperature_scaling import collect_logits
        from medknow.uncertainty.conformal import estimate_conformal
        cal_logits = np.asarray(kwargs.pop("cal_logits"))
        cal_labels = np.asarray(kwargs.pop("cal_labels"))
        alpha = float(kwargs.pop("alpha", 0.1))
        labels, logits = collect_logits(model, loader, device=device)
        return estimate_conformal(logits, labels, cal_logits, cal_labels, alpha=alpha)
    if method == "ensemble":
        from medknow.uncertainty.ensemble import estimate_ensemble
        model_paths: list[str | Any] = kwargs.pop("model_paths")
        return estimate_ensemble(
            model_paths, loader, temperature=temperature, device=device,
        )
    raise ValueError(f"Unknown uncertainty method: {method}")
