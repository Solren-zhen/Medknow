"""Conformal prediction for binary classification (LAC style).

Conformal prediction wraps any classifier with a *coverage guarantee* on the
calibration distribution: with probability at least 1 - alpha, the prediction
set contains the true label (assuming exchangeability). Under domain shift the
guarantee is expected to break — this module lets MedKnow quantify that break
(empirical coverage on external cohorts) and exposes a conformal uncertainty
signal through the same referral interface as the other methods.

Reference: Vovk, Gammerman & Shafer (2005); Angelopoulos & Bates (2023)
conformal prediction tutorial (LAC = least-ambiguous set-valued classifier).
"""
from __future__ import annotations

import numpy as np

from medknow.uncertainty.base import UncertaintyResult


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_lac_threshold(
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """LAC nonconformity threshold from a calibration set.

    score_i = 1 - softmax(cal_logits)[i, cal_label_i]; the threshold q is the
    ``ceil((n+1)(1-alpha))/n`` quantile, giving marginal coverage >= 1-alpha.
    """
    probs = softmax(cal_logits)
    n = len(cal_labels)
    scores = 1.0 - probs[np.arange(n), cal_labels]
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(
    logits: np.ndarray,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    """LAC prediction sets. Returns ``(set_sizes, memberships)``.

    ``set_sizes[i]`` in {0, 1, 2}; a size of 0 or 2 signals ambiguity (refer).
    """
    probs = softmax(logits)
    memberships = (1.0 - probs) <= q
    return memberships.sum(axis=1), memberships


def estimate_conformal(
    logits: np.ndarray,
    labels: np.ndarray,
    cal_logits: np.ndarray,
    cal_labels: np.ndarray,
    alpha: float = 0.1,
) -> UncertaintyResult:
    """Conformal referral signal + coverage diagnostics.

    ``scores`` = 1 - max softmax probability (compatible with the referral
    framework: higher = more uncertain = referred first); prediction-set sizes
    and empirical coverage are returned in ``extra``.
    """
    probs = softmax(logits)
    q = fit_lac_threshold(cal_logits, cal_labels, alpha)
    set_sizes, memberships = prediction_sets(logits, q)
    covered = memberships[np.arange(len(labels)), labels]
    return UncertaintyResult(
        labels=labels,
        probs=probs,
        preds=probs.argmax(axis=1),
        scores=1.0 - probs.max(axis=1),
        method="conformal",
        extra={
            "alpha": alpha,
            "q": q,
            "set_sizes": set_sizes,
            "empirical_coverage": float(covered.mean()),
        },
    )
