"""Calibration metrics (ECE, Brier, reliability diagram data)."""

from __future__ import annotations

import numpy as np


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Sample-size-weighted expected calibration error over ``n_bins`` bins."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left = probs >= bins[i] if i == 0 else probs > bins[i]
        mask = left & (probs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(probs)) * abs(
            labels[mask].mean() - probs[mask].mean()
        )
    return float(ece)


def compute_brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean squared difference between probability and label."""
    return float(np.mean((np.asarray(probs) - np.asarray(labels)) ** 2))


def reliability_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict:
    """Reliability diagram data (bin centers, accuracy, confidence, counts)."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, acc, conf, counts = [], [], [], []
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i + 1])
        if mask.sum() == 0:
            centers.append((bins[i] + bins[i + 1]) / 2)
            acc.append(np.nan)
            conf.append((bins[i] + bins[i + 1]) / 2)
            counts.append(0)
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        acc.append(float(labels[mask].mean()))
        conf.append(float(probs[mask].mean()))
        counts.append(int(mask.sum()))
    return {
        "bin_centers": centers,
        "accuracy": acc,
        "confidence": conf,
        "counts": counts,
    }

def compute_ece_confidence(probs, labels, n_bins=15):
    """ECE over predicted-class confidence (classical binary ECE).

    Bins by ``conf = max(p, 1-p)`` and compares the accuracy of the argmax
    prediction against the mean confidence within each bin. This is the formula
    used for the manuscript's NIH ChestXray-14 cohort, whereas
    :func:`compute_ece` (bins by positive-class probability) was used for the
    internal and RSNA cohorts. Both are labeled "ECE" in the manuscript.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    conf = np.maximum(probs, 1.0 - probs)
    preds = (probs >= 0.5).astype(int)
    acc = (preds == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        left = conf >= bins[i] if i == 0 else conf > bins[i]
        mask = left & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(conf)) * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)
