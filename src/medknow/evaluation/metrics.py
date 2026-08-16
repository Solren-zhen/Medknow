"""Unified evaluation metrics for MedKnow cohorts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from medknow.calibration.metrics import compute_brier, compute_ece


def compute_metrics(labels, p_pos, n_bins: int = 15) -> dict:
    """Compute the manuscript metric set from labels and positive probabilities.

    Returns AUC, AUPRC, accuracy, sensitivity, specificity, raw ECE and Brier
    (15 bins), plus the confusion matrix and cohort size.
    """
    labels = np.asarray(labels, dtype=int)
    p_pos = np.asarray(p_pos, dtype=float)
    preds = (p_pos >= 0.5).astype(int)
    acc = float((preds == labels).mean())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    two_classes = len(np.unique(labels)) > 1
    return {
        "n": len(labels),
        "pos_rate": float(labels.mean()),
        "auc": float(roc_auc_score(labels, p_pos)) if two_classes else None,
        "auprc": (
            float(average_precision_score(labels, p_pos)) if two_classes else None
        ),
        "accuracy": acc,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "ece_raw": compute_ece(p_pos, labels, n_bins=n_bins),
        "brier_raw": compute_brier(p_pos, labels),
    }
