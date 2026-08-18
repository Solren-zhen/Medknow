"""Unified evaluation metrics for MedKnow cohorts."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score

from medknow.calibration.metrics import compute_brier, compute_ece


def _calibration_parameters(labels: np.ndarray, p_pos: np.ndarray) -> tuple[float | None, float | None]:
    """Fit calibration intercept and slope on the logit scale.

    The fit is intentionally reported as a descriptive evaluation statistic;
    it must be estimated on a validation set when used for recalibration.
    """
    labels = np.asarray(labels, dtype=float)
    p_pos = np.clip(np.asarray(p_pos, dtype=float), 1e-7, 1.0 - 1e-7)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None, None
    logits = np.log(p_pos / (1.0 - p_pos))

    def objective(params: np.ndarray) -> float:
        linear = params[0] + params[1] * logits
        return float(np.mean(np.logaddexp(0.0, linear) - labels * linear))

    result = minimize(
        objective,
        np.array([0.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0), (-20.0, 20.0)],
    )
    if not np.isfinite(result.fun):
        return None, None
    return float(result.x[0]), float(result.x[1])


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
    calibration_intercept, calibration_slope = _calibration_parameters(labels, p_pos)
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
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
    }


def bootstrap_confidence_intervals(
    labels,
    p_pos,
    *,
    groups=None,
    n_bootstrap: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
    n_bins: int = 15,
) -> dict:
    """Estimate percentile CIs, optionally by patient/group cluster bootstrap.

    When ``groups`` is supplied, whole groups are sampled with replacement so
    correlated images from one patient stay together. The function returns
    point estimates and the number of valid bootstrap replicates for every
    metric; AUC-like metrics can be undefined in a replicate containing one
    class and are therefore omitted from that replicate.
    """
    labels = np.asarray(labels, dtype=int)
    p_pos = np.asarray(p_pos, dtype=float)
    if labels.ndim != 1 or p_pos.ndim != 1 or len(labels) != len(p_pos):
        raise ValueError("labels and p_pos must be one-dimensional arrays of equal length")
    if len(labels) == 0:
        raise ValueError("bootstrap requires at least one observation")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    if groups is None:
        group_indices = None
        n_units = len(labels)
    else:
        groups = np.asarray(groups)
        if groups.ndim != 1 or len(groups) != len(labels):
            raise ValueError("groups must be one-dimensional and match labels length")
        unique_groups, inverse = np.unique(groups, return_inverse=True)
        group_indices = [np.flatnonzero(inverse == i) for i in range(len(unique_groups))]
        n_units = len(group_indices)

    point = compute_metrics(labels, p_pos, n_bins=n_bins)
    metric_names = (
        "auc", "auprc", "accuracy", "sensitivity", "specificity",
        "ece_raw", "brier_raw", "calibration_intercept", "calibration_slope",
    )
    samples = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(n_bootstrap):
        sampled_units = rng.integers(0, n_units, size=n_units)
        if group_indices is None:
            sample_idx = sampled_units
        else:
            sample_idx = np.concatenate([group_indices[i] for i in sampled_units])
        metrics = compute_metrics(labels[sample_idx], p_pos[sample_idx], n_bins=n_bins)
        for name in metric_names:
            value = metrics[name]
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))

    lower_q, upper_q = alpha / 2.0, 1.0 - alpha / 2.0
    result = {}
    for name in metric_names:
        values = np.asarray(samples[name], dtype=float)
        result[name] = {
            "estimate": point[name],
            "lower": float(np.quantile(values, lower_q)) if len(values) else None,
            "upper": float(np.quantile(values, upper_q)) if len(values) else None,
            "n_valid": len(values),
        }
    result["bootstrap"] = {
        "n_replicates": int(n_bootstrap),
        "alpha": float(alpha),
        "seed": int(seed),
        "unit": "group" if groups is not None else "observation",
        "n_units": int(n_units),
    }
    return result
