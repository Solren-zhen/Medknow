"""Referral / selective prediction curves.

Core idea (manuscript): at a fixed referral rate, the most uncertain cases are
routed to a human reader and the error rate is measured on the retained
(automated) set. Comparing against random referral at matched rates isolates
whether the uncertainty signal carries information about correctness — not
merely the effect of dropping cases.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

DEFAULT_RATES: tuple[float, ...] = (
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
)


def _as_arrays(labels, preds, scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(labels, dtype=int),
        np.asarray(preds, dtype=int),
        np.asarray(scores, dtype=float),
    )


def referral_curves(
    labels,
    preds,
    scores,
    rates: Sequence[float] = DEFAULT_RATES,
) -> dict:
    """Referral curves at fixed referral rates (higher score = referred first).

    Returns a dict with, per rate: retained error, retained sensitivity,
    retained FNR, retained missed-case count, and retained coverage.
    """
    labels, preds, scores = _as_arrays(labels, preds, scores)
    n = len(labels)
    order = np.argsort(-scores)
    errors = preds != labels
    positives = labels == 1

    out = {
        "rates": [float(r) for r in rates],
        "retained_error": [],
        "retained_sensitivity": [],
        "retained_fnr": [],
        "retained_missed": [],
        "retained_coverage": [],
    }
    for rate in rates:
        k = round(rate * n)
        keep = np.ones(n, dtype=bool)
        keep[order[:k]] = False
        kept = keep.sum()
        if kept == 0:
            out["retained_error"].append(None)
            out["retained_sensitivity"].append(None)
            out["retained_fnr"].append(None)
            out["retained_missed"].append(None)
            out["retained_coverage"].append(0.0)
            continue
        ret_err = float(errors[keep].mean())
        ret_pos = positives[keep].sum()
        ret_fn = int((errors[keep] & positives[keep]).sum())
        sens = 1.0 - ret_fn / ret_pos if ret_pos > 0 else None
        out["retained_error"].append(ret_err)
        out["retained_sensitivity"].append(sens)
        out["retained_fnr"].append(1.0 - sens if sens is not None else None)
        out["retained_missed"].append(ret_fn)
        out["retained_coverage"].append(float(kept / n))
    return out


def random_referral_curves(
    labels,
    preds,
    rates: Sequence[float] = DEFAULT_RATES,
    n_trials: int = 100,
    seed: int = 42,
) -> dict:
    """Random referral control: mean/std retained metrics over ``n_trials``."""
    labels, preds, _ = _as_arrays(labels, preds, None)
    n = len(labels)
    errors = preds != labels
    positives = labels == 1
    rng = np.random.default_rng(seed)

    ret_err = np.zeros((n_trials, len(rates)))
    ret_missed = np.zeros((n_trials, len(rates)))
    for t in range(n_trials):
        order = rng.permutation(n)
        for j, rate in enumerate(rates):
            k = round(rate * n)
            keep = np.ones(n, dtype=bool)
            keep[order[:k]] = False
            if keep.sum() == 0:
                ret_err[t, j] = np.nan
                ret_missed[t, j] = np.nan
                continue
            ret_err[t, j] = errors[keep].mean()
            ret_missed[t, j] = (errors[keep] & positives[keep]).sum()

    return {
        "rates": [float(r) for r in rates],
        "n_trials": n_trials,
        "retained_error_mean": ret_err.mean(axis=0).tolist(),
        "retained_error_std": np.nanstd(ret_err, axis=0).tolist(),
        "retained_missed_mean": np.nanmean(ret_missed, axis=0).tolist(),
    }


def error_prediction_auc(labels, preds, scores) -> float | None:
    """AUC of the uncertainty score as a predictor of a wrong prediction."""
    labels, preds, scores = _as_arrays(labels, preds, scores)
    is_error = (preds != labels).astype(int)
    if len(np.unique(is_error)) < 2:
        return None
    return float(roc_auc_score(is_error, scores))


def risk_coverage(labels, preds, scores) -> dict:
    """Risk-coverage curve for selective prediction.

    Cases are accepted in order of increasing uncertainty (most confident
    first); ``coverage`` is the accepted fraction and ``risk`` the error rate
    among accepted cases.
    """
    labels, preds, scores = _as_arrays(labels, preds, scores)
    errors = (preds != labels).astype(int)
    order = np.argsort(scores)  # least uncertain first
    n = len(labels)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors[order]) / np.arange(1, n + 1)
    return {
        "coverage": coverage.tolist(),
        "risk": risk.tolist(),
    }
