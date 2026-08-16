"""Tests for referral / selective prediction curves."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medknow.referral.curves import (
    error_prediction_auc,
    random_referral_curves,
    referral_curves,
    risk_coverage,
)


def _synthetic_deterministic():
    """200 samples, exactly 40 errors (20% error rate)."""
    labels = np.zeros(200, dtype=int)
    preds = np.zeros(200, dtype=int)
    preds[:40] = 1  # 40 wrong predictions (false positives)
    perfect_scores = np.zeros(200)
    perfect_scores[:40] = 1.0
    rng = np.random.default_rng(0)
    random_scores = rng.random(200)
    return labels, preds, perfect_scores, random_scores


def test_perfect_signal_clears_errors_at_25_percent():
    labels, preds, perfect, _ = _synthetic_deterministic()
    curves = referral_curves(labels, preds, perfect, rates=(0.0, 0.1, 0.25))
    # 25% referral = 50 cases; all 40 errors are referred -> retained error 0.
    assert curves["retained_error"][0] == 0.2
    assert curves["retained_error"][-1] == 0.0


def test_random_signal_flat():
    labels, preds, _, random_scores = _synthetic_deterministic()
    curves = referral_curves(labels, preds, random_scores, rates=(0.0, 0.25, 0.5))
    assert curves["retained_error"][0] == 0.2
    assert abs(curves["retained_error"][1] - 0.2) < 0.05


def test_random_referral_matches_base_error():
    labels, preds, _, _ = _synthetic_deterministic()
    random_curves = random_referral_curves(
        labels, preds, rates=(0.0, 0.1, 0.25), n_trials=50, seed=1
    )
    assert abs(random_curves["retained_error_mean"][0] - 0.2) < 1e-12
    assert abs(random_curves["retained_error_mean"][1] - 0.2) < 0.03


def test_error_prediction_auc():
    labels, preds, perfect, random_scores = _synthetic_deterministic()
    assert error_prediction_auc(labels, preds, perfect) > 0.99
    assert 0.4 < error_prediction_auc(labels, preds, random_scores) < 0.6


def test_risk_coverage_bounds():
    labels, preds, perfect, _ = _synthetic_deterministic()
    rc = risk_coverage(labels, preds, perfect)
    coverage = np.asarray(rc["coverage"])
    risk = np.asarray(rc["risk"])
    assert abs(coverage[0] - 1 / len(labels)) < 1e-12
    assert abs(risk[-1] - 0.2) < 1e-12
    assert risk[0] == 0.0  # most confident case is correct
