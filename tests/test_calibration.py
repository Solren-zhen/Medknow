"""Tests for calibration metrics and temperature scaling."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medknow.calibration.metrics import (
    compute_brier,
    compute_ece,
    compute_ece_confidence,
)
from medknow.calibration.temperature_scaling import (
    apply_temperature,
    fit_temperature,
)


def test_ece_perfectly_calibrated():
    # All probabilities equal 0.5, and exactly half of the labels are positive:
    # within the (0.4, 0.5] bin, accuracy == confidence == 0.5, so ECE == 0.
    probs = np.full(100, 0.5)
    labels = np.array([0] * 50 + [1] * 50)
    assert compute_ece(probs, labels, n_bins=15) == 0.0


def test_ece_known_miscalibration_nonnegative():
    probs = np.array([0.9, 0.9])
    labels = np.array([0, 0])
    assert compute_ece(probs, labels, n_bins=15) > 0.0


def test_brier_known_values():
    assert compute_brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert abs(compute_brier(np.array([0.5, 0.5]), np.array([1, 0])) - 0.25) < 1e-12


def test_temperature_recovers_known_scale():
    rng = np.random.default_rng(1)
    raw_logits = rng.normal(0, 1, (3000, 2))
    true_t = 2.0
    probs = apply_temperature(raw_logits, true_t)
    labels = np.array([rng.random() < p for p in probs[:, 1]], dtype=int)
    fit_t = fit_temperature(raw_logits, labels)
    assert abs(fit_t - true_t) < 0.15


def test_apply_temperature_returns_probabilities():
    logits = np.array([[1.0, 0.0], [0.0, 1.0]])
    p = apply_temperature(logits, 2.0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)


def test_ece_confidence_perfectly_calibrated():
    # confidence 0.9 with exactly 90% accuracy -> ECE == 0
    probs = np.full(10, 0.9)
    labels = np.array([1] * 9 + [0])
    assert compute_ece_confidence(probs, labels, n_bins=15) == 0.0
    # confidence 0.9 with 100% accuracy -> overconfident, ECE == 0.1
    assert abs(compute_ece_confidence(np.full(10, 0.9), np.ones(10, dtype=int), n_bins=15) - 0.1) < 1e-12
