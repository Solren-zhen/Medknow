"""Tests for MedKnow inference helpers."""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medknow.models.factory import create_resnet18
from medknow.training.inference import (
    predict_batch_probs,
    predict_mc_dropout,
    predict_probs,
)


def test_predict_probs_sums_to_one():
    model = create_resnet18(pretrained=False).eval()
    p = predict_probs(model, torch.randn(1, 3, 224, 224))
    assert p.shape == (2,)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-6)


def test_predict_mc_dropout_shapes():
    model = create_resnet18(pretrained=False)
    mean, std = predict_mc_dropout(model, torch.randn(1, 3, 224, 224), n_samples=5)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.all(std >= 0)
    np.testing.assert_allclose(mean.sum(), 1.0, atol=1e-6)


def test_predict_batch_probs():
    model = create_resnet18(pretrained=False).eval()
    labels = torch.tensor([0, 1, 1])
    xs = torch.randn(3, 3, 224, 224)
    loader = DataLoader(TensorDataset(xs, labels), batch_size=2)
    y, p = predict_batch_probs(model, loader)
    assert y.shape == (3,)
    assert p.shape == (3, 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
