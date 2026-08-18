"""Tests for the unified uncertainty estimation interface."""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medknow.models.factory import create_resnet18
from medknow.uncertainty.base import estimate_uncertainty
from medknow.uncertainty.conformal import estimate_conformal


def _make_loader(n=8):
    xs = torch.randn(n, 3, 64, 64)
    labels = torch.tensor([0, 1] * (n // 2))
    return DataLoader(TensorDataset(xs, labels), batch_size=4)


def test_all_methods_return_same_structure():
    model = create_resnet18(pretrained=False)
    loader = _make_loader()
    for method, kwargs in [
        ("mc_dropout", {"n_samples": 3}),
        ("msp", {}),
        ("entropy", {}),
        ("random", {}),
    ]:
        result = estimate_uncertainty(model, loader, method=method, **kwargs)
        assert result.method == method
        assert result.labels.shape == (8,)
        assert result.probs.shape == (8, 2)
        assert result.preds.shape == (8,)
        assert result.scores.shape == (8,)
        assert np.all(result.scores >= 0)
        np.testing.assert_allclose(result.probs.sum(axis=1), 1.0, atol=1e-6)


def test_mc_dropout_uncertainty_positive():
    model = create_resnet18(pretrained=False)
    loader = _make_loader(n=4)
    result = estimate_uncertainty(model, loader, method="mc_dropout", n_samples=5)
    assert np.all(result.scores >= 0)
    assert result.extra["std_full"].shape == (4, 2)


def test_conformal_known_logits_labels():
    cal_logits = np.array([
        [4.0, 0.0],
        [0.0, 4.0],
        [3.0, 0.0],
        [0.0, 3.0],
    ])
    cal_labels = np.array([0, 1, 0, 1])
    logits = np.array([
        [5.0, 0.0],
        [0.0, 5.0],
        [5.0, 0.0],
        [0.0, 5.0],
    ])
    labels = np.array([0, 1, 0, 1])

    result = estimate_conformal(logits, labels, cal_logits, cal_labels, alpha=0.25)

    assert result.method == "conformal"
    assert result.labels.tolist() == [0, 1, 0, 1]
    assert result.probs.shape == (4, 2)
    assert result.extra["set_sizes"].tolist() == [1, 1, 1, 1]
    assert result.extra["empirical_coverage"] == 1.0


def test_conformal_dispatcher_keeps_logits_label_order():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, x):
            return x[:, :2] * self.weight

    images = torch.tensor([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
    ])
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    cal_logits = np.array([
        [4.0, 0.0],
        [0.0, 4.0],
        [3.0, 0.0],
        [0.0, 3.0],
    ])
    cal_labels = np.array([0, 1, 0, 1])

    result = estimate_uncertainty(
        TinyModel(),
        loader,
        method="conformal",
        cal_logits=cal_logits,
        cal_labels=cal_labels,
        alpha=0.25,
    )

    assert result.labels.tolist() == [0, 1, 0, 1]
    assert result.probs.shape == (4, 2)
    assert result.extra["set_sizes"].tolist() == [1, 1, 1, 1]
