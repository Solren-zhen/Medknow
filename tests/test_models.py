"""Tests for the MedKnow model factory."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medknow.models.factory import (
    create_resnet18,
    enable_dropout,
    get_target_layer,
    load_trained_model,
)


def test_create_model_output_shape_and_freeze():
    model = create_resnet18(num_classes=2, pretrained=False, freeze_backbone=True)
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 2)
    assert not model.layer1[0].conv1.weight.requires_grad
    assert model.layer4[-1].conv1.weight.requires_grad
    assert model.fc[1].weight.requires_grad


def test_target_layer():
    model = create_resnet18(pretrained=False)
    assert get_target_layer(model) is model.layer4[-1]


def test_enable_dropout():
    model = create_resnet18(pretrained=False)
    model.eval()
    assert model.fc[0].training is False
    enable_dropout(model)
    assert model.fc[0].training is True


def test_load_roundtrip_raw_state_dict(tmp_path):
    model = create_resnet18(pretrained=False)
    p = tmp_path / "raw.pth"
    torch.save(model.state_dict(), p)
    loaded = load_trained_model(p)
    assert loaded.fc[1].out_features == 2


def test_load_roundtrip_checkpoint_dict(tmp_path):
    model = create_resnet18(pretrained=False)
    p = tmp_path / "ckpt.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 3}, p)
    loaded = load_trained_model(p)
    assert loaded.training is False
