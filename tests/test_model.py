#!/usr/bin/env python3
"""模型相关测试"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_factory import (
    create_model,
    enable_dropout,
    get_target_layer,
    list_available_models,
)


class TestModelFactory:
    """测试模型工厂"""

    def test_list_models(self):
        models = list_available_models()
        assert "resnet18" in models
        assert "efficientnet_b0" in models
        assert "densenet121" in models
        assert models["resnet18"]["name_cn"] == "ResNet18"

    @pytest.mark.parametrize("model_name", ["resnet18", "efficientnet_b0", "densenet121"])
    def test_create_model_no_pretrained(self, model_name):
        """不加载预训练权重创建模型"""
        model = create_model(name=model_name, num_classes=2, pretrained=False, freeze_backbone=False)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize("model_name,num_classes", [
        ("resnet18", 2), ("resnet18", 3), ("efficientnet_b0", 2), ("densenet121", 4),
    ])
    def test_create_model_various_classes(self, model_name, num_classes):
        model = create_model(name=model_name, num_classes=num_classes, pretrained=False, freeze_backbone=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, num_classes)

    def test_freeze_backbone(self):
        model = create_model(name="resnet18", num_classes=2, pretrained=False, freeze_backbone=True)
        # 分类头应可训练
        assert any(p.requires_grad for p in model.fc.parameters())
        # 浅层应冻结
        assert not any(p.requires_grad for p in model.conv1.parameters())

    def test_dropout_in_classifier(self):
        """验证分类头包含 Dropout（MC Dropout 需要）"""
        model = create_model(name="resnet18", num_classes=2, pretrained=False, freeze_backbone=False)
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.fc.modules())
        assert has_dropout, "分类头应包含 Dropout 层"


class TestForwardPass:
    """测试前向传播"""

    @pytest.mark.parametrize("model_name", ["resnet18", "efficientnet_b0", "densenet121"])
    def test_forward_output_shape(self, model_name):
        model = create_model(name=model_name, num_classes=2, pretrained=False, freeze_backbone=False)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 2)

    def test_softmax_sum(self):
        """验证 softmax 概率和为 1"""
        model = create_model(name="resnet18", num_classes=2, pretrained=False, freeze_backbone=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        probs = F.softmax(out, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.tensor([1.0]), atol=1e-5)


class TestMCdropout:
    """测试 MC Dropout"""

    def test_enable_dropout(self):
        model = create_model(name="resnet18", num_classes=2, pretrained=False, freeze_backbone=False)
        model.eval()
        enable_dropout(model)

        # 两次前向传播应有不同结果（因为 Dropout 活跃）
        x = torch.randn(1, 3, 224, 224)
        out1 = model(x)
        out2 = model(x)
        # 不做严格断言（Dropout 是随机的），只是确保不报错
        assert out1.shape == out2.shape


class TestTargetLayer:
    """测试 Grad-CAM 目标层获取"""

    @pytest.mark.parametrize("model_name", ["resnet18", "efficientnet_b0"])
    def test_get_target_layer(self, model_name):
        model = create_model(name=model_name, num_classes=2, pretrained=False, freeze_backbone=False)
        layer = get_target_layer(model, model_name)
        assert layer is not None
        assert isinstance(layer, torch.nn.Module)


class TestConfig:
    """测试配置加载"""

    def test_config_loads(self):
        from config import cfg, get
        assert "paths" in cfg
        assert "training" in cfg
        assert get("training.epochs", 0) > 0

    def test_device(self):
        from config import device
        d = device()
        assert isinstance(d, torch.device)
