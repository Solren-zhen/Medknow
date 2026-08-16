#!/usr/bin/env python3
"""
模型工厂：统一创建接口，支持多种预训练架构
"""


import os
import warnings

import torch
from torch import nn
from torchvision import models

# ---- 可用的模型注册表 ----
AVAILABLE_MODELS = {
    "resnet18": {
        "name_cn": "ResNet18",
        "input_size": 224,
        "params": "11.7M",
        "description": "轻量残差网络，适合快速实验",
    },
    "resnet50": {
        "name_cn": "ResNet50",
        "input_size": 224,
        "params": "25.6M",
        "description": "更深残差网络，精度更高",
    },
    "efficientnet_b0": {
        "name_cn": "EfficientNet-B0",
        "input_size": 224,
        "params": "5.3M",
        "description": "高效架构，参数少精度高",
    },
    "densenet121": {
        "name_cn": "DenseNet121",
        "input_size": 224,
        "params": "8.0M",
        "description": "密集连接，特征复用充分",
    },
    "mobilenet_v2": {
        "name_cn": "MobileNetV2",
        "input_size": 224,
        "params": "3.5M",
        "description": "移动端优化，极致轻量",
    },
}


def list_available_models() -> dict:
    """列出所有可用模型"""
    return AVAILABLE_MODELS


def get_model_info(name: str) -> dict | None:
    """获取指定模型的信息"""
    return AVAILABLE_MODELS.get(name)


def create_model(
    name: str = "resnet18",
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """
    创建指定架构的分类模型

    Args:
        name: 模型名称 (resnet18, efficientnet_b0, densenet121 等)
        num_classes: 分类数
        pretrained: 是否加载 ImageNet 预训练权重
        freeze_backbone: 是否冻结骨干网络
        dropout_rate: dropout 比率（用于 MC Dropout）

    Returns:
        PyTorch 模型（未移到设备）
    """
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"未知模型: {name}。可选: {list(AVAILABLE_MODELS.keys())}")

    if name.startswith("resnet"):
        return _create_resnet(name, num_classes, pretrained, freeze_backbone, dropout_rate)
    elif name.startswith("efficientnet"):
        return _create_efficientnet(name, num_classes, pretrained, freeze_backbone, dropout_rate)
    elif name.startswith("densenet"):
        return _create_densenet(name, num_classes, pretrained, freeze_backbone, dropout_rate)
    elif name.startswith("mobilenet"):
        return _create_mobilenet(name, num_classes, pretrained, freeze_backbone, dropout_rate)
    else:
        raise NotImplementedError(f"模型 {name} 尚未实现")


# ---- 各模型创建函数 ----

def _create_resnet(name: str, num_classes: int, pretrained: bool, freeze: bool, dropout: float) -> nn.Module:
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"不支持的 ResNet 变体: {name}")

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        # 解冻最后一层
        for param in model.layer4.parameters():
            param.requires_grad = True
    # 解冻分类头
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def _create_efficientnet(name: str, num_classes: int, pretrained: bool, freeze: bool, dropout: float) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        # 解冻最后几层
        for param in model.features[-2:].parameters():
            param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def _create_densenet(name: str, num_classes: int, pretrained: bool, freeze: bool, dropout: float) -> nn.Module:
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.densenet121(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        # 解冻最后的 denseblock
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.classifier.in_features, num_classes),
    )
    return model


def _create_mobilenet(name: str, num_classes: int, pretrained: bool, freeze: bool, dropout: float) -> nn.Module:
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        # 解冻最后几个 block
        for param in model.features[-3:].parameters():
            param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.last_channel, num_classes),
    )
    return model


# ---- 辅助函数 ----

def get_target_layer(model: nn.Module, model_name: str):
    """
    获取 Grad-CAM 的目标卷积层

    Args:
        model: PyTorch 模型
        model_name: 模型名称

    Returns:
        目标层模块
    """
    if model_name.startswith("resnet"):
        return model.layer4[-1]
    elif model_name.startswith("efficientnet"):
        return model.features[-1][0]  # 最后一个 MBConv 的 conv
    elif model_name.startswith("densenet"):
        return model.features.denseblock4.denselayer16.conv2
    elif model_name.startswith("mobilenet"):
        return model.features[-1][0]
    else:
        raise ValueError(f"未知模型: {model_name}")


def enable_dropout(model: nn.Module):
    """启用所有 Dropout 层（用于 MC Dropout 推理）"""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def load_trained_model(
    name: str = "resnet18",
    num_classes: int = 2,
    model_path: str | None = None,
    device=None,
) -> nn.Module:
    """
    加载训练好的模型权重，自动按架构匹配权重文件。

    权重解析顺序：
      1. 显式传入的 model_path
      2. outputs/pneumonia_model.pth（默认部署模型，seed_42）
    注意：outputs/model_comparison/{name}.pth 为早期图片级切分架构对比产物，
    训练协议与论文模型不同，不再参与自动解析，避免 seed 评估误加载旧权重。

    Args:
        name: 模型架构名 (resnet18, efficientnet_b0, densenet121 ...)
        num_classes: 分类数
        model_path: 显式权重路径（可选）
        device: torch device（默认 CPU）

    Returns:
        已加载权重的 eval 模式模型

    Raises:
        ValueError: 权重与请求的模型架构不匹配（例如用 resnet18 权重加载 efficientnet_b0）
    """
    from config import OUTPUT_DIR

    device = device or torch.device("cpu")

    if model_path is None:
        model_path = os.path.join(OUTPUT_DIR, "pneumonia_model.pth")
        warnings.warn(
            "model_path 未指定，自动解析为部署模型: "
            f"{model_path}。评估特定 checkpoint 时请显式传 model_path。",
            stacklevel=2,
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型权重不存在: {model_path}（请先运行 python scripts/train.py）")

    model = create_model(name=name, num_classes=num_classes, pretrained=False, freeze_backbone=False)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise ValueError(
            f"权重与模型架构不匹配: {name} <- {model_path}\n"
            f"请使用与权重一致的模型名，或重新训练该架构: python scripts/train.py --model {name}\n"
            f"原始错误: {exc}"
        ) from exc

    model = model.to(device)
    model.eval()
    return model
