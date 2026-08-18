#!/usr/bin/env python3
"""
配置加载器：从 config.yaml 读取，提供类型安全的配置访问
"""

import os
from pathlib import Path
from typing import Any

import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.resolve()


def _load_config(config_path: str | None = None) -> dict[str, Any]:
    """加载 YAML 配置文件，解析路径"""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"⚠️ 配置文件不存在: {config_path}，使用默认值")
        return _default_config()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 解析相对路径为绝对路径
    _resolve_paths(cfg)
    return cfg


def _resolve_paths(cfg: dict[str, Any]) -> None:
    """将配置中的相对路径转为绝对路径"""
    for key in ("data_dir", "output_dir", "model_dir"):
        if key in cfg.get("paths", {}):
            p = Path(cfg["paths"][key])
            if not p.is_absolute():
                cfg["paths"][key] = str(PROJECT_ROOT / p)
    if "database" in cfg and "path" in cfg["database"]:
        p = Path(cfg["database"]["path"])
        if not p.is_absolute():
            cfg["database"]["path"] = str(PROJECT_ROOT / p)


def _default_config() -> dict[str, Any]:
    """默认配置（当 config.yaml 缺失时使用）"""
    return {
        "paths": {
            "data_dir": str(PROJECT_ROOT / "data" / "chest_xray"),
            "output_dir": str(PROJECT_ROOT / "outputs"),
            "model_dir": str(PROJECT_ROOT / "outputs"),
        },
        "device": {"selection": "auto"},
        "data": {"image_size": 224, "batch_size": 16, "num_workers": 2, "class_names": ["NORMAL", "PNEUMONIA"]},
        "training": {
            "model_name": "resnet18", "epochs": 25, "learning_rate": 1e-4,
            "weight_decay": 1e-5, "freeze_backbone": True, "unfreeze_layers": ["layer4"],
            "use_weighted_sampler": True,
            "early_stopping": {"enabled": True, "patience": 5, "monitor": "val_loss", "min_delta": 1e-3},
            "lr_scheduler": {"enabled": True, "type": "reduce_on_plateau", "factor": 0.5, "patience": 3, "min_lr": 1e-6},
            "use_amp": True,
            "augmentation": {"random_horizontal_flip": True, "random_rotation": 10, "color_jitter_brightness": 0.2},
            "tensorboard": True, "log_interval": 10,
        },
        "uncertainty": {"mc_samples": 30, "dropout_rate": 0.3, "std_threshold": 0.05},
        "explainability": {"methods": ["gradcam", "integrated_gradients", "occlusion"], "gradcam_target_layer": "layer4", "ig_steps": 50, "occlusion_size": 32},
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "max_upload_size_mb": 20,
            "max_image_pixels": 12000000,
            "max_mc_samples": 50,
            "allowed_content_types": [
                "image/jpeg",
                "image/png",
                "image/webp",
                "image/bmp",
                "image/tiff",
                "application/dicom",
                "application/octet-stream",
            ],
            "allowed_extensions": [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"],
            "allowed_origins": [
                "http://localhost:8501",
                "http://127.0.0.1:8501",
                "http://localhost:7860",
                "http://127.0.0.1:7860",
            ],
            "admin_token": None,
        },
        "database": {"path": str(PROJECT_ROOT / "outputs" / "predictions.db")},
    }


# ---------- 模块级配置对象 ----------
cfg = _load_config()


def get(key_path: str, default: Any = None) -> Any:
    """用点号分隔的路径获取配置，如 get('training.epochs')"""
    keys = key_path.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val


def device():
    """获取 torch device 对象"""
    import torch

    sel = get("device.selection", "auto")
    if sel == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(sel)


def ensure_output_dir():
    """确保输出目录存在"""
    os.makedirs(get("paths.output_dir"), exist_ok=True)


# 常用快捷访问
DATA_DIR = get("paths.data_dir")
OUTPUT_DIR = get("paths.output_dir")
MODEL_DIR = get("paths.model_dir")
IMAGE_SIZE = get("data.image_size", 224)
CLASS_NAMES = get("data.class_names", ["NORMAL", "PNEUMONIA"])
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = get("data.batch_size", 16)
NUM_WORKERS = get("data.num_workers", 2)
DEVICE = device()
