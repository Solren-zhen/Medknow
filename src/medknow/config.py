"""Configuration system for medknow.

All experiments are driven by YAML configs that are deep-merged over a
manuscript-derived default configuration. Relative paths are resolved against
the repository root, so configs are portable across machines.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default configuration, derived from the manuscript (v0.6):
#   "Uncertainty-guided referral for pneumonia screening improves in-domain
#    triage but fails to transfer across institutions" (MC Dropout, ResNet-18,
#    patient-level split, RSNA + NIH external validation).
DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "data_dir": "data/chest_xray",
        "split_dir": "data/split_patient",
        "external_dir": "data/external",
        "checkpoint_dir": "checkpoints",
        "output_dir": "results",
        "figures_dir": "results/figures",
        "tables_dir": "results/tables",
        "metrics_dir": "results/metrics",
        "logs_dir": "results/logs",
    },
    "device": {"selection": "auto"},
    "data": {
        "image_size": 224,
        "class_names": ["NORMAL", "PNEUMONIA"],
        "batch_size": 16,
        "num_workers": 2,
        "internal": {
            "name": "kermany",
            "root": "data/chest_xray",
            "split": {"train": 0.70, "val": 0.15, "test": 0.15},
            "seed": 42,
        },
        "external": {
            "rsna": {
                "name": "rsna",
                "root": "data/external/rsna",
                "label_file": "stage_2_detailed_class_info.csv",
            },
            "nih": {
                "name": "nih_2class",
                "root": "data/external/nih2class",
                "label_file": "filtered_dataset_2class.csv",
            },
        },
    },
    "model": {
        "name": "resnet18",
        "num_classes": 2,
        "pretrained": True,
        "freeze_backbone": True,
        "unfreeze_layers": ["layer4"],
        "dropout_rate": 0.3,
    },
    "training": {
        "seed": 42,
        "epochs": 25,
        "learning_rate": 1.0e-4,
        "weight_decay": 1.0e-5,
        "batch_size": 16,
        "use_amp": True,
        "early_stopping": {
            "enabled": True,
            "patience": 5,
            "monitor": "val_loss",
            "min_delta": 1e-3,
        },
        "lr_scheduler": {
            "enabled": True,
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 3,
            "min_lr": 1e-6,
        },
        "augmentation": {
            "random_rotation_deg": 10,
            "brightness": 0.2,
            "contrast": 0.1,
            "affine_translation_frac": 0.05,
        },
        "checkpoint": "checkpoints/seed_{seed}.pth",
    },
    "uncertainty": {
        "methods": ["mc_dropout", "msp", "entropy", "random"],
        "mc_samples": 30,
        "dropout_rate": 0.3,
        "std_threshold": 0.05,
        "batch_size": 32,
    },
    "calibration": {
        "ece_bins": 15,
        "temperature_path": "results/metrics/temperature.txt",
    },
    "referral": {
        "rates": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        "decision_threshold": 0.5,
    },
    "evaluation": {
        "bootstrap_iters": 2000,
        "bootstrap_seed": 42,
        "mc_samples": 30,
        "protocols": {
            "internal_test": {"inference": "single_pass_raw", "ece_style": "label_rate"},
            "rsna": {"inference": "single_pass_raw", "ece_style": "label_rate"},
            "nih": {"inference": "mc_mean_scaled", "ece_style": "confidence"},
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively merge ``override`` into ``base`` (in place)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _resolve(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return str(p)


def _resolve_paths(cfg: dict[str, Any]) -> None:
    """Resolve all config paths to absolute paths rooted at the repo."""
    for key in (
        "data_dir",
        "split_dir",
        "external_dir",
        "checkpoint_dir",
        "output_dir",
        "figures_dir",
        "tables_dir",
        "metrics_dir",
        "logs_dir",
    ):
        if key in cfg.get("paths", {}):
            cfg["paths"][key] = _resolve(cfg["paths"][key])

    internal = cfg.get("data", {}).get("internal", {})
    if "root" in internal:
        internal["root"] = _resolve(internal["root"])
    for ext in cfg.get("data", {}).get("external", {}).values():
        if "root" in ext:
            ext["root"] = _resolve(ext["root"])

    ckpt = cfg.get("training", {}).get("checkpoint")
    if ckpt:
        cfg["training"]["checkpoint"] = _resolve(ckpt)
    temp = cfg.get("calibration", {}).get("temperature_path")
    if temp:
        cfg["calibration"]["temperature_path"] = _resolve(temp)


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load a YAML config, deep-merged over the manuscript defaults.

    Args:
        path: Path to a YAML config file (absolute, or relative to the repo
            root). ``None`` returns the default config.

    Returns:
        A config dict with all paths resolved to absolute paths.

    Raises:
        FileNotFoundError: if the given config file does not exist.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        user_cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        _deep_merge(cfg, user_cfg)
    _resolve_paths(cfg)
    return cfg


def get(cfg: dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Dot-path access into a loaded config dict.

    Example: ``get(cfg, "training.epochs")``.
    """
    node = cfg
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
