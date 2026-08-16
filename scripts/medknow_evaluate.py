#!/usr/bin/env python3
"""MedKnow — evaluate a checkpoint on an internal split (default: test).

Computes AUC, AUPRC, accuracy, sensitivity, specificity, raw ECE and Brier
(15 bins, matching the manuscript), plus temperature-scaled variants when a
temperature is available (via --temperature or results/metrics/temperature.txt).

Usage:
    python scripts/medknow_evaluate.py --weights checkpoints/seed_42.pth
"""

import argparse
import json
import logging

logger = logging.getLogger(__name__)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from medknow.calibration.metrics import compute_brier, compute_ece
from medknow.config import get, load_config
from medknow.datasets.chest_xray import (
    base_transform,
    build_internal_dataset,
    make_loader,
)
from medknow.models.factory import load_trained_model
from medknow.training.inference import predict_batch_probs


def compute_metrics(labels: np.ndarray, p_pos: np.ndarray) -> dict:
    preds = (p_pos >= 0.5).astype(int)
    acc = float((preds == labels).mean())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    two_classes = len(np.unique(labels)) > 1
    return {
        "n": len(labels),
        "pos_rate": float(labels.mean()),
        "auc": float(roc_auc_score(labels, p_pos)) if two_classes else None,
        "auprc": float(average_precision_score(labels, p_pos)) if two_classes else None,
        "accuracy": acc,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "ece_raw": compute_ece(p_pos, labels),
        "brier_raw": compute_brier(p_pos, labels),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a MedKnow checkpoint internally")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--weights", required=True, help="checkpoint (.pth)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data_dir", default=None, help="override split dir")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    split_dir = args.data_dir or get(cfg, "paths.split_dir")
    image_size = int(get(cfg, "data.image_size"))
    batch_size = int(get(cfg, "data.batch_size"))
    num_workers = int(get(cfg, "data.num_workers"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_trained_model(args.weights, num_classes=2, device=device)
    loader = make_loader(
        build_internal_dataset(split_dir, args.split, base_transform(image_size)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    labels, probs = predict_batch_probs(model, loader, temperature=1.0)
    result = compute_metrics(labels, probs[:, 1])

    temperature = args.temperature
    if temperature is None:
        temp_path = Path(get(cfg, "calibration.temperature_path"))
        if temp_path.exists():
            temperature = float(temp_path.read_text(encoding="utf-8").strip())
    if temperature is not None and temperature != 1.0:
        _, scaled = predict_batch_probs(model, loader, temperature=temperature)
        result["temperature"] = temperature
        result["ece_scaled"] = compute_ece(scaled[:, 1], labels)
        result["brier_scaled"] = compute_brier(scaled[:, 1], labels)

    out_path = args.out or str(
        Path(get(cfg, "paths.metrics_dir")) / f"eval_internal_{args.split}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
