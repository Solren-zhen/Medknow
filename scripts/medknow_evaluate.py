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

import torch

from medknow.calibration.metrics import compute_brier, compute_ece
from medknow.config import get, load_config
from medknow.datasets.chest_xray import (
    base_transform,
    build_internal_dataset,
    make_loader,
)
from medknow.evaluation.metrics import bootstrap_confidence_intervals, compute_metrics
from medknow.models.factory import load_trained_model
from medknow.training.inference import predict_batch_probs


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a MedKnow checkpoint internally")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--weights", required=True, help="checkpoint (.pth)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data_dir", default=None, help="override split dir")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--bootstrap-iters", type=int, default=None)
    ap.add_argument("--bootstrap-seed", type=int, default=None)
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
    result = compute_metrics(labels, probs[:, 1], n_bins=int(get(cfg, "calibration.ece_bins", 15)))
    groups = getattr(loader.dataset, "patient_ids", None)
    bootstrap_iters = args.bootstrap_iters
    if bootstrap_iters is None:
        bootstrap_iters = int(get(cfg, "evaluation.bootstrap_iters", 2000))
    result["confidence_intervals"] = bootstrap_confidence_intervals(
        labels,
        probs[:, 1],
        groups=groups,
        n_bootstrap=bootstrap_iters,
        seed=int(args.bootstrap_seed if args.bootstrap_seed is not None else get(cfg, "evaluation.bootstrap_seed", 42)),
        n_bins=int(get(cfg, "calibration.ece_bins", 15)),
    )

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
