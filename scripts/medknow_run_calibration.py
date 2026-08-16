#!/usr/bin/env python3
"""MedKnow — fit temperature scaling and report in-domain calibration.

Fits T on the validation split (minimizing NLL), then reports raw vs
temperature-scaled ECE/Brier on the test split.

Usage:
    python scripts/medknow_run_calibration.py --weights checkpoints/seed_42.pth
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
from medknow.calibration.temperature_scaling import (
    apply_temperature,
    collect_logits,
    fit_temperature,
)
from medknow.config import get, load_config
from medknow.datasets.chest_xray import (
    base_transform,
    build_internal_dataset,
    make_loader,
)
from medknow.models.factory import load_trained_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val_dir", default=None, help="default: split val")
    ap.add_argument("--test_dir", default=None, help="default: split test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    split_dir = args.val_dir or get(cfg, "paths.split_dir")
    image_size = int(get(cfg, "data.image_size"))
    batch_size = int(get(cfg, "data.batch_size"))
    num_workers = int(get(cfg, "data.num_workers"))
    ece_bins = int(get(cfg, "calibration.ece_bins", 15))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_trained_model(args.weights, num_classes=2, device=device)
    val_loader = make_loader(
        build_internal_dataset(split_dir, "val", base_transform(image_size)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_dir = args.test_dir or split_dir
    test_loader = make_loader(
        build_internal_dataset(test_dir, "test", base_transform(image_size)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    logger.info("Collecting validation logits")
    val_labels, val_logits = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)

    logger.info("Collecting test logits")
    test_labels, test_logits = collect_logits(model, test_loader, device)
    raw_probs = apply_temperature(test_logits, 1.0)
    scaled_probs = apply_temperature(test_logits, temperature)
    p_raw = raw_probs[:, 1]
    p_scaled = scaled_probs[:, 1]

    report = {
        "weights": args.weights,
        "temperature_fitted": temperature,
        "ece_raw": compute_ece(p_raw, test_labels, n_bins=ece_bins),
        "ece_scaled": compute_ece(p_scaled, test_labels, n_bins=ece_bins),
        "brier_raw": compute_brier(p_raw, test_labels),
        "brier_scaled": compute_brier(p_scaled, test_labels),
        "ece_bins": ece_bins,
    }
    out_path = args.out or str(
        Path(get(cfg, "paths.metrics_dir")) / "calibration_internal.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
