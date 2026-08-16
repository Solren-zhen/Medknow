#!/usr/bin/env python3
"""MedKnow — run referral / selective-prediction analysis.

Computes referral curves at fixed rates for one or more uncertainty methods,
the random-referral control, error-prediction AUC, and risk-coverage curves.

Usage:
    python scripts/medknow_run_referral.py --weights checkpoints/seed_42.pth
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

from medknow.config import get, load_config
from medknow.datasets.chest_xray import (
    base_transform,
    build_internal_dataset,
    make_loader,
)
from medknow.models.factory import load_trained_model
from medknow.referral.curves import (
    error_prediction_auc,
    random_referral_curves,
    referral_curves,
    risk_coverage,
)
from medknow.uncertainty.base import estimate_uncertainty


def _resolve_temperature(cfg, cli_value):
    if cli_value is not None:
        return cli_value
    for candidate in (
        Path(get(cfg, "calibration.temperature_path")),
        PROJECT_ROOT / "outputs" / "temperature.txt",
    ):
        if candidate.exists():
            try:
                return float(candidate.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
    return 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--methods", default="mc_dropout,msp,entropy,random")
    ap.add_argument("--mc_samples", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--n_random_trials", type=int, default=100)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only first N images (smoke)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    split_dir = args.data_dir or get(cfg, "paths.split_dir")
    image_size = int(get(cfg, "data.image_size"))
    batch_size = int(get(cfg, "data.batch_size"))
    num_workers = int(get(cfg, "data.num_workers"))
    rates = list(get(cfg, "referral.rates"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = _resolve_temperature(cfg, args.temperature)

    model = load_trained_model(args.weights, num_classes=2, device=device)
    loader = make_loader(
        build_internal_dataset(split_dir, args.split, base_transform(image_size)),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    if args.limit is not None:
        loader.dataset.samples = loader.dataset.samples[: args.limit]

    report = {
        "weights": args.weights,
        "split": args.split,
        "temperature": temperature,
        "rates": rates,
        "methods": {},
        "random_control": None,
    }

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        logger.info("Estimating uncertainty: %s", method)
        result = estimate_uncertainty(
            model, loader, method=method,
            n_samples=args.mc_samples, temperature=temperature,
        )
        curves = referral_curves(result.labels, result.preds, result.scores, rates)
        rc = risk_coverage(result.labels, result.preds, result.scores)
        ep_auc = error_prediction_auc(result.labels, result.preds, result.scores)
        report["methods"][method] = {
            "referral": curves,
            "risk_coverage": rc,
            "error_prediction_auc": ep_auc,
        }
        if method == "random":
            report["random_control"] = curves

    report["random_control"] = random_referral_curves(
        result.labels, result.preds, rates,
        n_trials=args.n_random_trials, seed=42,
    )

    out_path = args.out or str(
        Path(get(cfg, "paths.metrics_dir"))
        / f"referral_internal_{args.split}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
