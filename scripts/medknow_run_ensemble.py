#!/usr/bin/env python3
"""MedKnow — Deep Ensemble referral analysis (three seed checkpoints).

Averages softmax probabilities across the three frozen-backbone checkpoints
(seed 42 / 2024 / 2026). The ensemble uncertainty score is the across-member
standard deviation of the pneumonia probability; referral curves are computed
against a matched random control for every cohort.

Predictions are cached under ``results/metrics/predictions/``.

Usage:
    python scripts/medknow_run_ensemble.py \
        --weights checkpoints/seed_42.pth,checkpoints/seed_2024.pth,checkpoints/seed_2026.pth
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from medknow.config import get, load_config
from medknow.datasets.chest_xray import (
    base_transform,
    build_internal_dataset,
    build_nih_dataset,
    build_rsna_dataset,
    make_loader,
)
from medknow.evaluation.metrics import compute_metrics
from medknow.models.factory import load_trained_model
from medknow.referral.curves import (
    error_prediction_auc,
    random_referral_curves,
    referral_curves,
)
from medknow.training.inference import predict_batch_probs

logger = logging.getLogger(__name__)


def _pred_dir(cfg) -> Path:
    return Path(get(cfg, "paths.metrics_dir")) / "predictions"


def _ensemble_member_probs(model_paths, loader, temperature, device):
    member_probs = []
    labels = None
    for path in model_paths:
        logger.info("Ensemble member: %s", Path(path).name)
        model = load_trained_model(path, num_classes=2, device=device)
        labels, probs = predict_batch_probs(model, loader, temperature=temperature)
        member_probs.append(probs)
    return labels, np.stack(member_probs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument(
        "--weights",
        required=True,
        help="comma-separated checkpoint paths (3 seeds)",
    )
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N images")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    image_size = int(get(cfg, "data.image_size"))
    batch_size = int(get(cfg, "data.batch_size"))
    num_workers = int(get(cfg, "data.num_workers"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = args.temperature
    if temperature is None:
        temp_path = Path(get(cfg, "calibration.temperature_path"))
        if temp_path.exists():
            temperature = float(temp_path.read_text(encoding="utf-8").strip())
        else:
            temperature = 1.0
    model_paths = [Path(p.strip()) for p in args.weights.split(",") if p.strip()]

    transform = base_transform(image_size)
    cohorts = {
        "internal_test": build_internal_dataset(get(cfg, "paths.split_dir"), "test", transform),
        "rsna": build_rsna_dataset(get(cfg, "data.external.rsna.root"), transform),
        "nih": build_nih_dataset(
            Path(get(cfg, "data.external.nih.root")) / "test_resized" / "test",
            transform,
        ),
    }

    rates = list(get(cfg, "referral.rates"))
    report = {
        "model_paths": [str(p) for p in model_paths],
        "temperature": temperature,
        "cohorts": {},
    }
    for name, dataset in cohorts.items():
        if args.limit is not None:
            if hasattr(dataset, "samples"):
                dataset.samples = dataset.samples[: args.limit]
            elif hasattr(dataset, "image_paths"):
                dataset.image_paths = dataset.image_paths[: args.limit]
                dataset.labels = dataset.labels[: args.limit]
                dataset.subgroups = dataset.subgroups[: args.limit]
                dataset.patient_ids = dataset.patient_ids[: args.limit]
        loader = make_loader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        cache = _pred_dir(cfg) / f"{name}_ensemble.npz"
        if cache.exists() and not args.recompute:
            data = np.load(cache)
            labels, member_probs = data["labels"], data["member_probs"]
        else:
            labels, member_probs = _ensemble_member_probs(
                model_paths, loader, temperature, device
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, labels=labels, member_probs=member_probs)

        mean_p = member_probs[:, :, 1].mean(axis=0)
        std = member_probs[:, :, 1].std(axis=0)
        preds = (mean_p >= 0.5).astype(int)
        metrics = compute_metrics(labels, mean_p)
        curves = referral_curves(labels, preds, std, rates)
        random_curves = random_referral_curves(labels, preds, rates, n_trials=100, seed=42)
        ep_auc = error_prediction_auc(labels, preds, std)
        report["cohorts"][name] = {
            "metrics": metrics,
            "referral": curves,
            "random_control": random_curves,
            "error_prediction_auc": ep_auc,
            "n_models": len(model_paths),
        }
        i25 = rates.index(0.25)
        logger.info(
            "[%s] ensemble: AUC %.4f  epAUC %.3f  MC@25%% %.4f vs random %.4f",
            name, metrics["auc"], ep_auc,
            curves["retained_error"][i25], random_curves["retained_error_mean"][i25],
        )

    out_path = args.out or str(
        Path(get(cfg, "paths.tables_dir")) / "ensemble_summary.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
