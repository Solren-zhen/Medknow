#!/usr/bin/env python3
"""MedKnow — evaluation-protocol sensitivity analysis.

Tests whether the manuscript's core conclusions are stable under different
evaluation protocols. For every cohort (internal / RSNA / NIH) we compute:

  A. manuscript protocol (inference + ECE style per ``evaluation.protocols``)
  B. single-pass raw probabilities, label-rate ECE
  C. MC Dropout mean (30 passes, fitted T), label-rate ECE
  D. single-pass temperature-scaled, label-rate ECE
  E. manuscript inference, confidence ECE

This run is also the **primary external-referral verification**: MC Dropout
referral curves (vs random control) are computed for RSNA and NIH from the new
pipeline, closing the in-domain-works / out-of-domain-fails story with
reproducible code.

Predictions are cached under ``results/metrics/predictions/`` so re-runs are
fast.

Usage:
    python scripts/medknow_run_protocol_sensitivity.py --weights checkpoints/seed_42.pth
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

from medknow.calibration.metrics import (
    compute_brier,
    compute_ece_confidence,
)
from medknow.calibration.temperature_scaling import apply_temperature, collect_logits
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
from medknow.uncertainty.mc_dropout import estimate_mc_dropout

PROTOCOL_SINGLE = "single_pass_raw"
PROTOCOL_MC = "mc_mean_scaled"
ECE_LABEL_RATE = "label_rate"
ECE_CONFIDENCE = "confidence"


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


def _pred_dir(cfg) -> Path:
    return Path(get(cfg, "paths.metrics_dir")) / "predictions"


def _load_or_compute(cfg, cache_name, compute_fn, recompute):
    cache = _pred_dir(cfg) / f"{cache_name}.npz"
    if cache.exists() and not recompute:
        data = np.load(cache)
        return {k: data[k] for k in data.files}
    out = compute_fn()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


def _loader(cfg, dataset, batch_size, num_workers):
    return make_loader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


def _metrics(labels, p_pos, ece_style):
    m = compute_metrics(labels, p_pos)
    if ece_style == ECE_CONFIDENCE:
        m["ece_raw"] = compute_ece_confidence(p_pos, labels, n_bins=15)
    m["brier_raw"] = compute_brier(p_pos, labels)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--mc_samples", type=int, default=None)
    ap.add_argument("--recompute", action="store_true", help="ignore caches")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N images per cohort")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    image_size = int(get(cfg, "data.image_size"))
    batch_size = int(get(cfg, "data.batch_size"))
    num_workers = int(get(cfg, "data.num_workers"))
    n_samples = args.mc_samples or int(get(cfg, "evaluation.mc_samples", 30))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = _resolve_temperature(cfg, args.temperature)

    model = load_trained_model(args.weights, num_classes=2, device=device)
    transform = base_transform(image_size)
    cohorts = {
        "internal_test": build_internal_dataset(get(cfg, "paths.split_dir"), "test", transform),
        "rsna": build_rsna_dataset(get(cfg, "data.external.rsna.root"), transform),
        "nih": build_nih_dataset(
            Path(get(cfg, "data.external.nih.root")) / "test_resized" / "test",
            transform,
        ),
    }

    table = []
    referral_summary = {}
    for name, dataset in cohorts.items():
        if args.limit is not None:
            if hasattr(dataset, "samples"):
                dataset.samples = dataset.samples[: args.limit]
            elif hasattr(dataset, "image_paths"):
                dataset.image_paths = dataset.image_paths[: args.limit]
                dataset.labels = dataset.labels[: args.limit]
                dataset.subgroups = dataset.subgroups[: args.limit]
                dataset.patient_ids = dataset.patient_ids[: args.limit]
        loader = _loader(cfg, dataset, batch_size, num_workers)
        proto = get(cfg, f"evaluation.protocols.{name}", {}) or {}
        inference = proto.get("inference", PROTOCOL_SINGLE)
        ece_style = proto.get("ece_style", ECE_LABEL_RATE)

        logger.info("[%s] single-pass probs", name)
        single = _load_or_compute(
            cfg, f"{name}_single",
            lambda loader=loader: _single(cfg, model, loader, device), args.recompute,
        )
        labels, probs_s = single["labels"], single["probs"]

        logger.info("[%s] single-pass logits", name)
        logits_data = _load_or_compute(
            cfg, f"{name}_logits",
            lambda loader=loader: _logits(model, loader, device), args.recompute,
        )
        logits = logits_data["logits"]

        logger.info("[%s] MC Dropout (%d samples, T=%.2f)", name, n_samples, temperature)
        mc = _load_or_compute(
            cfg, f"{name}_mc",
            lambda loader=loader: _mc(cfg, model, loader, n_samples, temperature, device), args.recompute,
        )
        mean_p, std = mc["mean_probs"], mc["std"]

        p_s = probs_s[:, 1]
        p_mc = mean_p[:, 1]
        p_scaled = apply_temperature(logits, temperature)[:, 1]

        protocols = {
            "A_manuscript": _metrics(labels, p_mc if inference == PROTOCOL_MC else p_s, ece_style),
            "B_single_raw": _metrics(labels, p_s, ECE_LABEL_RATE),
            "C_mc_scaled": _metrics(labels, p_mc, ECE_LABEL_RATE),
            "D_single_scaled": _metrics(labels, p_scaled, ECE_LABEL_RATE),
            "E_conf_ece": _metrics(labels, p_mc if inference == PROTOCOL_MC else p_s, ECE_CONFIDENCE),
        }
        for pkey, m in protocols.items():
            table.append({
                "cohort": name,
                "protocol": pkey,
                "auc": m["auc"],
                "auprc": m["auprc"],
                "accuracy": m["accuracy"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
                "ece": m["ece_raw"],
                "brier": m["brier_raw"],
            })

        # Referral (MC Dropout vs random) from the new pipeline.
        rates = list(get(cfg, "referral.rates"))
        preds_mc = (p_mc >= 0.5).astype(int)
        mc_curves = referral_curves(labels, preds_mc, std, rates)
        random_curves = random_referral_curves(labels, preds_mc, rates, n_trials=100, seed=42)
        ep_mc = error_prediction_auc(labels, preds_mc, std)
        ep_msp = error_prediction_auc(labels, preds_mc, 1.0 - np.maximum(p_mc, 1 - p_mc))
        referral_summary[name] = {
            "rates": rates,
            "mc_retained_error": mc_curves["retained_error"],
            "mc_retained_missed": mc_curves["retained_missed"],
            "random_retained_error": random_curves["retained_error_mean"],
            "error_prediction_auc_mc": ep_mc,
            "error_prediction_auc_msp": ep_msp,
        }
        logger.info("[%s] done", name)

    report = {
        "weights": args.weights,
        "temperature": temperature,
        "mc_samples": n_samples,
        "table": table,
        "referral_summary": referral_summary,
    }
    out_path = args.out or str(
        Path(get(cfg, "paths.tables_dir")) / "protocol_sensitivity.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    print("\n=== Protocol sensitivity (AUC | ECE | Brier) ===")
    for row in table:
        print(f"{row['cohort']:14s} {row['protocol']:16s} "
              f"AUC {_fmt(row['auc'])}  ECE {_fmt(row['ece'])}  Brier {_fmt(row['brier'])}")
    print("\n=== Referral (retained error @ 25%, MC vs random) ===")
    for name, s in referral_summary.items():
        i = rates.index(0.25)
        ep = "n/a" if s['error_prediction_auc_mc'] is None else f"{s['error_prediction_auc_mc']:.3f}"
        print(f"{name:14s} MC {_fmt(s['mc_retained_error'][i])}  "
              f"random {_fmt(s['random_retained_error'][i])}  "
              f"epAUC_mc {ep}")


def _single(cfg, model, loader, device):
    labels, probs = predict_batch_probs(model, loader, temperature=1.0)
    return {"labels": labels, "probs": probs}


def _logits(model, loader, device):
    labels, logits = collect_logits(model, loader, device)
    return {"labels": labels, "logits": logits}


def _mc(cfg, model, loader, n_samples, temperature, device):
    result = estimate_mc_dropout(
        model, loader, n_samples=n_samples, temperature=temperature, device=device
    )
    return {
        "labels": result.labels,
        "mean_probs": result.probs,
        "std": result.scores,
    }


if __name__ == "__main__":
    main()
