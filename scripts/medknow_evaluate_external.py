#!/usr/bin/env python3
"""MedKnow — unified domain-shift evaluation (Internal / RSNA / NIH).

Computes the manuscript metric set for all three cohorts with one command and
writes a summary table (results/tables/domain_shift_summary.json).

Protocol note: the manuscript used per-cohort evaluation protocols:

- inference: ``single_pass_raw`` (one forward pass, raw probabilities) for the
  internal test and RSNA; ``mc_mean_scaled`` (MC Dropout mean over 30 passes at
  T = 1.67) for NIH ChestXray-14;
- ECE style: ``label_rate`` (bins by positive-class probability) for internal
  and RSNA; ``confidence`` (bins by max confidence vs argmax accuracy) for NIH.

These choices are recorded in ``evaluation.protocols`` in the config so the
manuscript numbers are reproduced exactly and the differences are explicit.

Usage:
    python scripts/medknow_evaluate_external.py --weights checkpoints/seed_42.pth
    python scripts/medknow_evaluate_external.py --weights checkpoints/seed_42.pth --only nih
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

from medknow.calibration.metrics import (
    compute_brier,
    compute_ece,
    compute_ece_confidence,
)
from medknow.config import get, load_config
from medknow.datasets.chest_xray import (
    base_transform,
    build_internal_dataset,
    build_nih_dataset,
    build_rsna_dataset,
    make_loader,
)
from medknow.evaluation.metrics import bootstrap_confidence_intervals, compute_metrics
from medknow.evaluation.subgroups import subgroup_table
from medknow.models.factory import load_trained_model
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


def _ece(probs, labels, style, n_bins):
    p = probs[:, 1]
    if style == ECE_CONFIDENCE:
        return compute_ece_confidence(p, labels, n_bins=n_bins)
    return compute_ece(p, labels, n_bins=n_bins)


def _evaluate_cohort(
    model,
    dataset,
    *,
    inference,
    ece_style,
    temperature,
    n_bins,
    n_samples,
    batch_size,
    num_workers,
    device,
    n_bootstrap,
    bootstrap_seed,
):
    loader = make_loader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    if inference == PROTOCOL_MC:
        result = estimate_mc_dropout(
            model, loader, n_samples=n_samples, temperature=temperature, device=device
        )
        labels, probs = result.labels, result.probs
    else:
        labels, probs = predict_batch_probs(model, loader, temperature=1.0)

    row = compute_metrics(labels, probs[:, 1], n_bins=n_bins)
    row["confidence_intervals"] = bootstrap_confidence_intervals(
        labels,
        probs[:, 1],
        groups=getattr(dataset, "patient_ids", None),
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
        n_bins=n_bins,
    )
    row["ece_raw"] = _ece(probs, labels, ece_style, n_bins)
    row["inference"] = inference
    row["ece_style"] = ece_style
    if inference == PROTOCOL_SINGLE and temperature != 1.0:
        _, scaled = predict_batch_probs(model, loader, temperature=temperature)
        row["ece_scaled"] = _ece(scaled, labels, ece_style, n_bins)
        row["brier_scaled"] = compute_brier(scaled[:, 1], labels)
    return row, labels, probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--mc_samples", type=int, default=None)
    ap.add_argument("--only", default=None, choices=["internal_test", "rsna", "nih"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N images per cohort")
    ap.add_argument("--bootstrap-iters", type=int, default=None)
    ap.add_argument("--bootstrap-seed", type=int, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    image_size = int(get(cfg, "data.image_size"))
    batch_size = int(get(cfg, "data.batch_size"))
    num_workers = int(get(cfg, "data.num_workers"))
    ece_bins = int(get(cfg, "calibration.ece_bins", 15))
    n_samples = args.mc_samples or int(get(cfg, "evaluation.mc_samples", 30))
    n_bootstrap = args.bootstrap_iters
    if n_bootstrap is None:
        n_bootstrap = int(get(cfg, "evaluation.bootstrap_iters", 2000))
    bootstrap_seed = int(
        args.bootstrap_seed
        if args.bootstrap_seed is not None
        else get(cfg, "evaluation.bootstrap_seed", 42)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temperature = _resolve_temperature(cfg, args.temperature)

    model = load_trained_model(args.weights, num_classes=2, device=device)
    transform = base_transform(image_size)

    cohorts = [
        (
            "internal_test",
            "internal",
            build_internal_dataset(get(cfg, "paths.split_dir"), "test", transform),
            None,
        ),
        (
            "rsna",
            "external",
            build_rsna_dataset(get(cfg, "data.external.rsna.root"), transform),
            "rsna_subgroups",
        ),
        (
            "nih",
            "external",
            build_nih_dataset(
                Path(get(cfg, "data.external.nih.root")) / "test_resized" / "test",
                transform,
            ),
            None,
        ),
    ]

    table = []
    subgroups_report = None
    for name, vtype, dataset, subgroup_key in cohorts:
        if args.only is not None and name != args.only:
            continue
        if args.limit is not None and hasattr(dataset, "samples"):
            dataset.samples = dataset.samples[: args.limit]
            if hasattr(dataset, "targets"):
                dataset.targets = dataset.targets[: args.limit]
            if hasattr(dataset, "patient_ids"):
                dataset.patient_ids = dataset.patient_ids[: args.limit]
        proto = get(cfg, f"evaluation.protocols.{name}", {}) or {}
        inference = proto.get("inference", PROTOCOL_SINGLE)
        ece_style = proto.get("ece_style", ECE_LABEL_RATE)
        logger.info(
            "Evaluating cohort: %s (%d images, inference=%s, ece_style=%s)",
            name, len(dataset), inference, ece_style,
        )
        row, labels, probs = _evaluate_cohort(
            model, dataset,
            inference=inference, ece_style=ece_style, temperature=temperature,
            n_bins=ece_bins, n_samples=n_samples, batch_size=batch_size,
            num_workers=num_workers, device=device,
            n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed,
        )
        row.update({"dataset": name, "validation_type": vtype, "temperature": temperature})
        table.append(row)

        if subgroup_key == "rsna_subgroups":
            subgroups_report = subgroup_table(
                labels, probs[:, 1], dataset.subgroups,
                names=["Lung Opacity", "No Lung Opacity / Not Normal", "Normal"],
            )

    report = {"table": table, "rsna_subgroups": subgroups_report}
    out_path = args.out or str(
        Path(get(cfg, "paths.tables_dir")) / "domain_shift_summary.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    header = ["dataset", "inference", "ece_style", "n", "auc", "auprc", "accuracy", "sensitivity", "specificity", "ece_raw", "brier_raw"]
    print(" | ".join(header))
    for row in table:
        print(" | ".join(str(row.get(k, "")) for k in header))
    if subgroups_report:
        print("\nRSNA subgroups:")
        for name, s in subgroups_report.items():
            print(
                f"  {name}: n={s['n']} mean_prob={s['mean_prob']:.3f} "
                f"pred_pos={s['pos_rate_pred']:.3f} "
                f"fp={s['fp_contribution']} ({s['fp_share']:.1%})"
            )


if __name__ == "__main__":
    main()
