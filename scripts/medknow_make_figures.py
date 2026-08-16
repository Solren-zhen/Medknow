#!/usr/bin/env python3
"""MedKnow — regenerate all publication figures (fig01–fig08).

Single-pass and MC Dropout predictions are read from the caches written by
``scripts/medknow_run_protocol_sensitivity.py``
(``results/metrics/predictions/*_single.npz``, ``*_mc.npz``). Run that script
first so every figure is generated from the new pipeline:

    python scripts/medknow_run_protocol_sensitivity.py --weights checkpoints/seed_42.pth
    python scripts/medknow_make_figures.py --weights checkpoints/seed_42.pth

Referral curves in fig05 use MC Dropout (std), plain confidence (1 − max prob)
and a matched random control, computed from the caches for all three cohorts.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from medknow.config import get, load_config
from medknow.referral.curves import (
    random_referral_curves,
    referral_curves,
    risk_coverage,
)
from medknow.visualization.figures import (
    fig_calibration,
    fig_domain_summary,
    fig_pipeline,
    fig_pr,
    fig_referral,
    fig_risk_coverage,
    fig_roc,
    fig_subgroup,
)
from medknow.visualization.styles import COHORT_LABELS, apply_style

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--weights", default="checkpoints/seed_42.pth")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    apply_style()
    cfg = load_config(args.config)
    ece_bins = int(get(cfg, "calibration.ece_bins", 15))
    rates = list(get(cfg, "referral.rates"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(get(cfg, "paths.metrics_dir")) / "predictions"

    summary = _load_json(
        Path(get(cfg, "paths.tables_dir")) / "domain_shift_summary.json"
    )
    table = summary["table"]
    cohorts = ("internal_test", "rsna", "nih")

    probs_data = {}
    for name in cohorts:
        single = _load_npz(pred_dir / f"{name}_single.npz")
        mc = _load_npz(pred_dir / f"{name}_mc.npz")
        probs_data[name] = {
            "labels": single["labels"],
            "probs": single["probs"],
            "mc_mean": mc["mean_probs"],
            "mc_std": mc["std"],
        }

    # --- fig01 pipeline ---
    fig, ax = plt.subplots(figsize=(7, 5))
    fig_pipeline(ax)
    fig.savefig(out_dir / "fig01_pipeline.png")
    plt.close(fig)

    # --- fig02 ROC / fig03 PR ---
    aucs = {r["dataset"]: r["auc"] for r in table}
    auprcs = {r["dataset"]: r["auprc"] for r in table}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for name in cohorts:
        labels = probs_data[name]["labels"]
        p_pos = probs_data[name]["probs"][:, 1]
        fig_roc(axes[0], labels, p_pos, name, f"{aucs[name]:.3f}")
        fig_pr(axes[1], labels, p_pos, name, f"{auprcs[name]:.3f}")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="upper right")
    fig.savefig(out_dir / "fig02_roc.png")
    fig.savefig(out_dir / "fig03_pr.png")
    plt.close(fig)

    # --- fig04 calibration ---
    eces = {r["dataset"]: r["ece_raw"] for r in table}
    briers = {r["dataset"]: r["brier_raw"] for r in table}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, name in zip(axes, cohorts):
        labels = probs_data[name]["labels"]
        p_pos = probs_data[name]["probs"][:, 1]
        fig_calibration(ax, p_pos, labels, name, f"{eces[name]:.3f}",
                        f"{briers[name]:.3f}", n_bins=ece_bins)
        ax.set_title(COHORT_LABELS[name])
    axes[0].legend(loc="lower right", fontsize=7)
    fig.savefig(out_dir / "fig04_calibration.png")
    plt.close(fig)

    # --- fig05 referral (all from new-pipeline caches) ---
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, name in zip(axes, cohorts):
        data = probs_data[name]
        labels = data["labels"]
        preds_mc = (data["mc_mean"][:, 1] >= 0.5).astype(int)
        preds_single = (data["probs"][:, 1] >= 0.5).astype(int)
        mc_err = referral_curves(labels, preds_mc, data["mc_std"], rates)["retained_error"]
        conf_err = referral_curves(
            labels, preds_single, 1.0 - data["probs"].max(axis=1), rates
        )["retained_error"]
        rand_err = random_referral_curves(
            labels, preds_mc, rates, n_trials=100, seed=42
        )["retained_error_mean"]
        fig_referral(ax, rates, mc_err, conf_err, rand_err, name)
        ax.set_title(COHORT_LABELS[name])
    axes[0].legend(loc="upper right", fontsize=8)
    fig.savefig(out_dir / "fig05_referral.png")
    plt.close(fig)

    # --- fig06 risk-coverage (confidence-based) ---
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in cohorts:
        data = probs_data[name]
        scores = 1.0 - data["probs"].max(axis=1)
        preds = data["probs"].argmax(axis=1)
        rc = risk_coverage(data["labels"], preds, scores)
        fig_risk_coverage(ax, rc["coverage"], rc["risk"], name)
    ax.legend(loc="lower right")
    fig.savefig(out_dir / "fig06_risk_coverage.png")
    plt.close(fig)

    # --- fig07 subgroup ---
    fig, ax = plt.subplots(figsize=(6.5, 4))
    fig_subgroup(ax, summary["rsna_subgroups"])
    fig.savefig(out_dir / "fig07_subgroup.png")
    plt.close(fig)

    # --- fig08 domain-shift summary ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    fig_domain_summary(ax1, ax2, table)
    fig.savefig(out_dir / "fig08_domain_shift_summary.png")
    plt.close(fig)

    print(f"Figures written to {out_dir}")
    for f in sorted(out_dir.glob("fig*.png")):
        print(" ", f.name, f.stat().st_size)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
