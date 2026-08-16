"""Smoke tests for the MedKnow figure builders (Agg backend, no display)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _synthetic():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 100)
    p_pos = np.clip(labels + rng.normal(0, 0.3, 100), 0.02, 0.98)
    return labels, p_pos


def test_figure_builders_produce_files(tmp_path):
    labels, p_pos = _synthetic()
    rates = [0.0, 0.1, 0.25, 0.5]
    err = [0.2, 0.15, 0.1, 0.05]
    rand = [0.2, 0.2, 0.2, 0.2]
    report = {
        "Lung Opacity": {"mean_prob": 0.9, "n": 60},
        "No Lung Opacity / Not Normal": {"mean_prob": 0.7, "n": 118},
        "Normal": {"mean_prob": 0.2, "n": 88},
    }
    table = [
        {"dataset": "internal_test", "auc": 0.99, "auprc": 0.99, "ece_raw": 0.03},
        {"dataset": "rsna", "auc": 0.81, "auprc": 0.51, "ece_raw": 0.37},
        {"dataset": "nih", "auc": 0.66, "auprc": 0.01, "ece_raw": 0.27},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, name in zip(axes, ("internal_test", "rsna", "nih")):
        fig_roc(ax, labels, p_pos, name, "0.99")
        fig_pr(ax, labels, p_pos, name, "0.50")
        fig_calibration(ax, p_pos, labels, name, "0.03", "0.03")
    fig.savefig(tmp_path / "fig02_roc.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    fig_referral(ax, rates, err, err, rand, "internal_test")
    fig_risk_coverage(ax, rates, err, "rsna")
    fig.savefig(tmp_path / "fig05_referral.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    fig_subgroup(ax, report)
    fig_pipeline(ax)
    fig.savefig(tmp_path / "fig07_subgroup.png")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2)
    fig_domain_summary(a1, a2, table)
    fig.savefig(tmp_path / "fig08_domain.png")
    plt.close(fig)

    for name in ("fig02_roc.png", "fig05_referral.png", "fig07_subgroup.png", "fig08_domain.png"):
        assert (tmp_path / name).exists(), name
