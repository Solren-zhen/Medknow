"""Tests for unified evaluation metrics and subgroup analysis."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medknow.evaluation.metrics import bootstrap_confidence_intervals, compute_metrics
from medknow.evaluation.subgroups import subgroup_table


def test_compute_metrics_perfect_separation():
    labels = np.array([0, 0, 1, 1])
    p_pos = np.array([0.1, 0.2, 0.9, 0.8])
    m = compute_metrics(labels, p_pos)
    assert m["auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["confusion"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}


def test_compute_metrics_known_confusion():
    labels = np.array([1, 1, 0, 0])
    p_pos = np.array([0.9, 0.4, 0.6, 0.1])
    m = compute_metrics(labels, p_pos)
    assert m["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert m["accuracy"] == 0.5


def test_subgroup_table_counts_and_fp_shares():
    labels = np.array([1, 0, 0, 0, 1, 0])
    p_pos = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.6])
    subgroups = ["Lung Opacity", "No LO", "Normal", "Normal", "Lung Opacity", "No LO"]
    table = subgroup_table(
        labels, p_pos, subgroups,
        names=["Lung Opacity", "No LO", "Normal"],
    )
    assert table["Lung Opacity"]["n"] == 2
    assert table["Normal"]["n"] == 2
    # FPs: No LO index1 (0.8>0.5, label0), Normal index3 (0.1 no), No LO index5 (0.6, label0)
    fp_total = sum(v["fp_contribution"] for v in table.values())
    assert fp_total == 2
    assert abs(sum(v["fp_share"] for v in table.values()) - 1.0) < 1e-12


def test_patient_cluster_bootstrap_keeps_groups_together():
    labels = np.array([0, 0, 1, 1, 0, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    groups = np.array(["p1", "p1", "p2", "p2", "p3", "p3"])
    result = bootstrap_confidence_intervals(
        labels, probs, groups=groups, n_bootstrap=25, seed=7
    )
    assert result["bootstrap"]["unit"] == "group"
    assert result["bootstrap"]["n_units"] == 3
    assert 0 < result["auc"]["n_valid"] <= 25


def test_calibration_parameters_are_reported():
    metrics = compute_metrics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    )
    assert metrics["calibration_intercept"] is not None
    assert metrics["calibration_slope"] is not None
