"""Subgroup analysis (RSNA Lung Opacity / No LO / Normal)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def subgroup_table(
    labels: Sequence[int],
    p_pos: Sequence[float],
    subgroups: Sequence[str],
    names: Sequence[str] | None = None,
) -> dict:
    """Per-subgroup statistics for the manuscript RSNA subgroup analysis.

    Returns, per subgroup: sample count, positive rate, mean predicted
    probability, predicted-positive rate, false-positive contribution and its
    share of all false positives.
    """
    labels = np.asarray(labels, dtype=int)
    p_pos = np.asarray(p_pos, dtype=float)
    subgroups = list(subgroups)
    preds_pos = p_pos >= 0.5
    fp = preds_pos & (labels == 0)
    fp_total = int(fp.sum())

    result: dict = {}
    for name in names or sorted(set(subgroups)):
        mask = np.array([s == name for s in subgroups])
        n = int(mask.sum())
        if n == 0:
            result[name] = {
                "n": 0, "positive_rate": None, "mean_prob": None,
                "pos_rate_pred": None, "fp_contribution": 0,
                "fp_share": 0.0,
            }
            continue
        fp_contrib = int(fp[mask].sum())
        result[name] = {
            "n": n,
            "positive_rate": float(labels[mask].mean()),
            "mean_prob": float(p_pos[mask].mean()),
            "pos_rate_pred": float(preds_pos[mask].mean()),
            "fp_contribution": fp_contrib,
            "fp_share": fp_contrib / fp_total if fp_total > 0 else 0.0,
        }
    return result
