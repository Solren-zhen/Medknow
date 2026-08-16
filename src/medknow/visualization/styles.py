"""Unified publication style for MedKnow figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "mc": "#1f77b4",
    "confidence": "#ff7f0e",
    "random": "#7f7f7f",
    "internal": "#1f77b4",
    "rsna": "#d62728",
    "nih": "#2ca02c",
}

COHORT_LABELS = {
    "internal_test": "Internal (Kermany)",
    "rsna": "External: RSNA",
    "nih": "External: NIH",
}


def apply_style() -> None:
    """Apply the MedKnow figure style (300 DPI, unified fonts, no 3D)."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
    })
