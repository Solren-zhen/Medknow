#!/usr/bin/env python3
"""MedKnow — Experiment B: can external re-calibration rescue the referral signal?

The manuscript shows that uncertainty-driven referral fails under domain shift
(error-prediction AUC ~= 0.5 on RSNA and NIH). This experiment tests the
standard remedy: fit a temperature on a held-out calibration subset of an
EXTERNAL cohort, apply it to the full external cohort, and re-measure
calibration (ECE/Brier) and the referral signal (error-prediction AUC,
retained error at 25% referral).

Usage:
    python scripts/medknow_external_recalibration.py \
        --npz results/metrics/predictions/rsna_logits.npz
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medknow.calibration.metrics import compute_brier, compute_ece
from medknow.calibration.temperature_scaling import apply_temperature, fit_temperature
from medknow.referral.curves import (
    error_prediction_auc,
    random_referral_curves,
    referral_curves,
)

logger = logging.getLogger(__name__)


def _msp_scores(probs: np.ndarray) -> np.ndarray:
    """Plain-confidence signal: 1 - max softmax probability."""
    return 1.0 - probs.max(axis=1)


def _at(curves: dict, rate: float) -> float | None:
    for r, e in zip(curves["rates"], curves["retained_error"]):
        if abs(float(r) - rate) < 1e-9:
            return e
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", required=True, help="external logits npz with keys labels/logits")
    ap.add_argument("--cal-frac", type=float, default=0.2, help="held-out calibration fraction")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = np.load(args.npz)
    logits = np.asarray(data["logits"])
    labels = np.asarray(data["labels"]).astype(int)
    n = len(labels)

    rng = np.random.default_rng(args.seed)
    cal_idx = rng.choice(n, size=int(n * args.cal_frac), replace=False)
    te_idx = np.setdiff1d(np.arange(n), cal_idx)
    cal_logits, cal_labels = logits[cal_idx], labels[cal_idx]
    te_logits, te_labels = logits[te_idx], labels[te_idx]

    t_ext = fit_temperature(cal_logits, cal_labels)
    probs_raw = apply_temperature(te_logits, 1.0)
    probs_cal = apply_temperature(te_logits, t_ext)
    preds = probs_raw.argmax(axis=1)

    def block(name: str, probs: np.ndarray) -> dict:
        p1 = probs[:, 1]
        scores = _msp_scores(probs)
        curves = referral_curves(te_labels, preds, scores)
        rnd = random_referral_curves(te_labels, preds)
        rnd_25 = None
        for r, e in zip(rnd["rates"], rnd["retained_error_mean"]):
            if abs(float(r) - 0.25) < 1e-9:
                rnd_25 = e
                break
        return {
            "name": name,
            "ece": compute_ece(p1, te_labels),
            "brier": compute_brier(p1, te_labels),
            "error_prediction_auc": error_prediction_auc(te_labels, preds, scores),
            "retained_error_25": _at(curves, 0.25),
            "random_retained_error_25": rnd_25,
        }

    raw = block("raw", probs_raw)
    scaled = block("T_ext", probs_cal)

    summary = {
        "cohort": Path(args.npz).stem,
        "n": n,
        "cal_n": len(cal_idx),
        "test_n": len(te_idx),
        "t_ext": t_ext,
        "raw": raw,
        "scaled": scaled,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("saved -> %s", out_p)


if __name__ == "__main__":
    main()
