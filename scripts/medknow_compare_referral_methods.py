#!/usr/bin/env python3
"""MedKnow — referral-methods comparison (Random / MSP / MC Dropout / Ensemble).

Answers the question: does a sophisticated uncertainty estimator beat plain
model confidence for referral? For every cohort we compute the retained-case
error curve and error-prediction AUC for:

- MSP (plain confidence: 1 − max probability, single-pass);
- MC Dropout (std of the pneumonia probability, 30 passes);
- Deep Ensemble (across-member std, three seed checkpoints);
- Random referral (matched-rate control).

Reads prediction caches written by
``scripts/medknow_run_protocol_sensitivity.py`` and
``scripts/medknow_run_ensemble.py`` (``results/metrics/predictions/``).

Usage:
    python scripts/medknow_compare_referral_methods.py
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
    error_prediction_auc,
    random_referral_curves,
    referral_curves,
)

logger = logging.getLogger(__name__)


def _load_npz(path: Path) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _curve(labels, preds, scores, rates):
    return referral_curves(labels, preds, scores, rates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    rates = list(get(cfg, "referral.rates"))
    pred_dir = Path(get(cfg, "paths.metrics_dir")) / "predictions"

    summary = {}
    for name in ("internal_test", "rsna", "nih"):
        single = _load_npz(pred_dir / f"{name}_single.npz")
        mc = _load_npz(pred_dir / f"{name}_mc.npz")
        ensemble = _load_npz(pred_dir / f"{name}_ensemble.npz")
        labels = single["labels"]

        p_s = single["probs"][:, 1]
        preds_s = (p_s >= 0.5).astype(int)
        scores_s = 1.0 - single["probs"].max(axis=1)
        msp = _curve(labels, preds_s, scores_s, rates)
        ep_msp = error_prediction_auc(labels, preds_s, scores_s)

        p_mc = mc["mean_probs"][:, 1]
        preds_mc = (p_mc >= 0.5).astype(int)
        mcd = _curve(labels, preds_mc, mc["std"], rates)
        ep_mc = error_prediction_auc(labels, preds_mc, mc["std"])

        member_p = ensemble["member_probs"][:, :, 1]
        p_ens = member_p.mean(axis=0)
        preds_ens = (p_ens >= 0.5).astype(int)
        ens = _curve(labels, preds_ens, member_p.std(axis=0), rates)
        ep_ens = error_prediction_auc(labels, preds_ens, member_p.std(axis=0))

        random_curves = random_referral_curves(
            labels, preds_mc, rates, n_trials=100, seed=42
        )
        summary[name] = {
            "rates": rates,
            "msp": msp,
            "mc_dropout": mcd,
            "ensemble": ens,
            "random": random_curves,
            "error_prediction_auc": {
                "msp": ep_msp,
                "mc_dropout": ep_mc,
                "ensemble": ep_ens,
            },
        }
        logger.info("[%s] done", name)

    out_path = args.out or str(
        Path(get(cfg, "paths.tables_dir")) / "referral_methods_summary.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    i25 = rates.index(0.25)
    print("\n=== Referral methods @ 25% (retained error | epAUC) ===")
    print(f"{'cohort':14s} {'random':>12s} {'MSP':>14s} {'MC':>14s} {'Ensemble':>14s}")
    for name, s in summary.items():
        r = s["random"]["retained_error_mean"][i25]
        msp = s["msp"]["retained_error"][i25]
        mc = s["mc_dropout"]["retained_error"][i25]
        ens = s["ensemble"]["retained_error"][i25]
        print(f"{name:14s} {r:12.4f} {msp:14.4f} {mc:14.4f} {ens:14.4f}")
    print("\nError-prediction AUC:")
    for name, s in summary.items():
        e = s["error_prediction_auc"]
        print(f"  {name:14s} MSP {e['msp']:.3f}  MC {e['mc_dropout']:.3f}  "
              f"Ensemble {e['ensemble']:.3f}")


if __name__ == "__main__":
    main()
