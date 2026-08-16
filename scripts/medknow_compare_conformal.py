#!/usr/bin/env python3
"""MedKnow — Experiment C: does conformal prediction survive domain shift?

Fits a LAC conformal threshold on (a split of) the internal test set, then
measures empirical coverage and prediction-set sizes on the internal held-out
split and on two external cohorts (RSNA, NIH ChestXray-14).

Prediction: coverage ~= 1 - alpha in-domain, but collapses under domain shift —
conformal prediction, like MC Dropout / MSP / ensembles, does not rescue the
referral problem.

Usage:
    python scripts/medknow_compare_conformal.py \
        --internal results/metrics/predictions/internal_test_logits.npz \
        --cohorts results/metrics/predictions/rsna_logits.npz \
                 results/metrics/predictions/nih_logits.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medknow.uncertainty.conformal import fit_lac_threshold, prediction_sets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--internal", required=True)
    ap.add_argument("--cohorts", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--cal-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.internal)
    cal_logits = np.asarray(d["logits"])
    cal_labels = np.asarray(d["labels"]).astype(int)
    n = len(cal_labels)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)
    cal_n = int(n * args.cal_frac)
    cal_i, eval_i = idx[:cal_n], idx[cal_n:]
    cal_l, cal_lb = cal_logits[cal_i], cal_labels[cal_i]
    internal_eval = (cal_logits[eval_i], cal_labels[eval_i])

    q = fit_lac_threshold(cal_l, cal_lb, alpha=args.alpha)
    results = {
        "alpha": args.alpha,
        "coverage_target": 1.0 - args.alpha,
        "q": q,
        "cohorts": {},
    }

    def eval_cohort(name: str, logits: np.ndarray, labels: np.ndarray) -> None:
        sizes, memberships = prediction_sets(logits, q)
        covered = memberships[np.arange(len(labels)), labels]
        results["cohorts"][name] = {
            "n": len(labels),
            "empirical_coverage": float(covered.mean()),
            "mean_set_size": float(sizes.mean()),
            "frac_set_size_1": float((sizes == 1).mean()),
            "frac_set_size_0or2": float((sizes != 1).mean()),
        }

    eval_cohort("internal_heldout", *internal_eval)
    for cpath in args.cohorts:
        cd = np.load(cpath)
        eval_cohort(Path(cpath).stem, np.asarray(cd["logits"]), np.asarray(cd["labels"]).astype(int))

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
