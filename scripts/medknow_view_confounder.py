#!/usr/bin/env python3
"""MedKnow — AP/PA view-position confounder analysis (NIH external cohort).

ROADMAP V1.2 research direction #1. The internal Kermany cohort is PA-only,
while the NIH ChestXray-14 two-class external test set mixes PA and AP views.
Because the folder-based external loader drops the ``View Position`` metadata
carried in the NIH CSV, the manuscript's external AUC collapse conflates
(a) hospital/scanner/population shift and (b) acquisition-view shift.

This script quantifies (b):

1. Re-scoring the NIH external test set with the seed-42 checkpoint
   (single-pass raw probabilities, the internal/RSNA manuscript protocol);
2. Joining per-image predictions to the NIH CSV on ``Image Index``;
3. Stratifying AUC / ECE / sensitivity / specificity / referral-signal
   (error-prediction AUC of the low-confidence score) by view;
4. Contrasting the internal PA-only anchor with the NIH PA stratum
   (matched-view domain shift) vs. the NIH AP stratum (matched-shift + view).

Note: the manuscript's headline NIH number uses the ``mc_mean_scaled``
protocol (MC Dropout mean, 30 passes, T=1.67). View stratification here uses
single-pass probabilities for tractability; the overall single-pass AUC is
reported alongside the manuscript anchor for transparency.

Usage:
    python scripts/medknow_view_confounder.py \
        --weights ../pneumonia_classifier/checkpoints/seed_42.pth \
        --nih-root ../pneumonia_classifier/data/external/nih2class
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from medknow.config import get, load_config
from medknow.datasets.chest_xray import base_transform, build_internal_dataset
from medknow.models.factory import load_trained_model
from medknow.training.inference import predict_batch_probs

DEFAULT_WEIGHTS = PROJECT_ROOT.parent / "pneumonia_classifier" / "checkpoints" / "seed_42.pth"
DEFAULT_NIH_ROOT = PROJECT_ROOT.parent / "pneumonia_classifier" / "data" / "external" / "nih2class"

BOOTSTRAP_SEED = 42


def rank_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC via rank statistics (ties handled with mid-ranks)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auc_ci(y_true, scores, n_bootstrap=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    for _ in range(n_bootstrap):
        p = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ng = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p, ng])
        aucs.append(rank_auc(y_true[idx], scores[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def sens_spec_at_threshold(y_true, scores, t=0.5):
    pred = (scores >= t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {"sensitivity": sens, "specificity": spec, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def ece_label_rate(p_pos: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE over the positive-class probability (``label_rate`` style, repo default)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p_pos, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(y_true[m].mean() - p_pos[m].mean())
    return float(ece)


def error_pred_auc(y_true: np.ndarray, p_pos: np.ndarray) -> float:
    """AUC of the low-confidence score for predicting argmax errors (repo convention)."""
    pred = (p_pos >= 0.5).astype(int)
    err = (pred != y_true).astype(int)
    score = 1.0 - np.maximum(p_pos, 1.0 - p_pos)
    return rank_auc(err, score)


def stratum_report(y_true, scores, name):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    auc = rank_auc(y, s)
    lo, hi = bootstrap_auc_ci(y, s)
    rep = {
        "stratum": name,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()),
        "auc": auc,
        "auc_ci95": [lo, hi],
        "ece_label_rate": ece_label_rate(s, y),
        "error_pred_auc_low_confidence": error_pred_auc(y, s),
    }
    rep.update(sens_spec_at_threshold(y, s))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--nih-root", default=str(DEFAULT_NIH_ROOT))
    ap.add_argument("--split-dir", default=str(PROJECT_ROOT / "data" / "split_patient"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "tables" / "view_confounder_nih.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    cfg = load_config(args.config)
    transform = base_transform(int(get(cfg, "data.image_size", 224)))
    model = load_trained_model(args.weights, device=device)

    # ---- internal test anchor (PA-only cohort, manuscript protocol single_pass_raw) ----
    internal = build_internal_dataset(args.split_dir, "test", transform)
    loader = DataLoader(internal, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    labels, probs = predict_batch_probs(model, loader)
    internal_auc = rank_auc(labels, probs[:, 1])
    logger.info("internal test AUC (single-pass): %.4f (manuscript anchor 0.9919)", internal_auc)

    # ---- NIH external test set ----
    nih_test = Path(args.nih_root) / "test_resized" / "test"
    dataset = build_nih_dataset_local(nih_test, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    filenames, labels, probs = predict_with_filenames(model, loader, dataset)
    logger.info("NIH test images scored: %d", len(filenames))

    # ---- join view position from the NIH CSV ----
    csv_path = Path(args.nih_root) / "filtered_dataset_2class.csv"
    view_by_file = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            view_by_file[row["Image Index"]] = row["View Position"].strip().upper()
    views = np.array([view_by_file.get(Path(fn).name, "UNKNOWN") for fn in filenames])
    unmatched = int((views == "UNKNOWN").sum())
    if unmatched:
        raise SystemExit(f"{unmatched} test images not found in the NIH CSV — aborting")
    logger.info("view mix: %s", dict(Counter(views)))

    # ---- stratified metrics ----
    y = labels.astype(int)
    p_pneu = probs[:, 1]
    pa_mask, ap_mask = views == "PA", views == "AP"

    report = {
        "protocol": {
            "weights": str(Path(args.weights).resolve()),
            "inference": "single_pass_raw (T=1.0)",
            "note": ("manuscript NIH headline used mc_mean_scaled (30 passes, T=1.67); "
                     "view stratification uses single-pass probabilities"),
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "internal_anchor": {
            "n": int(len(internal)),
            "auc_single_pass": internal_auc,
            "manuscript_anchor_auc": 0.9919,
            "view": "PA-only (Kermany et al. 2018 release; all frontal PA acquisitions)",
        },
        "nih_overall": None,
        "nih_by_view": {},
        "pa_vs_ap": {},
    }

    report["nih_overall"] = stratum_report(y, p_pneu, "NIH overall (PA+AP)")
    report["nih_by_view"]["PA"] = stratum_report(y[pa_mask], p_pneu[pa_mask], "NIH PA")
    report["nih_by_view"]["AP"] = stratum_report(y[ap_mask], p_pneu[ap_mask], "NIH AP")

    # PA vs AP AUC difference with paired permutation over pooled positive/negative resampling
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    d_obs = report["nih_by_view"]["PA"]["auc"] - report["nih_by_view"]["AP"]["auc"]
    diffs = []
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    for _ in range(args.n_bootstrap):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        ys, ss, vs = y[idx], p_pneu[idx], views[idx]
        d = rank_auc(ys[vs == "PA"], ss[vs == "PA"]) - rank_auc(ys[vs == "AP"], ss[vs == "AP"])
        diffs.append(d)
    diffs = np.asarray(diffs)
    report["pa_vs_ap"] = {
        "delta_auc_pa_minus_ap": float(d_obs),
        "delta_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
    }

    report["view_mix"] = {
        "nih": {k: float(v) for k, v in Counter(views).items()},
        "internal": "PA (100%)",
    }
    report["interpretation_draft"] = (
        "Matched-view residual drop = internal PA anchor minus NIH-PA stratum; "
        "the NIH-PA vs NIH-AP contrast isolates the view effect within the "
        "external cohort. With 49 positives total (24 PA, 25 AP) per-stratum "
        "CIs are wide; treat as descriptive confounder quantification."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %s", out)
    logger.info("summary: internal %.4f | NIH overall %.4f (%.4f-%.4f) | PA %.4f | AP %.4f",
                internal_auc, report["nih_overall"]["auc"], *report["nih_overall"]["auc_ci95"],
                report["nih_by_view"]["PA"]["auc"], report["nih_by_view"]["AP"]["auc"])


def build_nih_dataset_local(root: Path, transform):
    """ImageFolder over ``{NORMAL,PNEUMONIA}`` (same layout as medknow.datasets)."""
    from medknow.datasets.chest_xray import CXRImageFolder

    return CXRImageFolder(root, transform=transform)


def predict_with_filenames(model, loader, dataset):
    """Single-pass inference preserving per-sample filenames (dataset.samples order)."""
    device = next(model.parameters()).device
    filenames = [Path(s[0]).name for s in dataset.samples]
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    return filenames, np.concatenate(all_labels), np.concatenate(all_probs)


if __name__ == "__main__":
    main()
