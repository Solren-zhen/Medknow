#!/usr/bin/env python3
"""温度缩放校准：在内部验证集（CheXpert-val）预测 CSV 上拟合温度 T。

用法：
  1) 先评估内部验证集：evaluate_external.py --dataset chexpert_val --save 预测 CSV（含 prob/label）
  2) 拟合：python experiments_multicenter/temperature_scaling.py --predictions outputs_multicenter/predictions_CheXpert-val.csv
  3) 外部评估时加 --temperature outputs_multicenter/temperature.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def nll_at_t(t, logits, y):
    p = 1.0 / (1.0 + np.exp(-logits / t))
    eps = 1e-12
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


def ece(p, y, n_bins=10) -> float:
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += m.sum() / len(conf) * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="内部验证预测 CSV（含 prob, label）")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    df = pd.read_csv(args.predictions)
    y = df["label"].to_numpy().astype(int)
    prob = df["prob"].to_numpy().astype(float)
    logits = np.log(np.clip(prob, 1e-7, 1 - 1e-7) / (1 - np.clip(prob, 1e-7, 1 - 1e-7)))

    ece_before = ece(prob, y)
    res = minimize_scalar(lambda t: nll_at_t(t, logits, y), bounds=(0.1, 10.0), method="bounded")
    t_opt = float(res.x)
    p_scaled = 1.0 / (1.0 + np.exp(-logits / t_opt))
    ece_after = ece(p_scaled, y)

    out_path = Path(args.output or (Path(args.predictions).parent / "temperature.json"))
    out_path.write_text(json.dumps({
        "temperature": t_opt,
        "nll_before": float(res.fun),
        "ece_before": ece_before,
        "ece_after": ece_after,
    }, indent=2), encoding="utf-8")
    print(f"T={t_opt:.4f} ECE {ece_before:.4f} -> {ece_after:.4f} saved to {out_path}")


if __name__ == "__main__":
    main()
