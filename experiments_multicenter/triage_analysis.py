#!/usr/bin/env python3
"""不确定性引导分诊分析：转诊率 × 敏感度 × 误诊率权衡。

输入：evaluate_external.py --mc-samples N 产出的预测 CSV（label, prob, entropy/variance）。
输出：权衡曲线 JSON + PNG、推荐操作点、H2 假设检验（错误样本不确定性是否更高）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--unc-col", default="entropy", choices=["entropy", "variance"])
    p.add_argument("--target-sensitivity", type=float, default=0.95)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.predictions)
    df["label"] = df["label"].astype(int)
    df["prob"] = df["prob"].astype(float)

    if args.unc_col not in df.columns:
        p = df["prob"].clip(1e-12, 1 - 1e-12)
        df["entropy"] = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        df["variance"] = p * (1 - p)
    unc = df[args.unc_col].to_numpy()

    df["error"] = ((df["prob"] >= 0.5).astype(int) != df["label"]).astype(int)

    # 权衡曲线：按不确定性分位阈值扫描
    rows = []
    thresholds = np.unique(np.quantile(unc, np.linspace(0, 1, 101)))
    for tau in thresholds:
        referred = unc > tau
        if referred.sum() == 0 or (~referred).sum() == 0:
            continue
        acc = df.loc[~referred]
        sens = float((acc["label"] == 1).sum() / max((df["label"] == 1).sum(), 1))
        spec = float((acc["label"] == 0).sum() / max((df["label"] == 0).sum(), 1))
        acc_err = float(acc["error"].mean())
        ref_err = float(df.loc[referred, "error"].mean())
        rows.append({
            "threshold": float(tau),
            "referral_rate": float(referred.mean()),
            "accepted_sensitivity": sens,
            "accepted_specificity": spec,
            "accepted_error_rate": acc_err,
            "referred_error_rate": ref_err,
            "n_referred": int(referred.sum()),
        })
    curve = pd.DataFrame(rows)

    # 推荐操作点：自动筛查敏感度 >= target 时的最低转诊率
    op = None
    if len(curve):
        ok = curve[curve["accepted_sensitivity"] >= args.target_sensitivity]
        if len(ok):
            op = ok.loc[ok["referral_rate"].idxmin()].to_dict()

    # H2 检验：错误样本 vs 正确样本的不确定性
    h2 = {}
    if df["error"].nunique() > 1:
        _, u_p = mannwhitneyu(unc[df["error"] == 1], unc[df["error"] == 0], alternative="greater")
        h2["mannwhitney_u_p"] = float(u_p)
        # 上下四分位错误率对比（χ²）
        q_hi = unc >= np.quantile(unc, 0.75)
        q_lo = unc <= np.quantile(unc, 0.25)
        table = np.array([[int(df.loc[q_hi, "error"].sum()), int((~df.loc[q_hi, "error"]).sum())],
                          [int(df.loc[q_lo, "error"].sum()), int((~df.loc[q_lo, "error"]).sum())]])
        _, chi_p, _, _ = chi2_contingency(table)
        h2["chi2_p"] = float(chi_p)
        h2["top_quartile_error_rate"] = float(df.loc[q_hi, "error"].mean())
        h2["bottom_quartile_error_rate"] = float(df.loc[q_lo, "error"].mean())

    out = {
        "uncertainty_column": args.unc_col,
        "n_images": len(df),
        "overall_error_rate": float(df["error"].mean()),
        "operating_point": op,
        "hypothesis_H2": h2,
        "curve": curve.to_dict(orient="records"),
    }

    out_dir = Path(args.output_dir or Path(args.predictions).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.predictions).stem
    (out_dir / f"triage_{tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(curve["referral_rate"] * 100, curve["accepted_sensitivity"] * 100,
                label="accepted sensitivity", marker="o", ms=3)
        ax.plot(curve["referral_rate"] * 100, curve["accepted_specificity"] * 100,
                label="accepted specificity", marker="s", ms=3)
        ax.plot(curve["referral_rate"] * 100, curve["referred_error_rate"] * 100,
                label="referred error rate", marker="^", ms=3)
        if op:
            ax.axvline(op["referral_rate"] * 100, color="red", ls="--", lw=1)
        ax.set_xlabel("Referral rate (%)"); ax.set_ylabel("(%)")
        ax.set_title(f"Uncertainty-guided triage ({args.unc_col})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"triage_{tag}.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - 曲线图失败不阻断
        print(f"[warn] 曲线图失败: {exc}")

    print(json.dumps({"operating_point": op, "hypothesis_H2": h2}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
