#!/usr/bin/env python3
"""外部/内部评估：RSNA、VinDr-CXR、CheXpert 验证集。

输出：指标 JSON（AUC 95%CI、敏感度@95%特异度、ECE、Brier）+ ROC 图 + 预测 CSV。
若 --mc-samples > 0，同时输出不确定性列，供 triage_analysis.py 使用。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments_multicenter.datasets import (
    CheXpertDataset,
    RSNADataset,
    VinDrCXRSet,
    get_transform,
)
from experiments_multicenter.uncertainty_mc import mc_predict
from models.model_factory import create_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "experiments_multicenter/config_multicenter.yaml"))
    p.add_argument("--checkpoint", required=True, help="train_chexpert.py 产出的 .pth")
    p.add_argument("--dataset", required=True, choices=["rsna", "vindr", "chexpert_val"])
    p.add_argument("--mc-samples", type=int, default=0, help=">0 启用 MC Dropout")
    p.add_argument("--temperature", default=None, help="temperature_scaling.py 产出的 temp.json")
    p.add_argument("--patient-level", action="store_true", help="外部测试按患者去重（默认开）")
    p.add_argument("--index-csv", default=None, help="RSNA 索引 CSV 覆盖路径")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def bootstrap_auc_ci(y, p, n_boot=2000, seed=42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        aucs.append(roc_auc_score(y[s], p[s]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def ece_binary(p, y, n_bins=10) -> float:
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


def sens_at_spec(y, p, spec=0.95) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    idx = int(np.argmin(np.abs(fpr - (1 - spec))))
    return float(tpr[idx])


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ckpt.get("config", cfg)
    model_name = model_cfg["model"]["name"]
    model = create_model(name=model_name, num_classes=2, pretrained=False,
                         freeze_backbone=False, dropout_rate=model_cfg["model"]["dropout_rate"])
    model.load_state_dict(ckpt["state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    model.to(device)
    model.eval()

    d = cfg["data"]
    transform = get_transform(d["image_size"], train=False)
    out_dir = Path(args.output_dir or cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "rsna":
        idx_csv = args.index_csv or str(Path(cfg["paths"]["rsna_dir"]) / "rsna_index.csv")
        ds = RSNADataset(idx_csv, transform=transform, patient_level=args.patient_level)
        ds_name = "RSNA"
    elif args.dataset == "vindr":
        vdir = Path(cfg["paths"]["vindr_dir"])
        ds = VinDrCXRSet(vdir / "train.csv", vdir, transform=transform,
                         patient_level=args.patient_level)
        ds_name = "VinDr-CXR"
    else:
        cdir = Path(cfg["paths"]["chexpert_dir"])
        val_patients = set(pd.read_csv(out_dir / "chexpert_val_patients.csv")["patient_id"].astype(str))
        ds = CheXpertDataset(cdir / "train.csv", cdir, transform=transform,
                             uncertain_policy=d["chexpert_uncertain_policy"],
                             views=d["chexpert_views"], label_cols=d["chexpert_label_cols"],
                             patient_ids=val_patients)
        ds_name = "CheXpert-val"

    loader = torch.utils.data.DataLoader(ds, batch_size=d["eval_batch_size"],
                                         shuffle=False, num_workers=d["num_workers"])

    temp = 1.0
    if args.temperature:
        with open(args.temperature, "r", encoding="utf-8") as f:
            temp = float(json.load(f)["temperature"])

    if args.mc_samples > 0:
        mean_p, var, entropy, labels, patient_ids = mc_predict(
            model, loader, n_samples=args.mc_samples, device=device)
        probs = mean_p
    else:
        probs, labels, patient_ids = [], [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                out = model(x) / temp
                probs.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                labels.append(y.numpy())
                patient_ids.extend(np.array(ds.patient_ids)[y.cpu().numpy().astype(int)].tolist())
        probs = np.concatenate(probs)
        labels = np.concatenate(labels)
        var = entropy = None

    # 指标
    auc = float(roc_auc_score(labels, probs))
    ci_lo, ci_hi = bootstrap_auc_ci(labels, probs)
    fpr, tpr, _ = roc_curve(labels, probs)
    youden = int(np.argmax(tpr - fpr))
    sens_youden, spec_youden = float(tpr[youden]), float(1 - fpr[youden])
    sens95 = sens_at_spec(labels, probs, spec=cfg["metrics"]["sensitivity_at_specificity"])
    pred50 = (probs >= 0.5).astype(int)
    acc = float((pred50 == labels).mean())
    tn = ((pred50 == 0) & (labels == 0)).sum(); fp = ((pred50 == 1) & (labels == 0)).sum()
    fn = ((pred50 == 0) & (labels == 1)).sum(); tp = ((pred50 == 1) & (labels == 1)).sum()
    sens50 = float(tp / max(tp + fn, 1)); spec50 = float(tn / max(tn + fp, 1))
    ece = ece_binary(probs, labels)
    brier = float(brier_score_loss(labels, probs))
    mr = float(fn / max(fn + tp, 1))  # 漏诊率 @0.5

    report = {
        "dataset": ds_name,
        "n_patients": len(np.unique(patient_ids)) if patient_ids else None,
        "n_images": len(labels),
        "n_positive": int(labels.sum()),
        "temperature": temp,
        "auc": auc,
        "auc_95ci": [ci_lo, ci_hi],
        "accuracy_0.5": acc,
        "sensitivity_0.5": sens50,
        "specificity_0.5": spec50,
        "sensitivity_youden": sens_youden,
        "specificity_youden": spec_youden,
        f"sensitivity_at_{int(cfg['metrics']['sensitivity_at_specificity'] * 100)}spec": sens95,
        "miss_rate_0.5": mr,
        "ece": ece,
        "brier": brier,
    }
    tag = f"{ds_name.replace('/', '_')}"
    with open(out_dir / f"report_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # ROC 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"AUC={auc:.3f} ({ci_lo:.3f}-{ci_hi:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
        ax.set_title(f"{ds_name} ROC"); ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"roc_{tag}.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - ROC 图失败不阻断
        print(f"[warn] ROC 图失败: {exc}")

    # 预测 CSV（供 triage / 后续分析）
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "label": labels,
        "prob": probs,
    })
    if var is not None:
        df["variance"] = var
        df["entropy"] = entropy
    df.to_csv(out_dir / f"predictions_{tag}.csv", index=False)
    print(f"[save] {out_dir / ('report_' + tag + '.json')}")


if __name__ == "__main__":
    main()
