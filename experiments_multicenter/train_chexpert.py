#!/usr/bin/env python3
"""在 CheXpert 上训练肺炎二分类模型（患者级划分，早停，AMP）。

用法（在 3060 上，项目根目录运行）：
    python experiments_multicenter/train_chexpert.py
    python experiments_multicenter/train_chexpert.py --model resnet18 --epochs 25
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments_multicenter.datasets import (
    CheXpertDataset,
    get_transform,
)
from models.model_factory import create_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(PROJECT_ROOT / "experiments_multicenter/config_multicenter.yaml"))
    p.add_argument("--model", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--workers", type=int, default=None)
    return p.parse_args()


def load_config(args) -> dict:
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["data"]["train_batch_size"] = args.batch_size
    if args.val_fraction:
        cfg["data"]["val_fraction"] = args.val_fraction
    if args.seed:
        cfg["training"]["seed"] = args.seed
    if args.workers:
        cfg["data"]["num_workers"] = args.workers
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    cfg = load_config(args)
    out_dir = Path(args.output_dir or cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    d = cfg["data"]
    t = cfg["training"]
    m = cfg["model"]
    set_seed(t["seed"])
    device = pick_device()
    print(f"[train] device={device} model={m['name']} epochs={t['epochs']}")

    chexpert_dir = Path(cfg["paths"]["chexpert_dir"])
    train_csv = chexpert_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"未找到 {train_csv}，请先在 3060 上放置 CheXpert 数据")

    # ---- 患者级划分 ----
    full = pd.read_csv(train_csv)
    full.columns = [str(c).strip() for c in full.columns]
    patient_col = "Patient" if "Patient" in full.columns else "Path"
    groups = full[patient_col].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=d["val_fraction"], random_state=t["seed"])
    train_idx, val_idx = next(gss.split(full, groups=groups))
    val_patients = set(groups.iloc[val_idx])
    print(f"[split] train={len(train_idx)} val={len(val_idx)} val_patients={len(val_patients)}")

    pd.DataFrame({"patient_id": sorted(val_patients)}).to_csv(
        out_dir / "chexpert_val_patients.csv", index=False)

    train_ds = CheXpertDataset(
        train_csv, chexpert_dir, transform=get_transform(d["image_size"], train=True),
        uncertain_policy=d["chexpert_uncertain_policy"], views=d["chexpert_views"],
        label_cols=d["chexpert_label_cols"], patient_ids=set(groups.iloc[train_idx]))
    val_ds = CheXpertDataset(
        train_csv, chexpert_dir, transform=get_transform(d["image_size"], train=False),
        uncertain_policy=d["chexpert_uncertain_policy"], views=d["chexpert_views"],
        label_cols=d["chexpert_label_cols"], patient_ids=val_patients)
    print(f"[data] train_ds={len(train_ds)} val_ds={len(val_ds)}")

    # 类别权重（样本不平衡）
    labels = np.array([int(train_ds[i][1]) for i in range(len(train_ds))])
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    weights = torch.tensor([1.0 / n_neg, 1.0 / n_pos], dtype=torch.float)
    weights = weights / weights.sum() * 2.0
    sample_weights = torch.tensor([weights[int(l)] for l in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=d["train_batch_size"], sampler=sampler,
                              num_workers=d["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=d["eval_batch_size"], shuffle=False,
                            num_workers=d["num_workers"], pin_memory=True)

    model = create_model(name=m["name"], num_classes=2, pretrained=m["pretrained"],
                         freeze_backbone=m["freeze_backbone"], dropout_rate=m["dropout_rate"])
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=t["use_amp"] and device.type == "cuda")

    best_auc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        running = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=t["use_amp"] and device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            n_batches += 1
        train_loss = running / max(n_batches, 1)

        # 验证
        model.eval()
        val_loss, probs, ys = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item() * x.size(0)
                probs.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                ys.append(y.cpu().numpy())
        probs = np.concatenate(probs)
        ys = np.concatenate(ys)
        val_loss /= len(ys)
        val_auc = float(roc_auc_score(ys, probs)) if len(np.unique(ys)) > 1 else 0.5
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        print(f"[epoch {epoch}/{t['epochs']}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(val_auc)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= t["early_stopping_patience"]:
                print(f"[early stop] epoch {epoch}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save({
        "state_dict": best_state,
        "config": cfg,
        "best_val_auc": best_auc,
        "history": history,
    }, out_dir / f"{m['name']}_chexpert_best.pth")

    with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # 训练曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(history["train_loss"], label="train loss")
        ax[0].plot(history["val_loss"], label="val loss")
        ax[0].set_title("Loss"); ax[0].legend()
        ax[1].plot(history["val_auc"], label="val AUC")
        ax[1].set_title("Val AUC"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "training_curves.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - 曲线画不出不阻断训练
        print(f"[warn] 曲线保存失败: {exc}")

    print(f"[done] best_val_auc={best_auc:.4f} saved to {out_dir}")


if __name__ == "__main__":
    main()
