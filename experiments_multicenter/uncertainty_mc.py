#!/usr/bin/env python3
"""MC Dropout 不确定性量化与风险-转诊分析辅助函数。"""

from __future__ import annotations

import numpy as np
import torch


def enable_dropout(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d)):
            m.train()


def mc_predict(model, loader, n_samples: int = 30, device=None):
    """MC Dropout 推理：返回均值概率、方差、熵、标签、患者 ID。"""
    model.eval()
    enable_dropout(model)
    all_probs, all_var, all_ent, all_labels, all_patients = [], [], [], [], []
    ds = loader.dataset
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            samples = []
            for _ in range(n_samples):
                samples.append(torch.softmax(model(x), dim=1)[:, 1].cpu().numpy())
            S = np.stack(samples, axis=0)  # (n_samples, batch)
            mean_p = S.mean(axis=0)
            var = S.var(axis=0)
            ent = -mean_p * np.log(np.clip(mean_p, 1e-12, 1)) - (1 - mean_p) * np.log(np.clip(1 - mean_p, 1e-12, 1))
            all_probs.append(mean_p)
            all_var.append(var)
            all_ent.append(ent)
            all_labels.append(y.numpy())
            all_patients.extend(np.array(ds.patient_ids)[y.cpu().numpy().astype(int)].tolist())
    return (np.concatenate(all_probs), np.concatenate(all_var), np.concatenate(all_ent),
            np.concatenate(all_labels), all_patients)
