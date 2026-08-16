#!/usr/bin/env python3
"""多中心胸片数据集加载器：CheXpert / RSNA / VinDr-CXR。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(image_size: int = 224, train: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_png(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class CheXpertDataset(Dataset):
    """CheXpert 肺炎二分类（1=阳性, -1=阴性, 0=不确定）。

    uncertain_policy:
      ignore      — 剔除不确定样本（默认，最干净）
      as_negative — 0 视为阴性
      as_positive — 0 视为阳性
    """

    def __init__(
        self,
        csv_path: str | Path,
        img_root: str | Path,
        transform=None,
        uncertain_policy: str = "ignore",
        views: Iterable[str] = ("Frontal",),
        label_cols: Iterable[str] = ("Pneumonia",),
        patient_level: bool = False,
        patient_ids: set | None = None,
    ):
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        label_cols = list(label_cols)

        if views:
            df = df[df["Frontal/Lateral"].isin(list(views))]
        df = df.dropna(subset=["Path"])
        for c in label_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if uncertain_policy == "ignore":
            df = df[df[label_cols].fillna(-9).ne(0).all(axis=1)]
        elif uncertain_policy == "as_negative":
            df[label_cols] = df[label_cols].replace(0, -1)
        elif uncertain_policy == "as_positive":
            df[label_cols] = df[label_cols].replace(0, 1)
        else:
            raise ValueError(f"未知 uncertain_policy: {uncertain_policy}")
        df = df.dropna(subset=label_cols)

        # 多标签列：任一阳性即阳性
        df["_label"] = (df[label_cols] == 1).any(axis=1).astype(int)

        # 患者级
        if "Patient" in df.columns:
            df["_patient"] = df["Patient"].astype(str)
        else:
            df["_patient"] = df["Path"].apply(lambda p: str(Path(p).parent))
        if patient_ids is not None:
            df = df[df["_patient"].isin(patient_ids)]
        if patient_level:
            df = df.drop_duplicates(subset="_patient", keep="first")

        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = _load_png(str(self.img_root / row["Path"]))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(int(row["_label"]), dtype=torch.long)

    @property
    def patient_ids(self) -> list[str]:
        return list(self.df["_patient"])


class RSNADataset(Dataset):
    """RSNA 肺炎检测（预处理后）：读取 rsna_index.csv（含 path, label）。

    索引列：image_id, patient_id, label(0/1), path
    """

    def __init__(self, index_csv: str | Path, transform=None,
                 patient_level: bool = False, patient_ids: set | None = None):
        df = pd.read_csv(index_csv)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(subset=["path", "label"])
        df["label"] = df["label"].astype(int)
        df["_patient"] = df["patient_id"].astype(str)
        if patient_ids is not None:
            df = df[df["_patient"].isin(patient_ids)]
        if patient_level:
            df = df.drop_duplicates(subset="_patient", keep="first")
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = _load_png(row["path"])
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(int(row["label"]), dtype=torch.long)

    @property
    def patient_ids(self) -> list[str]:
        return list(self.df["_patient"])


class VinDrCXRSet(Dataset):
    """VinDr-CXR 肺炎二分类。

    train.csv 列：image_id, rad_id, patient_id, sex, age, view_position, 28 个发现(0/1/2)
    pneumonia==1 → 阳性；全部发现==0 → 阴性；pneumonia==2（不确定）→ 剔除。
    """

    PNEUMONIA_COL = "Pneumonia"
    FINDING_COLS: ClassVar[list[str]] = [
        "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
        "Clavicle fracture", "Consolidation", "Edema", "Emphysema",
        "Enlarged PA", "Fibrosis", "Fracture", "Hernia", "Infiltration",
        "Lung cavity", "Lung cyst", "Lung opacity", "Mediastinal shift",
        "Nodule/Mass", "Pleural effusion", "Pleural thickening", "Pneumonia",
        "Pneumothorax", "Pulmonary fibrosis", "Rib fracture", "Scoliosis",
        "Other lesion", "COPD", "Bronchiectasis", "Tuberculosis",
    ]

    def __init__(self, csv_path: str | Path, img_root: str | Path, transform=None,
                 patient_level: bool = False, patient_ids: set | None = None):
        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        if self.PNEUMONIA_COL not in df.columns:
            raise KeyError(f"VinDr CSV 缺少 {self.PNEUMONIA_COL} 列")

        df["_pneumonia"] = pd.to_numeric(df[self.PNEUMONIA_COL], errors="coerce")
        # 剔除不确定（2）与缺失
        df = df[df["_pneumonia"].isin([0, 1])]
        # 阴性：全部发现为 0（无异常）
        cols = [c for c in self.FINDING_COLS if c in df.columns]
        if cols:
            only_normal = (df[cols] == 0).all(axis=1)
        else:
            only_normal = pd.Series(True, index=df.index)
        df["_label"] = df["_pneumonia"].astype(int)
        df.loc[only_normal, "_label"] = 0

        if "patient_id" in df.columns:
            df["_patient"] = df["patient_id"].astype(str)
        else:
            df["_patient"] = df["image_id"].astype(str)
        if patient_ids is not None:
            df = df[df["_patient"].isin(patient_ids)]
        if patient_level:
            df = df.drop_duplicates(subset="_patient", keep="first")

        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = _load_png(str(self.img_root / f"{row['image_id']}.png"))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(int(row["_label"]), dtype=torch.long)

    @property
    def patient_ids(self) -> list[str]:
        return list(self.df["_patient"])
