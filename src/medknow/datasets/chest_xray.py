"""Chest X-ray dataset loaders for the three cohorts used in MedKnow.

- Internal: Kermany et al. Chest X-Ray Images, patient-level split
  (ImageFolder-compatible class folders under ``data/split_patient/``).
- External RSNA: images ``{patient_id}.jpg`` plus image-level labels/subgroups
  from CSV.
- External NIH: two-class subset (NORMAL / PNEUMONIA) in class folders.

No medical image is distributed with this repository; the loaders only point at
locally downloaded data.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMG_EXTS = {".jpg", ".jpeg", ".png"}


class CXRImageFolder(ImageFolder):
    """Internal Kermany / NIH two-class folders (NORMAL, PNEUMONIA)."""

    def __init__(self, root: str | Path, transform: Callable | None = None):
        super().__init__(str(root), transform=transform)


class NIHChestXray(CXRImageFolder):
    """NIH ChestXray-14 two-class subset (NORMAL / PNEUMONIA folders)."""


def find_image_dir(root: Path) -> Path | None:
    """Locate the directory containing RSNA image files."""
    for cand in (
        root / "input" / "images",
        root / "images",
        root,
    ):
        if cand.is_dir():
            try:
                if any(f.suffix.lower() in IMG_EXTS for f in cand.iterdir()):
                    return cand
            except OSError:
                continue
    return None


class RSNAChestXray(Dataset):
    """RSNA Pneumonia Detection Challenge cohort.

    Images are ``{patient_id}.jpg`` files; labels come from
    ``stage_2_detailed_class_info.csv`` (Lung Opacity = positive) or
    ``stage_2_train_labels.csv`` (Target column). The detailed CSV provides the
    subgroup labels used in the manuscript subgroup analysis (Lung Opacity /
    No Lung Opacity / Not Normal / Normal).

    Two RSNA-specific details are handled here, matching the manuscript:

    1. The CSV contains rows for images without downloaded files (the local
       ``input/images`` directory holds the 26,684-image train subset); only
       patient IDs with an existing image are kept.
    2. A patient ID may have several annotation rows; the maximal label is
       kept (any positive annotation makes the image positive).

    The dataset returns ``(image, label)`` and exposes parallel ``subgroups``
    and ``patient_ids`` lists for subgroup analysis.
    """

    POSITIVE_CLASS = "Lung Opacity"

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        label_csv: str | Path | None = None,
        class_csv: str | Path | None = None,
        image_dir: str | Path | None = None,
    ):
        root = Path(root)
        self.root = root
        self.transform = transform

        if class_csv is None:
            class_csv = root / "stage_2_detailed_class_info.csv"
        if label_csv is None:
            label_csv = root / "input" / "stage_2_train_labels.csv"
        class_csv = Path(class_csv)
        label_csv = Path(label_csv)

        img_dir = Path(image_dir) if image_dir else find_image_dir(root)
        if img_dir is None or not img_dir.is_dir():
            raise FileNotFoundError(f"No RSNA image directory found under {root}")

        rows: list[tuple[str, int, str]] = []
        if class_csv.exists():
            rows = self._read_detailed(class_csv)
        elif label_csv.exists():
            rows = self._read_targets(label_csv)
        else:
            raise FileNotFoundError(
                "No RSNA label file found: expected "
                f"{class_csv.name} or {label_csv.name}"
            )

        # Deduplicate by patient ID keeping the maximal label.
        by_pid: dict[str, tuple[int, str]] = {}
        for pid, label, subgroup in rows:
            prev = by_pid.get(pid)
            if prev is None or label > prev[0]:
                by_pid[pid] = (label, subgroup)

        self.patient_ids: list[str] = []
        self.labels: list[int] = []
        self.subgroups: list[str] = []
        self.image_paths: list[Path] = []
        for pid, (label, subgroup) in by_pid.items():
            img_path = img_dir / f"{pid}.jpg"
            if not img_path.exists():
                continue
            self.patient_ids.append(pid)
            self.labels.append(label)
            self.subgroups.append(subgroup)
            self.image_paths.append(img_path)

    @staticmethod
    def _read_detailed(path: Path) -> list[tuple[str, int, str]]:
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = row["patientId"]
                cls = row["class"]
                rows.append((pid, 1 if cls == "Lung Opacity" else 0, cls))
        return rows

    @staticmethod
    def _read_targets(path: Path) -> list[tuple[str, int, str]]:
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append((row["patientId"], int(row["Target"]), "Unknown"))
        return rows

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def base_transform(image_size: int = 224) -> transforms.Compose:
    """Inference transform: resize to ``image_size`` + ImageNet normalization."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def train_transform(
    image_size: int = 224,
    rotation_deg: float = 10.0,
    brightness: float = 0.2,
    contrast: float = 0.1,
    affine_translation_frac: float = 0.05,
) -> transforms.Compose:
    """Training augmentation, mirroring the manuscript (rotation / color /
    affine translation; no horizontal flip is used)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(rotation_deg),
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
        transforms.RandomAffine(
            degrees=0,
            translate=(affine_translation_frac, affine_translation_frac),
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_internal_dataset(
    split_dir: str | Path,
    split: str,
    transform: Callable | None = None,
) -> CXRImageFolder:
    """Internal patient-level split folder: ``split_dir/{train,val,test}/{class}``."""
    return CXRImageFolder(Path(split_dir) / split, transform=transform)


def build_rsna_dataset(
    root: str | Path,
    transform: Callable | None = None,
) -> RSNAChestXray:
    return RSNAChestXray(root, transform=transform)


def build_nih_dataset(
    root: str | Path,
    transform: Callable | None = None,
) -> CXRImageFolder:
    """NIH two-class subset: ``root/{NORMAL,PNEUMONIA}`` folders."""
    return CXRImageFolder(root, transform=transform)


def make_loader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
