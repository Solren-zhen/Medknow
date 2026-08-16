"""Tests for the MedKnow dataset loaders."""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medknow.datasets.chest_xray import (
    CXRImageFolder,
    RSNAChestXray,
    find_image_dir,
)


def _make_image(path: Path, size=(32, 32), color=(120, 120, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _make_internal(tmp_path) -> Path:
    root = Path(tmp_path)
    for cls in ("NORMAL", "PNEUMONIA"):
        for i in range(3):
            _make_image(root / "train" / cls / f"{cls}_{i}.png")
        for i in range(2):
            _make_image(root / "test" / cls / f"{cls}_{i}.png")
    return root


def test_internal_imagefolder(tmp_path):
    root = _make_internal(tmp_path)
    ds = CXRImageFolder(root / "train")
    assert len(ds) == 6
    assert ds.classes == ["NORMAL", "PNEUMONIA"]
    _, label = ds[0]
    assert label in (0, 1)


def test_rsna_loader_labels_and_subgroups(tmp_path):
    img_dir = Path(tmp_path) / "input" / "images"
    img_dir.mkdir(parents=True)
    for pid in ("a", "b", "c"):
        _make_image(img_dir / f"{pid}.jpg")
    csv_path = Path(tmp_path) / "stage_2_detailed_class_info.csv"
    csv_path.write_text(
        "patientId,class\n"
        "a,Lung Opacity\n"
        "b,Normal\n"
        "c,No Lung Opacity / Not Normal\n",
        encoding="utf-8",
    )
    ds = RSNAChestXray(tmp_path)
    assert len(ds) == 3
    assert [ds.labels[i] for i in range(3)] == [1, 0, 0]
    assert ds.subgroups[2] == "No Lung Opacity / Not Normal"
    _, label = ds[0]
    assert label == 1


def test_rsna_missing_label_file_raises(tmp_path):
    img_dir = Path(tmp_path) / "input" / "images"
    img_dir.mkdir(parents=True)
    _make_image(img_dir / "a.jpg")
    with pytest.raises(FileNotFoundError):
        RSNAChestXray(tmp_path)


def test_find_image_dir(tmp_path):
    d = Path(tmp_path) / "input" / "images"
    d.mkdir(parents=True)
    _make_image(d / "x.jpg")
    assert find_image_dir(Path(tmp_path)) == d


def test_rsna_dedup_max_label_and_missing_images(tmp_path):
    img_dir = Path(tmp_path) / "input" / "images"
    img_dir.mkdir(parents=True)
    _make_image(img_dir / "a.jpg")
    _make_image(img_dir / "b.jpg")
    # "a" has two rows (max label wins); "c" has no image and must be skipped.
    csv_path = Path(tmp_path) / "stage_2_detailed_class_info.csv"
    csv_path.write_text(
        "patientId,class\n"
        "a,Normal\n"
        "a,Lung Opacity\n"
        "b,Normal\n"
        "c,Lung Opacity\n",
        encoding="utf-8",
    )
    ds = RSNAChestXray(tmp_path)
    assert len(ds) == 2
    assert ds.patient_ids == ["a", "b"]
    assert ds.labels == [1, 0]
    assert ds.subgroups[0] == "Lung Opacity"
