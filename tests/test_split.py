"""Tests for patient-level splitting."""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medknow.datasets.split import analyze, patient_id, split_and_copy


def _make(root: Path, split: str, cls: str, name: str) -> None:
    d = root / split / cls
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (10, 20, 30)).save(d / name)


def test_patient_id_prefixes():
    assert patient_id("person100_virus_46.jpeg") == "person100"
    assert patient_id("IM-0119-0001.jpeg") == "IM-0119"
    assert patient_id("other.png") == "other"


def test_split_patients_never_cross_partitions(tmp_path):
    root = tmp_path / "chest"
    # patient person1 has 3 images, person2 has 2, person3 has 1
    _make(root, "train", "NORMAL", "person1_a.jpeg")
    _make(root, "test", "PNEUMONIA", "person1_b.jpeg")
    _make(root, "val", "NORMAL", "person1_c.jpeg")
    _make(root, "train", "PNEUMONIA", "person2_a.jpeg")
    _make(root, "train", "PNEUMONIA", "person2_b.jpeg")
    _make(root, "train", "NORMAL", "person3_a.jpeg")

    patients = analyze(root)
    assert len(patients) == 3

    out = tmp_path / "split"
    report = split_and_copy(patients, out, val_ratio=0.34, test_ratio=0.33, seed=42)
    assert report["n_patients"] == 3

    # every patient appears in exactly one partition directory
    for split in ("train", "val", "test"):
        files = list((out / split).rglob("*"))
        pids = {patient_id(f.name) for f in files if f.is_file()}
        for pid in pids:
            for other in ("train", "val", "test"):
                if other == split:
                    continue
                assert not list((out / other).rglob(f"{pid}_*"))
    assert (out / "split_report.json").exists()


def test_split_refuses_existing_output(tmp_path):
    root = tmp_path / "chest"
    _make(root, "train", "NORMAL", "person1_a.jpeg")
    patients, _ = analyze(root), None
    out = tmp_path / "split"
    out.mkdir()
    with pytest.raises(FileExistsError):
        split_and_copy(patients, out, seed=42)
