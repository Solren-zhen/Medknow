"""Patient-level splitting (fixes image-level data leakage).

The internal Kermany dataset ships with an image-level train/val/test split
that allows the same patient to appear in multiple partitions. This module
regroups images by patient ID (``person\\d+`` / ``IM-\\d+`` filename prefixes)
and re-splits by patient at the requested ratios, so a patient never crosses
partitions. The original data is kept read-only; a copy is written to the
split directory together with a JSON split report.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

PATIENT_RE = re.compile(r"^(person\d+|IM-\d+)")
IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png")


def patient_id(fname: str) -> str:
    """Extract the patient ID from a filename prefix."""
    m = PATIENT_RE.match(fname)
    if m:
        return m.group(1)
    return Path(fname).stem


def collect(root: Path) -> tuple[dict[str, dict[str, list[str]]], dict[str, str]]:
    """Return ``(patients, split_of)`` from the original train/val/test layout.

    ``patients[pid][class]`` is a list of absolute image paths; ``split_of[path]``
    records which original partition the image came from.
    """
    root = Path(root)
    patients: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    split_of: dict[str, str] = {}
    for split in ("train", "val", "test"):
        base = root / split
        if not base.is_dir():
            continue
        for cls_dir in base.iterdir():
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            for f in cls_dir.iterdir():
                if not f.name.lower().endswith(IMAGE_SUFFIXES):
                    continue
                pid = patient_id(f.name)
                patients[pid][cls].append(str(f))
                split_of[str(f)] = split
    return patients, split_of


def analyze(root: Path) -> dict[str, dict[str, list[str]]]:
    """Print leakage statistics; returns the patient map."""
    root = Path(root)
    patients, split_of = collect(root)
    pid_splits: dict[str, set] = defaultdict(set)
    for path, split in split_of.items():
        pid_splits[patient_id(Path(path).name)].add(split)

    overlap: dict[tuple, list[str]] = defaultdict(list)
    for pid, splits in pid_splits.items():
        overlap[tuple(sorted(splits))].append(pid)

    total_imgs = sum(len(paths) for cm in patients.values() for paths in cm.values())
    print(f"[analysis] images: {total_imgs} | patients: {len(patients)}")
    for key, pids in sorted(overlap.items(), key=lambda x: -len(x[1])):
        sample = ", ".join(pids[:3])
        print(f"  [leakage] across {list(key)}: {len(pids)} patients (e.g. {sample})")

    images_by_cls: dict[str, int] = defaultdict(int)
    patients_by_cls: dict[str, int] = defaultdict(int)
    for pid, cm in patients.items():
        for cls, paths in cm.items():
            images_by_cls[cls] += len(paths)
            patients_by_cls[cls] += 1
    print(f"  images by class: {dict(images_by_cls)}")
    print(f"  patients by class: {dict(patients_by_cls)}")
    return patients


def split_and_copy(
    patients: dict[str, dict[str, list[str]]],
    out_dir: Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """Shuffle patients, assign to train/val/test, and copy images.

    Args:
        patients: Map from ``medknow.datasets.split.collect``.
        out_dir: Destination (must not already exist).
        val_ratio: Validation fraction of patients.
        test_ratio: Test fraction of patients.
        seed: Random seed for the patient shuffle.

    Returns:
        The split report dict (also written to ``out_dir/split_report.json``).
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(
            f"{out_dir} already exists; remove it or choose another directory "
            "to avoid mixing old data."
        )

    pids = list(patients.keys())
    random.Random(seed).shuffle(pids)
    n_val = int(len(pids) * val_ratio)
    n_test = int(len(pids) * test_ratio)
    val_pids = set(pids[:n_val])
    test_pids = set(pids[n_val:n_val + n_test])
    train_pids = set(pids[n_val + n_test:])

    split_map = {"train": train_pids, "val": val_pids, "test": test_pids}
    for split_name, pid_set in split_map.items():
        for pid in pid_set:
            for cls, paths in patients[pid].items():
                dest = out_dir / split_name / cls
                dest.mkdir(parents=True, exist_ok=True)
                for p in paths:
                    target = dest / Path(p).name
                    if target.exists():
                        print(f"  [conflict] target exists, skipped: {target}")
                        continue
                    shutil.copy2(p, target)

    report = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_patients": len(pids),
        "n_patients_train": len(train_pids),
        "n_patients_val": len(val_pids),
        "n_patients_test": len(test_pids),
    }
    for split_name, pid_set in split_map.items():
        imgs: dict[str, int] = defaultdict(int)
        for pid in pid_set:
            for cls, paths in patients[pid].items():
                imgs[cls] += len(paths)
        report[f"images_{split_name}"] = dict(imgs)
        report[f"images_{split_name}_total"] = sum(imgs.values())

    (out_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
