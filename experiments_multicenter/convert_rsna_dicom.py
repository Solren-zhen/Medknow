#!/usr/bin/env python3
"""RSNA 肺炎检测 DICOM → PNG 转换 + 索引 CSV。

输入目录结构（3060 上）：
    D:/datasets/rsna/
      stage_1_train_images/*.dcm        （文件名 = patientId）
      stage_1_train_labels.csv          （patientId, x1, y1, x2, y2, Target, class）
输出：
    D:/datasets/rsna/png/*.png
    D:/datasets/rsna/rsna_index.csv     （image_id, patient_id, label, path）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rsna-dir", default="D:/datasets/rsna")
    p.add_argument("--labels-csv", default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def dcm_to_png(dcm_path: Path, png_path: Path, pydicom):
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:  # 多帧取第一帧
        arr = arr[0]
    if getattr(ds, "WindowCenter", None) is not None:
        wc = float(ds.WindowCenter) if np.ndim(ds.WindowCenter) == 0 else float(ds.WindowCenter[0])
        ww = float(ds.WindowWidth) if np.ndim(ds.WindowWidth) == 0 else float(ds.WindowWidth[0])
        lo, hi = wc - ww / 2, wc + ww / 2
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (arr * 255).astype(np.uint8)
    from PIL import Image
    Image.fromarray(img).save(str(png_path))


def main():
    args = parse_args()
    rsna = Path(args.rsna_dir)
    labels_csv = Path(args.labels_csv) if args.labels_csv else rsna / "stage_1_train_labels.csv"
    img_dir = rsna / "stage_1_train_images"
    out_dir = Path(args.out_dir or rsna)
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    import pydicom
    df = pd.read_csv(labels_csv)
    df.columns = [str(c).strip() for c in df.columns]
    # 患者/图片级标签：该图是否有肺炎阴影框（Target=1）
    df["label"] = (df["Target"] == 1).astype(int)

    rows = []
    for _, row in df.iterrows():
        pid = str(row["patientId"])
        src = img_dir / f"{pid}.dcm"
        if not src.exists():
            continue
        dst = png_dir / f"{pid}.png"
        if not dst.exists():
            try:
                dcm_to_png(src, dst, pydicom)
            except Exception as exc:  # noqa: BLE001 - 单张转换失败跳过，不阻断
                print(f"[skip] {src.name}: {exc}")
                continue
        rows.append({"image_id": pid, "patient_id": pid,
                     "label": int(row["label"]), "path": str(dst)})

    idx = pd.DataFrame(rows)
    idx.to_csv(out_dir / "rsna_index.csv", index=False)
    print(f"[done] {len(idx)} images -> {out_dir / 'rsna_index.csv'} "
          f"(positive={int(idx['label'].sum())})")


if __name__ == "__main__":
    main()
