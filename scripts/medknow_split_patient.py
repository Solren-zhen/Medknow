#!/usr/bin/env python3
"""MedKnow — patient-level split (fixes image-level data leakage).

Regroups the internal dataset by patient ID and re-splits at patient level so
no patient crosses train/val/test. The original data is kept read-only.

Usage:
    python scripts/medknow_split_patient.py --seed 42
    python scripts/medknow_split_patient.py --analyze-only
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medknow.config import get, load_config
from medknow.datasets.split import analyze, split_and_copy

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Patient-level split")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data_root", default=None, help="default: config data.internal.root")
    ap.add_argument("--out_dir", default=None, help="default: config paths.split_dir")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    data_root = Path(args.data_root or get(cfg, "data.internal.root"))
    out_dir = Path(args.out_dir or get(cfg, "paths.split_dir"))

    logger.info("Data root: %s", data_root)
    patients = analyze(data_root)
    if args.analyze_only:
        return
    report = split_and_copy(
        patients, out_dir,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
    )
    print(f"[split] report: {out_dir / 'split_report.json'}")
    print(f"  train: {report['images_train']} ({report['images_train_total']})")
    print(f"  val:   {report['images_val']} ({report['images_val_total']})")
    print(f"  test:  {report['images_test']} ({report['images_test_total']})")


if __name__ == "__main__":
    main()
