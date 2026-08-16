# Dataset & Evaluation Protocol Notes

This file records dataset quirks and evaluation-protocol choices that affect
reproducibility. These details are the difference between "I got the same
numbers" and "I understand why someone else might not".

## 1. RSNA label file: 30,227 rows vs 26,684 images

`stage_2_detailed_class_info.csv` contains **30,227 rows**, but the local image
directory (`input/images/`) holds the **26,684-image** train subset used in the
manuscript. The extra 3,543 rows are patient IDs **without** downloaded images
and are all `Lung Opacity` — this is why a naive read of the CSV yields 9,555
positives instead of the manuscript's 6,012.

Resolution (implemented in `RSNAChestXray`):

1. Keep only patient IDs with an existing image file;
2. Deduplicate by patient ID keeping the **maximal** label (any positive
   annotation makes the image positive).

This yields exactly the manuscript's subgroup distribution:
**Lung Opacity 6,012 · No Lung Opacity / Not Normal 11,821 · Normal 8,851**
(26,684 total).

## 2. Two evaluation protocols coexist in the manuscript

The manuscript (v0.6) used **different protocols per cohort**, and MedKnow
records them explicitly in `configs/` (`evaluation.protocols`) rather than
silently "unifying" them:

| Cohort | Inference | ECE definition |
|---|---|---|
| internal_test | single forward, raw probabilities | label-rate (bins by `p_pos`) |
| rsna | single forward, raw probabilities | label-rate |
| nih | MC Dropout mean (30 passes, T = 1.67) | confidence (bins by `max(p, 1−p)` vs argmax accuracy) |

Both ECE variants are implemented (`compute_ece` and
`compute_ece_confidence` in `medknow/calibration/metrics.py`).

## 3. Documented deviations (all non-bugs)

- **RSNA AUPRC**: 0.5085 vs manuscript 0.514 — scikit-learn version
  sensitivity in precision-recall interpolation (AUC/ECE/Brier match exactly).
- **RSNA subgroup FP off by one** (8,281 vs 8,280): one image sits exactly on
  the 0.5 boundary; PIL-version interpolation differences can flip it.
- **NIH sensitivity run-to-run spread** (61.2%–63.3%): MC Dropout is unseeded
  (as in the manuscript); 49 positive cases → ±1 case.
- **Fitted temperature** 1.569 vs manuscript 1.67: different optimizers
  (scipy bounded vs LBFGS); the raw→scaled ECE change is reproduced.

## 4. Environment

Numbers were reproduced with Python 3.11, PyTorch 2.5.1 + CUDA 12.1, on a
single RTX 3060 6 GB. Deep-learning runs have minor run-to-run variation across
hardware/software; allow ±0.001-level differences in AUC when re-running.

## 5. What is NOT in this repository

- No chest X-ray images, DICOM files, or patient information (see
  `data/README.md` for sources and licenses).
- No model weights (`*.pth`); trained checkpoints are regenerable via
  `scripts/medknow_train.py` (or obtainable from the authors).
