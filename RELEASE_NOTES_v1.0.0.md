# MedKnow v1.0.0 Release Notes

**Teaching Medical AI When Not to Know** — uncertainty-aware referral & selective
prediction benchmark for chest X-ray classification, with honest in-domain vs
domain-shift results.

## What's in this release

- Reproducible pipeline: patient-level split → train (3 seeds) → uncertainty
  (MC Dropout / MSP / entropy / random / ensemble / conformal) → calibration →
  referral → external validation (RSNA, NIH ChestXray-14)
- 74 passing tests + CI + lint-clean codebase
- Hugging Face Space package (`hf_space/`) and one-click Colab demo
- Bilingual manuscript and 24 reproducibility result JSONs under `paper/output/repro/`

## Model weights

ResNet-18 pneumonia classifier (manuscript architecture: ImageNet-pretrained,
frozen backbone + layer4 + dropout head, p=0.3). Trained on Kermany chest X-ray
(5,856 images, patient-level 70/15/15 split, seed 42 primary).

| File | Purpose | SHA-256 |
|---|---|---|
| `seed_42.pth` | Primary analysis checkpoint | `89704561D849E8C9CEFB0203FE841625E6ACF7127A460CCFF940F8C2CFC42518` |
| `seed_2024.pth` | Robustness seed | `F8F46C5506AF392CCA05F0F261AA33E05CD9B135F8580063762622068D16C6DB` |
| `seed_2026.pth` | Robustness seed | `CF6CFC6AE5693B83B9931E2F91346C0596F189D75D1FA8FE62A1352584034395` |
| `temperature.txt` | Temperature scaling (T≈1.67) | `0B7A0F9EA0F5429C7912D0702528A49595CC03328EEC2458B1ACA3C44F28BE4C` |

## Reproduce the headline numbers

```bash
pip install -e ".[dev]"
# internal evaluation (AUC 0.992, ECE 0.034)
python scripts/medknow_evaluate.py --weights checkpoints/seed_42.pth
# calibration (T≈1.67)
python scripts/medknow_run_calibration.py --weights checkpoints/seed_42.pth
# referral / selective prediction
python scripts/medknow_run_referral.py --weights checkpoints/seed_42.pth
# external validation (RSNA / NIH)
python scripts/medknow_evaluate_external.py --weights checkpoints/seed_42.pth
```

## Honest positioning

This is a benchmark/education project. The core finding — uncertainty-driven
referral works in-domain but fails under domain shift — is established in the
machine-learning literature (Ovadia et al., NeurIPS 2019). This repository's
contribution is a *controlled, reproducible measurement* (matched-rate random
referral controls on two external cohorts) and an evaluation tool the community
can reuse, including a conformal-prediction method whose coverage guarantee is
shown to break under domain shift.

## Citation

```bibtex
@software{medknow2026,
  author = {MedKnow authors},
  title  = {MedKnow: Teaching Medical AI When Not to Know},
  year   = {2026},
  url    = {https://github.com/Solren-zhen/Medknow}
}
```
