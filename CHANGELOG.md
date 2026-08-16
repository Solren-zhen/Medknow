# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- (your contribution here)

## [1.0.0] - 2026-08-14

### Added
- Reproducible research pipeline: patient-level split → train (3 seeds) →
  uncertainty → calibration → referral → external validation → figures.
- ResNet-18 pneumonia classifier (manuscript architecture) with MC Dropout /
  MSP / entropy / random / ensemble uncertainty estimators.
- External validation on RSNA and NIH ChestXray-14 with per-cohort protocol
  records (`configs/` `evaluation.protocols`).
- Referral / selective-prediction analysis with matched random controls,
  error-prediction AUC and risk-coverage curves.
- Temperature scaling calibration analysis (T≈1.67).
- 74 passing unit tests + ruff lint + GitHub Actions CI.
- Full manuscript (EN + ZH), figures, and research notes in `paper/`.
- Bilingual README with verified results and benchmark leaderboard.
- Hugging Face Space package (`hf_space/`) and one-click Colab demo
  (`notebooks/`).
- Contribution guide, issue/PR templates, code of conduct, security policy,
  Dependabot, and PyPI publishing workflow.

### Verified
- All manuscript numbers regenerated and cross-checked
  (`results/tables/manuscript_verification.md`).
- Internal AUC 0.992 · RSNA AUC 0.807 · NIH AUC 0.658;
  ECE 0.034 (internal) → 0.369 (RSNA) under domain shift.
