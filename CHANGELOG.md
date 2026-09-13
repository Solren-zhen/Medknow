# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- AP/PA view-position confounder analysis for the NIH external cohort
  (`scripts/medknow_view_confounder.py`, results in
  `results/tables/view_confounder_nih.{json,md}`): the external discrimination
  collapse is **not** explained by view mix (NIH-PA 0.619 vs NIH-AP 0.623;
  matched-view internal-PA → NIH-PA still drops 0.992 → 0.619), while the
  low-confidence referral signal fails in a **view-structured** way
  (error-prediction AUC 0.746 in PA vs 0.236 in AP, pooled ≈ 0.49).
- README (EN/ZH): clinical-framing section for the intended audience;
  PyPI badge corrected to "planned" (package not yet published).

### Fixed
- README PyPI badge pointed to a nonexistent package (pypi.org/project/medknow → 404).

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
