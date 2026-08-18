<div align="center">

# 🩻 MedKnow

## Teaching Medical AI When Not to Know

**When should medical AI say "I don't know"?** — an open-source evaluation framework
for uncertainty-aware referral and selective prediction in medical imaging, showing
that the signal works **in-domain** but **fails under domain shift**.

[![Live Demo](https://img.shields.io/badge/🖥️-Live%20Demo-FF4B4B)](https://huggingface.co/spaces/ojdanajakir848-a11y/medknow-pneumonia-xray)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Solren-zhen/Medknow/blob/main/notebooks/medknow_colab.ipynb)
[![PyPI](https://img.shields.io/pypi/v/medknow.svg)](https://pypi.org/project/medknow/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-74%20passing-brightgreen)]()
[![Paper](https://img.shields.io/badge/📄-Manuscript-blue)](paper/output/doc/manuscript.md)

**[📄 Manuscript](paper/output/doc/manuscript.md)** · [🇨🇳 中文](README_zh-CN.md) ·
[🖥️ Live Demo](https://huggingface.co/spaces/ojdanajakir848-a11y/medknow-pneumonia-xray) ·
[🚀 Colab](notebooks/medknow_colab.ipynb)

</div>

> ⚠️ **Research / education only.** This is not a medical device and must not be
> used for diagnosis.

---

## 📌 TL;DR

A ResNet-18 pneumonia classifier trained on 5,856 chest X-rays (patient-level
split, three seeds) reaches **internal AUC 0.992**. Routing the 25% most
uncertain cases to a human reader drops retained-set error from 4.0% to **0.3%
with zero missed pneumonias** (seed_42 primary analysis). Deploy the *same model*
unchanged to two external cohorts and everything falls apart:

| | Internal | RSNA (external) | NIH (external) |
|---|---:|---:|---:|
| **AUC** | **0.992** | **0.807** | **0.658** |
| **ECE** (calibration error) | **0.034** | **0.369** | **0.265** |
| **Uncertainty-driven referral** | beats random ✅ | ≈ random ❌ | ≈ random ❌ |

**Core conclusion: a model can be confident and wrong under domain shift.**
Confidence- or uncertainty-based triage in medical AI cannot be trusted on the
strength of internal validation alone — discrimination, calibration, and the
referral signal must be re-validated in every new deployment environment.

---

## 🚀 Quickstart (30 seconds)

| What | Where |
|---|---|
| **Zero-install demo** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Solren-zhen/Medknow/blob/main/notebooks/medknow_colab.ipynb) — load weights, run MC Dropout on sample X-rays, plot the referral curves. Free CPU/GPU runtime. |
| **Live web demo** | [🖥️ Hugging Face Space](https://huggingface.co/spaces/ojdanajakir848-a11y/medknow-pneumonia-xray) — upload a chest X-ray, see prediction + uncertainty + Grad-CAM. |
| **pip install** | `pip install -e ".[dev]"` (or `conda env create -f environment.yml`) |
| **Package** | `pip install medknow` — uncertainty / calibration / referral as a library |

**Reproduce the full paper pipeline** (data → 3 seeds → uncertainty → calibration →
referral → external validation → figures): see [`Reproduce the experiments`](#-reproduce-the-experiments).

## Related work & positioning

This project does **not** claim to have discovered that uncertainty fails under
domain shift — that finding is established in the machine-learning literature
([Ovadia et al., NeurIPS 2019](https://arxiv.org/abs/1906.02530)), and the clinical concern that
high-uncertainty predictions should be referred rather than acted upon was
articulated by [Kompa et al. (npj Digital Medicine, 2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8575697/). Its
contributions are:

1. **A controlled, reproducible measurement** of the in-domain gain and the
   out-of-domain failure of uncertainty-driven referral in chest X-ray
   pneumonia classification, using *matched-rate random-referral controls* on
   two independent external cohorts (RSNA, NIH ChestXray-14) — the control
   that isolates "the signal carries information about correctness" from
   "we just removed some cases".
2. **A negative result with practical force**: temperature scaling fixes
   internal calibration but does not transfer across domains (external ECE
   stays ≈0.36 with the internal temperature), qualifying the standard remedy.
3. **A community benchmark**: public code, 74 passing tests, CI, and every
   reported number reproducible from a single command.

**Closest prior work.** Srivastava et al. (2023, [arXiv:2311.16766](https://arxiv.org/abs/2311.16766)) study selective-classification referral failures under
domain shift and propose *rescuing* strategies. Our results complement theirs:
in a different modality (chest X-ray) and with matched random-referral
controls, we show that post-hoc calibration alone does not rescue the referral
signal. AT-CXR (2025, arXiv:2508.19322) explores uncertainty-aware agentic
triage for chest X-rays; this repository provides the reproducible
evaluation baseline such systems should be measured against.

---
## 1. Project overview

MedKnow is an open-source evaluation framework for uncertainty-aware referral and
selective prediction in medical imaging. The research question is simple: when
should a medical AI say "I don't know"? We trained a ResNet-18 pneumonia
classifier on 5,856 chest X-rays (patient-level split, three seeds) and asked
whether predictive uncertainty can identify unreliable predictions and route
them to human review.

The results show a sharp in-domain / out-of-domain contrast. Under the internal
data distribution, referring the 25% most uncertain cases reduced retained-case
missed pneumonias to zero in the primary seed-42 analysis (≤0.2% across seeds;
internal AUC 0.992). Deployed unchanged to two independent external cohorts,
discrimination dropped (RSNA AUC 0.807; NIH ChestXray-14 AUC 0.658, 95% CI
0.585–0.735 on 49 positive cases), calibration collapsed (ECE 0.034 → 0.369),
and the uncertainty-driven referral strategy became no better than random
referral on either external cohort.

## 2. Research question

> Can predictive uncertainty identify unreliable chest X-ray predictions — and
> does that signal survive domain shift?

This is a *selective prediction / referral* problem, not a "build a better
classifier" problem. The benchmark treats "I don't know" as a first-class,
quantifiable model output.

## 3. Key findings

1. **Referral works in-domain.** MC Dropout uncertainty ranks internal
   predictions by correctness: at a 10% referral rate retained-case error fell
   from 4.0% to 1.4%, and at 25% to 0.3% with zero missed pneumonias in the
   primary analysis (χ² = 9.56, P = 0.002 and χ² = 16.30, P < 0.001 vs random).
2. **Referral fails under domain shift.** On both external cohorts the
   uncertainty signal was no better than random referral (error-prediction AUC
   ≈ 0.5), and calibration collapsed (internal ECE 0.034 vs RSNA 0.369).
3. **The failure is not specific to MC Dropout.** A plain-confidence referral
   signal behaves identically — effective in-domain, uninformative externally.

## 4. Study design

Retrospective evaluation of a binary pneumonia classifier with patient-level
splitting, three training seeds, MC Dropout uncertainty, referral analysis with
matched random controls, temperature scaling, and external validation on two
independent cohorts (STARD 2015 / CLAIM 2020 / TRIPOD).

![Study flow](results/figures/fig01_pipeline.png)

## 5. Dataset

| Split | Dataset | Images | Positive definition | Prevalence |
|---|---|---|---:|---:|
| Internal | Kermany Chest X-Ray Images | 5,856 (test 896) | Pneumonia | 75.6% (test) |
| External #1 | RSNA Pneumonia Detection Challenge | 26,684 | Lung Opacity | 22.5% |
| External #2 | NIH ChestXray-14 (2-class) | 9,103 | Pneumonia | 0.5% |

Patient-level split: 70/15/15, seed 42 → train 4,076 (2,222 patients), val 884
(476), test 896 (476). Dataset quirks and protocol choices: [DATASET_NOTES.md](DATASET_NOTES.md). Access instructions and license notes:
[data/README.md](data/README.md).

## 6. Model

ResNet-18, ImageNet-pretrained, frozen backbone with the final residual block
(layer4) and a dropout head fine-tuned (dropout rate 0.3). Trained with AdamW
(lr 1e-4, weight decay 1e-5), batch 16, early stopping (patience 5), AMP, on a
single RTX 3060 (PyTorch 2.5). Checkpoints: `checkpoints/seed_42.pth`,
`seed_2024.pth`, `seed_2026.pth` — also published as a
[GitHub Release](https://github.com/Solren-zhen/Medknow/releases) and in the
[Hugging Face Space](https://huggingface.co/spaces/ojdanajakir848-a11y/medknow-pneumonia-xray).

## 7. Uncertainty methods

All methods share one interface (`estimate_uncertainty(model, loader, method)`)
and output scores where *higher = more uncertain*:

- `mc_dropout` — 30 stochastic forward passes, std of the pneumonia probability
  (primary estimator);
- `msp` — 1 − max softmax probability (plain confidence);
- `entropy` — predictive entropy;
- `random` — matched-rate control;
- `ensemble` — across-member std (results: internal epAUC 0.938, works; external ≈ random, fails).

## 8. Referral strategy

At each fixed referral rate (0–50%), the most uncertain cases are routed to a
human reader and the error rate, sensitivity, FNR, and missed-case count are
measured on the retained set. Random referral at matched rates isolates whether
the signal carries information about correctness — not merely the effect of
dropping cases. Includes error-prediction AUC and risk-coverage curves.

![Referral curves](results/figures/fig05_referral.png)

## 9. Calibration

ECE (15 bins) and Brier score, raw and temperature-scaled, in-domain and
external. Temperature scaling (T = 1.67, fit on the internal validation set)
fixes in-domain miscalibration but does **not** transfer: external ECE stays
≈0.36 with the internal T. **Temperature scaling is not a domain-adaptation
method.**

![Calibration](results/figures/fig04_calibration.png)

## 10. Domain shift

Internal / RSNA / NIH are evaluated through one interface with per-cohort
protocols recorded in `configs/` (`evaluation.protocols`):

| Cohort | Inference protocol | ECE style |
|---|---|---|
| internal_test | single-pass raw | label-rate |
| rsna | single-pass raw | label-rate |
| nih | MC Dropout mean (30, T=1.67) | confidence |

![Domain-shift summary](results/figures/fig08_domain_shift_summary.png)

## 11. Results

Verification against the manuscript (seed_42; all regenerated by
`scripts/medknow_evaluate_external.py`):

| Metric | Internal | RSNA | NIH |
|---|---:|---:|---:|
| AUC | 0.992 | 0.807 | 0.658 |
| AUPRC | 0.997 | 0.514 | 0.010 |
| Accuracy | 96.1% | 60.7% | 62.3% |
| Sensitivity | 96.4% | 90.7% | 61.2% |
| Specificity | 95.0% | 51.9% | 62.3% |
| ECE (raw) | 0.034 | 0.369 | 0.265 |
| Brier (raw) | 0.032 | 0.355 | 0.315 |

Referral (internal): 0% → 4.0% error; 10% → 1.4%; 25% → 0.3%, zero missed
pneumonias (primary seed-42 analysis; ≤0.2% across seeds). External: referral
curves overlap random referral. Subgroup analysis (RSNA): No Lung Opacity /
Not Normal images are classified as pneumonia in 70.0% of cases and contribute
83.3% of false positives — the main driver of low external specificity.

![ROC](results/figures/fig02_roc.png) ![Subgroups](results/figures/fig07_subgroup.png)

Referral-method comparison (retained error @ 25% referral): internally, all three
signals beat random (random 3.9% → MSP 0.3% / MC 0.15% / Ensemble 0.45%), MC
slightly ahead; externally, none beat random (RSNA: random 39.4% → MSP 37.7% /
MC 37.6% / Ensemble 41.1%; NIH similar). Complex uncertainty estimators provide
no advantage over plain confidence, and no confidence-based signal survives
domain shift (see [referral_methods_summary.json](results/tables/referral_methods_summary.json)).

**Verification status:** all manuscript numbers are regenerated and verified by
the new pipeline (see [results/tables/manuscript_verification.md](results/tables/manuscript_verification.md)).
External MC Dropout referral and Deep Ensemble results are computed from the new
pipeline ([protocol sensitivity](results/tables/protocol_sensitivity.json),
[ensemble summary](results/tables/ensemble_summary.json)); an evaluation-protocol
sensitivity analysis shows AUC is stable across protocols (internal 0.992 /
RSNA 0.807 / NIH 0.659).

## 12. Leaderboard — submit your method

MedKnow doubles as a **benchmark for uncertainty-aware referral / selective
prediction under domain shift**. Every published result is reproducible from one
command. To add your method, open a PR that updates the table and includes your
`results/metrics/predictions/*.npz`.

| Method | Internal AUC | Internal ep-AUC | Internal ECE | RSNA AUC | RSNA ECE | NIH AUC | NIH ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-18 + **MC Dropout** (30, T=1.67) | 0.992 | 0.932 | 0.034 | 0.807 | 0.369 | 0.658 | 0.265 |
| ResNet-18 + **MSP** (plain confidence) | 0.992 | 0.931 | 0.034 | 0.807 | 0.369 | 0.658 | 0.265 |
| ResNet-18 + **Deep Ensemble** (3 seeds) | 0.993 | 0.938 | 0.024 | — | — | — | — |
| ResNet-18 + **Random** (control) | 0.992 | 0.56 | 0.034 | 0.807 | 0.369 | 0.658 | 0.265 |

- ep-AUC = error-prediction AUC (how well the uncertainty signal ranks errors).
- Internal referral @25% retained error: MC 0.15% / MSP 0.30% / Ensemble 0.45% /
  Random 3.9% — vs external, where none beat random.
- Run `scripts/medknow_compare_referral_methods.py` to regenerate the referral
  comparison, and `scripts/medknow_evaluate_external.py` for the per-cohort metrics.

## 13. Reproduce the experiments

```bash
# 1. Environment
conda env create -f environment.yml        # or: pip install -e ".[dev]"
conda activate medknow

# 2. Data (images are NOT in this repo; see data/README.md)
python scripts/medknow_split_patient.py --seed 42  # patient-level split

# 3. Train (three seeds)
python scripts/medknow_train.py --config configs/baseline.yaml --seed 42
python scripts/medknow_train.py --config configs/baseline.yaml --seed 2024
python scripts/medknow_train.py --config configs/baseline.yaml --seed 2026

# 4. Internal evaluation
python scripts/medknow_evaluate.py --weights checkpoints/seed_42.pth

# 5. Calibration
python scripts/medknow_run_calibration.py --weights checkpoints/seed_42.pth

# 6. Referral / selective prediction
python scripts/medknow_run_referral.py --weights checkpoints/seed_42.pth

# 7. External validation (internal + RSNA + NIH)
python scripts/medknow_evaluate_external.py --weights checkpoints/seed_42.pth

# 8. Figures
python scripts/medknow_make_figures.py --weights checkpoints/seed_42.pth
```

## 14. Project structure

```text
medknow/
├── src/medknow/
│   ├── datasets/       # internal / RSNA / NIH loaders
│   ├── models/         # ResNet-18 factory (manuscript architecture)
│   ├── training/       # reproducible training + inference
│   ├── uncertainty/    # MC Dropout / MSP / entropy / random / ensemble
│   ├── calibration/    # ECE, Brier, reliability, temperature scaling
│   ├── referral/       # referral curves, risk-coverage, error-prediction AUC
│   ├── evaluation/     # unified metrics + subgroup analysis
│   └── visualization/  # fig01–fig08 builders
├── configs/            # baseline / mc_dropout / temperature_scaling
├── scripts/            # train / evaluate / referral / calibration / figures
├── tests/              # unit tests (74 passing)
├── results/            # figures / tables / metrics (regenerated)
├── notebooks/          # 🆕 one-click Colab demo
├── hf_space/           # 🆕 self-contained Hugging Face Space package
├── demo_app.py         # interactive Gradio demo (see hf_space/ for deploy)
├── paper/              # manuscript, figures, research notes
└── data/README.md      # dataset provenance (no images distributed)
```

## 15. Limitations

- Internal data is single-site and pediatric-centric (Kermany); external
  conclusions are bounded by the RSNA and NIH distributions.
- The NIH cohort has only 49 positive cases (0.5% prevalence); its AUC has a
  wide CI and its NLP-derived labels are noisier than expert adjudication.
- Referral thresholds were heuristic, not tuned to a target referral rate on a
  validation set (full curves are provided so institutions can choose).
- Bootstrap CIs resample images, not patients (optimistic for the internal
  test, which contains multiple images per patient).
- No radiologist comparison; reference standards are expert labels, not
  pathological confirmation.
- View-position confounding (AP/PA) could not be stratified because view
  metadata was not retained for RSNA.

## 16. Citation

If you use MedKnow in your research, please cite it:

```bibtex
@software{medknow2026,
  author = {MedKnow authors},
  title  = {MedKnow: Teaching Medical AI When Not to Know},
  year   = {2026},
  url    = {https://github.com/Solren-zhen/Medknow}
}
```

<!-- TODO(publish): add arXiv ID after preprint release -->

## 17. Dataset access

Images are not distributed in this repository. See
[data/README.md](data/README.md) for sources, licenses, and terms of use
(Kermany / Kaggle, RSNA competition, NIH ChestXray-14).

## 18. Contributing

Contributions are welcome — new uncertainty methods, more external cohorts,
bug reports, and documentation. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[issue templates](.github/ISSUE_TEMPLATE/). All contributions must keep the
project's honesty rules: **no fabricated numbers, no unrun claims** (see
[ROADMAP.md](ROADMAP.md) principles).

## 19. Disclaimer

This project is for research and education only. It is not a medical device,
has not been validated for clinical use, and must not be used for diagnosis or
treatment decisions.

---

If this project is useful to you, please ⭐ it — it tells us that honest
failure stories are worth publishing.
