# MedKnow model card

## Model

The published baseline is a binary ResNet-18 classifier with ImageNet
initialisation, a frozen backbone except for `layer4`, and a dropout head
(`p=0.3`). It predicts `NORMAL` and `PNEUMONIA` from a 224×224 RGB chest
radiograph. MC Dropout is available for an uncertainty estimate.

## Intended use

This model is intended for research, education, and reproducibility studies of
selective prediction and domain shift. It is not a medical device and must not
be used for diagnosis, triage, treatment, or unsupervised clinical decisions.

## Evaluation context

The reported internal results use the Kermany chest X-ray dataset with a
patient-level split. External evaluation uses RSNA Pneumonia Detection
Challenge and a two-class NIH ChestXray-14 subset. The cohorts have different
label semantics, prevalence, acquisition environments, and populations.
Headline metrics in the README are repository result artifacts, not claims of
clinical performance.

## Limitations and risks

- The internal cohort is predominantly pediatric and single-site.
- The NIH external cohort contains very few positive cases and NLP-derived
  labels, so uncertainty is substantial.
- External specificity and calibration can degrade sharply under domain shift.
- A probability threshold or uncertainty threshold must be revalidated locally.
- The model may encode acquisition, demographic, or dataset-specific shortcuts.
- No radiologist comparison, prospective study, or clinical impact analysis is
  provided.

## Reproducibility

Training checkpoints save model, preprocessing, class-order, dropout, seed,
Python, and PyTorch metadata. Use the matching repository commit and the
evaluation protocol in `docs/research/evaluation_protocol.md`.
