---
title: MedKnow — Pneumonia X-Ray AI
emoji: 🩻
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🩻 MedKnow — Pneumonia X-Ray AI

**An honest medical-AI demo.** Upload a chest X-ray to see prediction + MC Dropout
uncertainty + Grad-CAM heatmap — and learn *when medical AI should say "I don't know".*

> ⚠️ **Research / education demo only — not a medical device.** Do not use for diagnosis.

## Model

| | |
|---|---|
| Task | Binary pneumonia classification (NORMAL / PNEUMONIA) |
| Architecture | ResNet-18, ImageNet-pretrained, frozen backbone + layer4 + dropout head (p=0.3) |
| Training data | Kermany chest X-ray, 5,856 images, patient-level split, 3 seeds |
| Internal AUC | 0.992 (seed_42 test) |
| External AUC | 0.807 (RSNA) · 0.658 (NIH ChestXray-14) |
| Calibration | ECE 0.034 internal → 0.369 (RSNA) under domain shift |
| Uncertainty | MC Dropout (30 stochastic passes), std of pneumonia probability |

## What this demo shows

1. **Prediction** — probability of pneumonia (temperature-scaled, T=1.67).
2. **Uncertainty** — MC Dropout std; high std ⇒ "uncertain, refer to human".
3. **Grad-CAM** — where the model looks in the image.

The headline finding: uncertainty-aware referral works *within the training domain*
(25% referral ⇒ zero missed pneumonias) but **fails across institutions** — the model
can be confident and wrong under domain shift. Confidence-based medical-AI triage
cannot be trusted on internal validation alone.

## Files in this Space

- `app.py` — the Gradio app (self-contained, no other repo files needed)
- `model.pth` — ResNet-18 seed_42 checkpoint
  (`outputs/pneumonia_model.pth` in the repository)
- `temperature.txt` — temperature-scaling constant (T=1.67)
- `examples/` — sample chest X-rays

## Source & citation

- GitHub: <https://github.com/Solren-zhen/Medknow>
- Manuscript: `paper/output/doc/manuscript.md`
- Citation: see `CITATION.cff` in the repository

## Run locally

```bash
pip install -r requirements.txt
python app.py
```
