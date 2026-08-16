# -*- coding: utf-8 -*-
"""MedKnow — Streamlit demo (free forever on Streamlit Community Cloud).

Minimal single-file demo: upload a chest X-ray, get prediction + MC Dropout
uncertainty + Grad-CAM. CPU-only; weights auto-download from the GitHub
Release on first run.

Deploy: push this repo, then connect it at https://share.streamlit.io
"""
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE = 224
MC_SAMPLES = 15  # lower than the paper (30) so the free CPU tier stays responsive
STD_THRESHOLD = 0.05
DROPOUT_RATE = 0.3
TEMPERATURE = 1.67
MODEL_URL = "https://github.com/Solren-zhen/Medknow/releases/download/v1.0.0/seed_42.pth"


def _model_path() -> Path:
    for p in (Path("model.pth"), Path("hf_space/model.pth"), Path("streamlit_app/model.pth")):
        if p.exists():
            return p
    cache = Path(tempfile.gettempdir()) / "medknow_seed42.pth"
    if not cache.exists():
        with st.spinner("Downloading model weights (~43 MB)..."):
            urllib.request.urlretrieve(MODEL_URL, cache)
    return cache


@st.cache_resource
def load_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(model.fc.in_features, len(CLASS_NAMES)),
    )
    state = torch.load(_model_path(), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def enable_dropout(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_inference(model, x: torch.Tensor):
    model.eval()
    enable_dropout(model)
    probs = []
    for _ in range(MC_SAMPLES):
        logits = model(x) / TEMPERATURE
        probs.append(F.softmax(logits, dim=1)[0].numpy())
    probs = np.stack(probs)
    return probs.mean(axis=0), probs.std(axis=0)


def gradcam(model, x: torch.Tensor) -> np.ndarray:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(1)])[0]
    heat = Image.fromarray((grayscale * 255).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(heat, dtype=np.float32) / 255.0


def main() -> None:
    st.set_page_config(page_title="MedKnow — Pneumonia X-Ray AI", page_icon="🩻", layout="wide")
    st.title("🩻 MedKnow — Teaching Medical AI When Not to Know")
    st.caption("AUC 0.992 in-domain · 0.807 (RSNA) / 0.658 (NIH) external. Research demo only — not a medical device.")
    st.markdown("Upload a chest X-ray to see prediction, MC Dropout uncertainty, and where the model looks.")

    model = load_model()
    uploaded = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        x = transform(image).unsqueeze(0)

        probs, stds = mc_inference(model, x)
        p_pneu = float(probs[1])
        std = float(stds[1])
        verdict = "⚠️ uncertain → recommend human review" if std > STD_THRESHOLD else "✅ confident"

        c1, c2 = st.columns(2)
        with c1:
            st.image(image, caption="Input chest X-ray", use_container_width=True)
        with c2:
            st.metric("Pneumonia probability", f"{p_pneu:.1%}")
            st.metric("MC Dropout std (uncertainty)", f"{std:.3f}")
            st.write(f"**{verdict}**")
            st.write("Where the model looks (Grad-CAM):")
            heat = gradcam(model, x)
            st.image(heat, caption="Grad-CAM (pneumonia class)", use_container_width=True)


if __name__ == "__main__":
    main()
