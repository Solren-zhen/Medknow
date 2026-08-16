#!/usr/bin/env python3
"""
🩻 Pneumonia X-Ray AI — Interactive Gradio Demo

A one-click web demo for the pneumonia chest X-ray classifier:
prediction + MC Dropout uncertainty + Grad-CAM heatmap.

Local run:
    pip install -r requirements-demo.txt
    python demo_app.py

Deploy to Hugging Face Spaces (free, no GPU needed):
    see HF_SPACE.md
"""

import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEVICE, OUTPUT_DIR, cfg
from models.model_factory import (
    enable_dropout,
    get_target_layer,
    load_trained_model,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = list(cfg["data"]["class_names"])            # ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE = int(cfg["data"]["image_size"])               # 224
DEFAULT_MC_SAMPLES = int(cfg["uncertainty"]["mc_samples"])  # 30
STD_THRESHOLD = float(cfg["uncertainty"]["std_threshold"])  # 0.05

_ENV_MODEL_PATH = os.environ.get("MODEL_PATH")
MODEL_CANDIDATES = [
    Path(_ENV_MODEL_PATH).resolve() if _ENV_MODEL_PATH else None,
    PROJECT_ROOT / "outputs" / "pneumonia_model.pth",
    PROJECT_ROOT / "checkpoints" / "seed_42.pth",
    PROJECT_ROOT / "model.pth",
]
MODEL_CANDIDATES = [p for p in MODEL_CANDIDATES if p is not None]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

_model_cache = None
_temperature_cache = None


# ---------------------------------------------------------------------------
# Model / temperature loading
# ---------------------------------------------------------------------------
def _find_model_path() -> Path:
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Model weights not found. Looked in: "
        + ", ".join(str(p) for p in MODEL_CANDIDATES)
        + ". Train first (python scripts/train.py) or set the MODEL_PATH env var."
    )


def _load_temperature() -> float:
    global _temperature_cache
    if _temperature_cache is not None:
        return _temperature_cache
    candidates = [Path(OUTPUT_DIR) / "temperature.txt", PROJECT_ROOT / "temperature.txt"]
    _temperature_cache = 1.0
    for p in candidates:
        if p.exists():
            try:
                _temperature_cache = float(p.read_text(encoding="utf-8").strip())
                break
            except (OSError, ValueError):
                _temperature_cache = 1.0
    return _temperature_cache


def get_model():
    """Lazy-load the trained model (cached across requests)."""
    global _model_cache
    if _model_cache is None:
        model_path = _find_model_path()
        _model_cache = load_trained_model(
            name="resnet18",
            num_classes=len(CLASS_NAMES),
            model_path=str(model_path),
            device=DEVICE,
        )
        print(f"[demo] model loaded from {model_path} on {DEVICE}")
    return _model_cache, _load_temperature()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    img = tensor[0].cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def _jet_colormap(x: np.ndarray) -> np.ndarray:
    """Minimal jet-like colormap (pure numpy — no cv2/matplotlib needed)."""
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    return np.stack([r, g, b], axis=-1)


def gradcam_overlay(model, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
    """Grad-CAM heatmap overlaid on the input image (pneumonia class)."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    model.eval()
    target_layer = get_target_layer(model, "resnet18")
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])[0]

    img = denormalize(input_tensor)  # HxWx3, float 0..1
    h, w = img.shape[:2]
    heat = Image.fromarray((grayscale * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    heat = np.asarray(heat, dtype=np.float32) / 255.0
    overlay = 0.55 * img + 0.45 * _jet_colormap(heat)
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_with_uncertainty(model, input_tensor, n_samples: int, temperature: float):
    """MC Dropout: N stochastic forward passes -> mean probability + std."""
    model.eval()
    enable_dropout(model)
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(input_tensor) / temperature
            samples.append(F.softmax(logits, dim=1)[0].cpu().numpy())
    model.eval()
    samples = np.stack(samples)  # (N, C)
    return samples.mean(axis=0), samples.std(axis=0)


def interpret(p_pneu: float, std: float):
    """Bilingual verdict based on probability and uncertainty."""
    unc_note = (
        "⚠️ 模型对此病例不确定，建议人工复核 / The model is uncertain about this case — "
        "recommend human review."
        if std > STD_THRESHOLD
        else "✅ 模型对此预测较有把握 / The model is confident in this prediction."
    )
    tail = f"（肺炎概率 {p_pneu:.1%}）" if p_pneu >= 0.5 else f"（正常概率 {1 - p_pneu:.1%}）"
    if p_pneu >= 0.7:
        return "🔴 高度疑似肺炎 / High suspicion of pneumonia", (
            f"肺炎概率 {p_pneu:.1%}，建议结合临床进一步确诊。{unc_note}"
        )
    if p_pneu >= 0.5:
        return "🟠 疑似肺炎 / Suspicious for pneumonia", (
            f"模型认为存在肺炎可能（{p_pneu:.1%}），建议影像科复核。{unc_note}"
        )
    if p_pneu >= 0.3:
        return "🟡 边界情况 / Borderline", (
            f"模型倾向正常但把握不大{tail}，建议随访复查。{unc_note}"
        )
    return "🟢 倾向正常 / Normal-appearing", (
        f"模型认为未见明显肺炎征象{tail}。{unc_note}"
    )


def analyze(image, use_uncertainty: bool, mc_samples: int, show_heatmap: bool):
    if image is None:
        return None, {}, "请上传胸片 / Please upload a chest X-ray image.", ""
    try:
        model, temperature = get_model()
    except FileNotFoundError as exc:
        return None, {}, f"模型权重缺失 / Model weights missing: {exc}", ""

    pil = Image.fromarray(image.astype("uint8")).convert("RGB")
    x = preprocess(pil)

    if use_uncertainty and int(mc_samples) > 1:
        probs, stds = predict_with_uncertainty(model, x, int(mc_samples), temperature)
    else:
        model.eval()
        with torch.no_grad():
            logits = model(x) / temperature
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        stds = np.zeros_like(probs)

    p_norm = float(probs[0])
    p_pneu = float(probs[1])
    std = float(stds[1])

    label_map = {CLASS_NAMES[0]: p_norm, CLASS_NAMES[1]: p_pneu}
    prob_text = (
        f"Pneumonia 肺炎概率: {p_pneu:.1%}  |  MC Dropout std: {std:.3f}  |  "
        f"Temperature: {temperature:.2f}"
    )
    verdict, note = interpret(p_pneu, std)

    heatmap = gradcam_overlay(model, x, target_class=1) if show_heatmap else None
    if heatmap is None:
        heatmap = np.ascontiguousarray(image)
    return heatmap, label_map, prob_text, f"{verdict}\n{note}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
HEADER = """
<div align="center">
<h1>🩻 Pneumonia X-Ray AI</h1>
<p><b>AUC 0.992 at home · 0.81 at another hospital.</b> Upload a chest X-ray and see
what a medical AI actually does: prediction, uncertainty, and where it looks.</p>
<p><i>Research / education demo only — not a medical device.</i></p>
</div>
"""

DISCLAIMER = """
---
⚠️ **免责声明 / Disclaimer**: This demo is for research and education only.
It is not a medical device and must not be used for diagnosis. Uncertainty-aware
referral works within the training domain and **fails across institutions** —
see the project README for the honest results.
"""


def build_demo():
    examples = sorted(str(p) for p in (PROJECT_ROOT / "examples").glob("*.jpeg"))

    with gr.Blocks(title="Pneumonia X-Ray AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    type="numpy", height=420, label="胸片 / Chest X-ray (JPG/PNG)"
                )
                with gr.Accordion("⚙️ 高级选项 / Advanced options", open=False):
                    use_unc = gr.Checkbox(value=True, label="启用 MC Dropout 不确定性")
                    mc_samples = gr.Slider(
                        5, 100, value=DEFAULT_MC_SAMPLES, step=5, label="MC 采样次数"
                    )
                    show_cam = gr.Checkbox(value=True, label="显示 Grad-CAM 热力图")
                btn = gr.Button("🔍 分析 / Analyze", variant="primary")
            with gr.Column(scale=1):
                heatmap = gr.Image(height=420, label="Grad-CAM 热力图 / Where the model looks")
                pred = gr.Label(label="预测 / Prediction")
                prob = gr.Textbox(label="概率与不确定性 / Probability & uncertainty", interactive=False)
                note = gr.Textbox(label="解读 / Interpretation", lines=4, interactive=False)

        btn.click(
            analyze,
            inputs=[image, use_unc, mc_samples, show_cam],
            outputs=[heatmap, pred, prob, note],
        )

        def analyze_example(img):
            return analyze(img, True, DEFAULT_MC_SAMPLES, True)

        gr.Examples(
            examples=examples,
            inputs=[image],
            outputs=[heatmap, pred, prob, note],
            fn=analyze_example,
            label="示例 / Try an example",
        )
        gr.Markdown(DISCLAIMER)
    return demo


if __name__ == "__main__":
    build_demo().queue().launch()
