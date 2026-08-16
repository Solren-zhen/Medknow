#!/usr/bin/env python3
"""
🩻 肺炎X光AI辅助诊断系统 — Web应用 v2.0
============================================
新特性:
  - 多模型选择 (ResNet18 / EfficientNet-B0 / DenseNet121)
  - 多维可解释性仪表盘 (Grad-CAM + Integrated Gradients + Occlusion)
  - 不确定性量化 (MC Dropout)
  - DICOM 格式支持
  - 结构化报告生成
  - 预测历史记录

运行: streamlit run app.py
"""

import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEVICE, NUM_CLASSES, OUTPUT_DIR, cfg
from models.model_factory import (
    enable_dropout,
    get_target_layer,
    list_available_models,
    load_trained_model,
)
from utils.dicom_utils import extract_dicom_metadata, read_dicom

# ═══════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="肺炎X光AI辅助诊断系统",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# 缓存资源
# ═══════════════════════════════════════════════════════════

@st.cache_resource
def load_model_cached(model_name: str = "resnet18", model_path: str | None = None):
    """加载模型（带 Streamlit 缓存）"""
    return load_trained_model(name=model_name, num_classes=NUM_CLASSES, model_path=model_path, device=DEVICE)


# ═══════════════════════════════════════════════════════════
# 图像预处理
# ═══════════════════════════════════════════════════════════

def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """PIL Image → 模型输入 tensor"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """将归一化的 tensor 还原为可视化 numpy array"""
    img = tensor[0].cpu().numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)


# ═══════════════════════════════════════════════════════════
# 推理函数
# ═══════════════════════════════════════════════════════════

def load_temperature():
    """读取训练阶段保存的温度参数（outputs/temperature.txt），不存在则返回 None"""
    temp_path = Path(OUTPUT_DIR) / "temperature.txt"
    if temp_path.exists():
        try:
            return float(temp_path.read_text().strip())
        except (OSError, ValueError):
            return None
    return None


def predict(model, input_tensor, temperature=None):
    """单次推理"""
    with torch.no_grad():
        outputs = model(input_tensor)
        if temperature:
            outputs = outputs / temperature
        probs = F.softmax(outputs, dim=1)[0]
    return probs


def predict_with_uncertainty(model, input_tensor, n_samples: int = 30, temperature=None):
    """MC Dropout 推理 — 返回均值、标准差和所有样本"""
    model.eval()
    enable_dropout(model)

    all_probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(input_tensor)
            if temperature:
                outputs = outputs / temperature
            probs = F.softmax(outputs, dim=1)[0]
            all_probs.append(probs.cpu().numpy())

    all_probs = np.stack(all_probs)  # (N, 2)
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)

    # 恢复为 eval 模式
    model.eval()
    return mean_probs, std_probs, all_probs


def explain_gradcam(model, input_tensor, target_class, model_name):
    """Grad-CAM 热力图"""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        target_layer = get_target_layer(model, model_name)
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])[0]
        img_np = denormalize(input_tensor)
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        return visualization
    except Exception as e:  # noqa: BLE001 - UI 友好提示，故意捕获所有异常
        st.warning(f"Grad-CAM 生成失败: {e}")
        return None


def explain_occlusion(model, input_tensor, target_class, window_size=32, stride=16):
    """Occlusion sensitivity 快速版"""
    _, _, H, W = input_tensor.shape
    sensitivity_map = np.zeros((H, W))
    count_map = np.zeros((H, W))

    with torch.no_grad():
        orig_prob = F.softmax(model(input_tensor), dim=1)[0, target_class].item()

    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            occluded = input_tensor.clone()
            occluded[:, :, y:y+window_size, x:x+window_size] = 0
            prob = F.softmax(model(occluded), dim=1)[0, target_class].item()
            drop = max(0, orig_prob - prob)
            sensitivity_map[y:y+window_size, x:x+window_size] += drop
            count_map[y:y+window_size, x:x+window_size] += 1

    sensitivity_map = sensitivity_map / (count_map + 1e-8)
    sensitivity_map = (sensitivity_map - sensitivity_map.min()) / (sensitivity_map.max() - sensitivity_map.min() + 1e-8)

    # 叠加到原图
    img_np = denormalize(input_tensor)
    heatmap = (sensitivity_map * 255).astype(np.uint8)
    heatmap_colored = np.array(Image.fromarray(heatmap).resize(
        (img_np.shape[1], img_np.shape[0]), Image.BILINEAR
    )) / 255.0
    heatmap_colored = plt.cm.jet(heatmap_colored)[:, :, :3]

    overlay = img_np * 0.5 + heatmap_colored * 0.5
    return (np.clip(overlay, 0, 1) * 255).astype(np.uint8)


def interpret_result(prob_normal, prob_pneumonia, std_pneumonia=0):
    """医学语言解读 + 不确定性"""
    p = prob_pneumonia
    std = std_pneumonia
    uncertainty_note = ""

    if std > 0.05:
        uncertainty_note = "\n⚠️ 模型对该病例存在不确定性，强烈建议结合临床检查复核"

    if p >= 0.7:
        return "🔴 高度疑似肺炎", f"模型置信度较高（{p*100:.1f}%），建议结合临床表现进一步确诊{uncertainty_note}"
    elif p >= 0.5:
        return "🟠 疑似肺炎", f"模型认为存在肺炎可能（{p*100:.1f}%），建议影像科复核{uncertainty_note}"
    elif p >= 0.3:
        return "🟡 边界情况", f"模型倾向正常但把握不大（肺炎概率{p*100:.1f}%），建议随访复查{uncertainty_note}"
    else:
        return "🟢 倾向正常", f"模型认为未见明显肺炎征象（正常概率{(1-p)*100:.1f}%）{uncertainty_note}"


# ═══════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🩻 肺炎X光 AI 辅助诊断")
    st.caption("v2.0 · 可解释性医学影像AI")

    st.divider()

    # 模型选择
    available_models = list_available_models()
    model_options = list(available_models.keys())
    selected_model = st.selectbox(
        "🧠 模型架构",
        model_options,
        index=model_options.index("resnet18") if "resnet18" in model_options else 0,
        format_func=lambda x: f"{available_models[x]['name_cn']} ({available_models[x]['params']})",
    )

    # 推理选项
    st.divider()
    st.subheader("⚙️ 推理选项")
    use_uncertainty = st.checkbox("启用不确定性量化 (MC Dropout)", value=True)
    mc_samples = st.slider("MC 采样次数", 10, 100, 30, 10) if use_uncertainty else 30
    show_xai = st.checkbox("显示可解释性分析", value=True)
    xai_method = st.radio("XAI 方法", ["Grad-CAM", "Occlusion Sensitivity", "两者对比"], index=0) if show_xai else None

    st.divider()
    st.subheader("📋 系统信息")
    st.write(f"""
    - **框架**: PyTorch + Streamlit
    - **设备**: {DEVICE}
    - **模型**: ResNet18/EfficientNet/DenseNet
    - **XAI**: Grad-CAM + Occlusion
    - **不确定性**: MC Dropout
    - **格式**: JPG/PNG/DICOM
    """)

    st.divider()
    st.warning("⚠️ 本项目为科研演示，不构成任何医疗诊断建议")

    # 预测历史
    if "history" in st.session_state and st.session_state.history:
        st.divider()
        st.subheader("📜 预测历史")
        for i, h in enumerate(reversed(st.session_state.history[-10:])):
            with st.expander(f"{h['time']} — {h['verdict']}", expanded=False):
                st.write(f"正常: {h['prob_normal']:.1f}% | 肺炎: {h['prob_pneumonia']:.1f}%")
                if h.get("uncertain"):
                    st.warning("⚠️ 模型不确定")


# ═══════════════════════════════════════════════════════════
# 主界面
# ═══════════════════════════════════════════════════════════

st.title("🩻 肺炎X光 AI 辅助诊断系统")
st.caption("可解释性医学影像AI | 深度学习 + 多维XAI + 不确定性量化 | 仅供科研学习")

# 初始化 session state
if "history" not in st.session_state:
    st.session_state.history = []

# ── 上传区域 ──
col_upload, col_result = st.columns([1, 1])

with col_upload:
    st.subheader("📤 上传X光胸片")
    uploaded = st.file_uploader(
        "选择图片 (JPG/PNG/DICOM)",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        help="支持普通图片和DICOM医学影像格式",
    )

    if uploaded is not None:
        # 判断文件类型
        file_bytes = uploaded.read()
        uploaded.seek(0)  # reset for later

        is_dcm = uploaded.name.lower().endswith((".dcm", ".dicom"))

        if is_dcm:
            # DICOM 处理
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                dicom_meta = extract_dicom_metadata(tmp_path)
                image = Image.fromarray(read_dicom(tmp_path))
                st.success("✅ DICOM 文件已加载")
                with st.expander("📋 DICOM 元信息"):
                    for k, v in dicom_meta.items():
                        if v and v != "未知":
                            st.text(f"{k}: {v}")
            except ImportError:
                st.error("请安装 pydicom: pip install pydicom")
                st.stop()
            except Exception as e:  # noqa: BLE001 - UI 友好提示，故意捕获所有异常
                st.error(f"DICOM 读取失败: {e}")
                st.stop()
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        else:
            image = Image.open(uploaded).convert("RGB")

        st.image(image, caption="原始胸片", use_container_width=True)
        st.caption(f"分辨率: {image.size[0]}×{image.size[1]}")

        # 保存 session 状态
        st.session_state.current_image = image
        st.session_state.is_dicom = is_dcm

# ── 结果区域 ──
if uploaded is not None and "current_image" in st.session_state:
    image = st.session_state.current_image

    # 加载模型
    try:
        model = load_model_cached(selected_model)
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.info("请先运行: python scripts/train.py 训练模型")
        st.stop()
    except Exception as e:  # noqa: BLE001 - UI 友好提示，故意捕获所有异常
        st.error(f"❌ 模型加载失败: {e}")
        st.stop()

    input_tensor = preprocess_image(image)
    temperature = load_temperature()

    with col_result:
        st.subheader("🤖 AI 诊断结果")

        with st.spinner("AI 分析中..."):
            if use_uncertainty:
                mean_probs, std_probs, all_probs = predict_with_uncertainty(
                    model, input_tensor, mc_samples, temperature
                )
                prob_normal = mean_probs[0]
                prob_pneumonia = mean_probs[1]
                std_pneumonia = std_probs[1]
                uncertain = std_pneumonia > cfg.get("uncertainty.std_threshold", 0.05)
            else:
                probs = predict(model, input_tensor, temperature)
                prob_normal = probs[0].item()
                prob_pneumonia = probs[1].item()
                std_pneumonia = 0
                uncertain = False

            verdict, note = interpret_result(prob_normal, prob_pneumonia, std_pneumonia)

            # 生成 Grad-CAM
            gradcam_img = None
            occlusion_img = None
            if show_xai and (xai_method in ["Grad-CAM", "两者对比"]):
                gradcam_img = explain_gradcam(model, input_tensor, target_class=1, model_name=selected_model)
            if show_xai and (xai_method in ["Occlusion Sensitivity", "两者对比"]):
                with st.spinner("Occlusion Sensitivity 计算中 (约30秒)..."):
                    try:
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as plt
                        _ = plt.cm.jet  # ensure colormap accessible
                        occlusion_img = explain_occlusion(model, input_tensor, target_class=1, window_size=32, stride=16)
                    except Exception as e:  # noqa: BLE001 - UI 友好提示，故意捕获所有异常
                        st.warning(f"Occlusion 计算失败: {e}")

    # ── 结果展示 ──
    st.divider()

    # 指标行
    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("正常概率", f"{prob_normal*100:.1f}%")
    with metric_cols[1]:
        st.metric("肺炎概率", f"{prob_pneumonia*100:.1f}%")
    with metric_cols[2]:
        verdict_short, _ = interpret_result(prob_normal, prob_pneumonia, 0)
        st.metric("诊断", verdict_short)
    with metric_cols[3]:
        device_name = "MPS" if str(DEVICE) == "mps" else str(DEVICE).upper()
        st.metric("推理设备", device_name)
    with metric_cols[4]:
        if use_uncertainty:
            uncertain_label = "⚠️ 不确定" if uncertain else "✅ 有把握"
            st.metric("不确定度", uncertain_label, delta=f"σ={std_pneumonia*100:.1f}%")

    # 临床解读
    st.info(f"📋 **临床解读**: {note}")

    # ── 可解释性展示 ──
    if show_xai and (gradcam_img is not None or occlusion_img is not None):
        st.divider()
        st.subheader("🔍 可解释性分析")

        if xai_method == "两者对比" and gradcam_img is not None and occlusion_img is not None:
            xai_col1, xai_col2 = st.columns(2)
            with xai_col1:
                st.image(gradcam_img, caption="Grad-CAM 热力图", use_container_width=True)
                st.caption("梯度加权类激活映射 — 展示模型做判断时最关注的区域")
            with xai_col2:
                st.image(occlusion_img, caption="Occlusion Sensitivity", use_container_width=True)
                st.caption("遮挡敏感度 — 遮挡不同区域后概率变化最大的位置")
        elif gradcam_img is not None and xai_method == "Grad-CAM":
            st.image(gradcam_img, caption="Grad-CAM 热力图（红色=模型关注的病灶区域）", use_container_width=True)
        elif occlusion_img is not None and xai_method == "Occlusion Sensitivity":
            st.image(occlusion_img, caption="Occlusion Sensitivity（红色=对分类影响最大的区域）", use_container_width=True)

    # ── 不确定性可视化 ──
    if use_uncertainty and 'all_probs' in dir():
        st.divider()
        st.subheader("🎲 不确定性分析")

        uncert_col1, uncert_col2 = st.columns(2)
        with uncert_col1:
            # 概率分布直方图
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(all_probs[:, 1], bins=20, color="steelblue", edgecolor="white", alpha=0.8)
            ax.axvline(x=prob_pneumonia, color="red", linestyle="--", linewidth=2, label=f"均值={prob_pneumonia*100:.1f}%")
            ax.set_xlabel("肺炎概率")
            ax.set_ylabel("频次")
            ax.set_title(f"MC Dropout 采样分布 (N={mc_samples})")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        with uncert_col2:
            st.write("**不确定性解读**")
            st.write(f"- **均值**: {prob_pneumonia*100:.1f}% — 最可能的肺炎概率")
            st.write(f"- **标准差**: {std_pneumonia*100:.1f}% — 模型预测的波动程度")
            st.write(f"- **方差**: {std_pneumonia**2:.5f} — 不确定性度量")
            st.write(f"- **阈值**: {cfg.get('uncertainty.variance_threshold', 0.05)}")

            if uncertain:
                st.error("⚠️ **高不确定度！** 模型对该病例的预测存在显著分歧，强烈建议由放射科医生复核。")
            else:
                st.success("✅ **低不确定度** — 模型对该预测有较高把握。")

            st.divider()
            st.caption("""
            **MC Dropout 原理**: 推理时保持 Dropout 层活跃，多次前向传播产生不同的预测结果。
            若模型对同一张图片的预测结果差异很大，说明模型对该病例"不确定"——这在临床中比单一预测更重要。
            """)

    # ── 记录历史 ──
    st.session_state.history.append({
        "time": datetime.now(UTC).astimezone().strftime("%H:%M:%S"),
        "model": selected_model,
        "prob_normal": prob_normal * 100,
        "prob_pneumonia": prob_pneumonia * 100,
        "verdict": verdict_short,
        "uncertain": uncertain if use_uncertainty else None,
    })

else:
    with col_result:
        st.subheader("🤖 AI 诊断结果")
        st.info("👈 请先上传一张胸部X光片")
        st.write("")
        st.write("**支持的格式**:")
        st.markdown("- 🖼️ 普通图片: JPG, JPEG, PNG")
        st.markdown("- 🏥 医学影像: DICOM (.dcm)")
        st.write("")
        st.write("**示例路径**:")
        st.code("data/chest_xray/test/PNEUMONIA/", language="text")
        st.code("data/chest_xray/test/NORMAL/", language="text")

# ═══════════════════════════════════════════════════════════
# 页脚
# ═══════════════════════════════════════════════════════════
st.divider()
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("**技术栈**: PyTorch · Streamlit · ResNet/EfficientNet/DenseNet · Grad-CAM · MC Dropout")
with footer_col2:
    st.caption(f"**设备**: {DEVICE} | **架构**: {selected_model}")
with footer_col3:
    st.caption("**项目**: 肺炎X光AI辅助诊断系统 | 临床医学 × 人工智能 · 医工交叉")
