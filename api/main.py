#!/usr/bin/env python3
"""
FastAPI 后端服务
=================
端点:
  POST /predict         — 单张图片预测
  GET  /health          — 健康检查
  GET  /model_info      — 模型信息
  GET  /history         — 预测历史
  GET  /history/{id}    — 单条记录
  PUT  /history/{id}/feedback — 提交反馈

启动: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import hashlib
import io
import os
import secrets
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.concurrency import run_in_threadpool
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from api.database import (
    get_history,
    get_prediction,
    get_statistics,
    insert_prediction,
    update_feedback,
)
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
    PredictionResponse,
)
from medknow.config import get as _get
from medknow.config import load_config
from medknow.models.factory import (
    enable_dropout,
    list_available_models,
    load_trained_model,
)
from utils.dicom_utils import read_dicom

_CONFIG = load_config(str(PROJECT_ROOT / "config.yaml"))


def cfg_get(key_path: str, default=None):
    """Read the formal package configuration using one dotted-path API."""
    return _get(_CONFIG, key_path, default)


CLASS_NAMES = list(cfg_get("data.class_names", ["NORMAL", "PNEUMONIA"]))
NUM_CLASSES = len(CLASS_NAMES)
_DEVICE_SELECTION = cfg_get("device.selection", "auto")
if _DEVICE_SELECTION == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(_DEVICE_SELECTION)
OUTPUT_DIR = Path(cfg_get("paths.output_dir", PROJECT_ROOT / "outputs"))

DEFAULT_CORS_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:7860",
    "http://127.0.0.1:7860",
]
DEFAULT_ALLOWED_CONTENT_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "application/dicom",
    "application/octet-stream",
]
DEFAULT_ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"]

MAX_UPLOAD_BYTES = int(float(cfg_get("api.max_upload_size_mb", 20)) * 1024 * 1024)
MAX_IMAGE_PIXELS = int(cfg_get("api.max_image_pixels", 12_000_000))
MAX_MC_SAMPLES = int(cfg_get("api.max_mc_samples", 50))
DEFAULT_MC_SAMPLES = min(int(cfg_get("uncertainty.mc_samples", 30)), MAX_MC_SAMPLES)
ALLOWED_CONTENT_TYPES = {
    str(item).lower()
    for item in cfg_get("api.allowed_content_types", DEFAULT_ALLOWED_CONTENT_TYPES)
}
ALLOWED_EXTENSIONS = {
    str(item).lower()
    for item in cfg_get("api.allowed_extensions", DEFAULT_ALLOWED_EXTENSIONS)
}

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def _list_from_config_or_env(env_name: str, key_path: str, default: list[str]) -> list[str]:
    raw_env = os.environ.get(env_name)
    if raw_env:
        return [item.strip() for item in raw_env.split(",") if item.strip()]

    value = cfg_get(key_path, default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return default


def _admin_token() -> str | None:
    token = os.environ.get("MEDKNOW_API_ADMIN_TOKEN") or cfg_get("api.admin_token")
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token.strip()


def require_admin(
    x_medknow_admin_token: str | None = Header(default=None, alias="X-MedKnow-Admin-Token"),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    """Protect history/statistics/feedback endpoints with an admin token."""
    expected = _admin_token()
    if expected is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="管理端点未启用：请设置 MEDKNOW_API_ADMIN_TOKEN 或 api.admin_token",
        )

    provided = x_medknow_admin_token or _bearer_token(authorization)
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要有效的 X-MedKnow-Admin-Token 或 Bearer token",
        )


def _is_dicom_upload(file: UploadFile) -> bool:
    suffix = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").split(";", 1)[0].lower()
    return suffix in {".dcm", ".dicom"} or content_type == "application/dicom"


def _validate_upload_metadata(file: UploadFile) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    content_type = (file.content_type or "").split(";", 1)[0].lower()

    if suffix and suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"不支持的文件扩展名: {suffix}")
    if not suffix and content_type not in {"application/dicom", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="无法识别上传文件类型")
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"不支持的 MIME 类型: {content_type}")


def _validate_image_pixels(image: Image.Image) -> None:
    pixels = int(image.width) * int(image.height)
    if pixels > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"图像像素数过大（{pixels}），上限 {MAX_IMAGE_PIXELS}",
        )


# ── 应用初始化 ──
app = FastAPI(
    title="肺炎X光AI辅助诊断 API",
    description="基于深度学习的胸部X光肺炎检测 REST API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_list_from_config_or_env("MEDKNOW_CORS_ORIGINS", "api.allowed_origins", DEFAULT_CORS_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-MedKnow-Admin-Token"],
)

# ── 模型管理 ──
_model_cache = {}


def get_model(model_name: str = "resnet18"):
    """获取缓存的模型实例"""
    if model_name not in _model_cache:
        if model_name != "resnet18":
            raise ValueError("正式 medknow 包当前仅支持 resnet18 API 推理")
        model_path = OUTPUT_DIR / "pneumonia_model.pth"
        model = load_trained_model(model_path, num_classes=NUM_CLASSES, device=DEVICE)
        _model_cache[model_name] = model
    return _model_cache[model_name]


def preprocess(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def _load_temperature() -> float | None:
    """读取训练阶段保存的温度参数（outputs/temperature.txt），不存在则返回 None"""
    temp_path = Path(OUTPUT_DIR) / "temperature.txt"
    if temp_path.exists():
        try:
            return float(temp_path.read_text().strip())
        except (OSError, ValueError):
            return None
    return None


def predict_with_uncertainty(model, input_tensor, n_samples=DEFAULT_MC_SAMPLES, temperature=None):
    """MC Dropout 推理"""
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
    all_probs = np.stack(all_probs)
    mean = all_probs.mean(axis=0)
    std = all_probs.std(axis=0)
    model.eval()
    return mean, std


def _run_model_prediction(
    image: Image.Image,
    model_name: str,
    enable_uncertainty: bool,
    mc_samples: int,
) -> dict:
    """Synchronous prediction helper, called from the API via a threadpool."""
    try:
        model = get_model(model_name)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    input_tensor = preprocess(image)
    temperature = _load_temperature()

    if enable_uncertainty:
        mean_probs, std_probs = predict_with_uncertainty(model, input_tensor, mc_samples, temperature)
        prob_normal, prob_pneumonia = float(mean_probs[0]), float(mean_probs[1])
        std_pneumonia = float(std_probs[1])
        uncertain = std_pneumonia > cfg_get("uncertainty.std_threshold", 0.05)
    else:
        with torch.no_grad():
            outputs = model(input_tensor)
            if temperature:
                outputs = outputs / temperature
            probs = F.softmax(outputs, dim=1)[0]
        prob_normal, prob_pneumonia = float(probs[0]), float(probs[1])
        std_pneumonia = None
        uncertain = None

    pred_idx = 1 if prob_pneumonia >= prob_normal else 0
    return {
        "prediction": CLASS_NAMES[pred_idx],
        "probability_normal": prob_normal,
        "probability_pneumonia": prob_pneumonia,
        "uncertainty_std": std_pneumonia,
        "uncertain": uncertain,
    }


# ═══════════════════════════════════════════════
# 端点
# ═══════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """服务健康检查"""
    model_loaded = os.path.exists(os.path.join(OUTPUT_DIR, "pneumonia_model.pth"))
    return HealthResponse(
        status="healthy" if model_loaded else "no_model",
        model_loaded=model_loaded,
        device=str(DEVICE),
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
    """获取模型和系统信息"""
    return ModelInfoResponse(
        model_name=cfg_get("training.model_name", "resnet18"),
        device=str(DEVICE),
        class_names=CLASS_NAMES,
        available_models=list(list_available_models().keys()),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),  # noqa: B008 - FastAPI 必需的依赖注入写法
    model_name: str = Query("resnet18", description="模型架构"),
    enable_uncertainty: bool = Query(True, description="启用不确定性量化"),
    mc_samples: int = Query(DEFAULT_MC_SAMPLES, ge=1, le=MAX_MC_SAMPLES, description="MC Dropout 采样次数"),
):
    """
    上传胸部X光图片进行肺炎检测

    - 支持 JPG/PNG/DICOM 格式
    - 返回诊断结果 + 概率 + 不确定性
    """
    if model_name not in list_available_models():
        raise HTTPException(
            status_code=400,
            detail=f"未知模型: {model_name}。可选: {list(list_available_models().keys())}",
        )

    # 读取文件
    _validate_upload_metadata(file)
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{len(content)} 字节），上限 {MAX_UPLOAD_BYTES} 字节 "
                   f"({cfg_get('api.max_upload_size_mb', 20)} MB)",
        )
    file_hash = hashlib.sha256(content).hexdigest()[:16]

    # 判断文件类型
    is_dcm = _is_dicom_upload(file)

    tmp_path = None
    try:
        if is_dcm:
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            image = Image.fromarray(read_dicom(tmp_path)).convert("RGB")
        else:
            image = Image.open(io.BytesIO(content))
            image.load()
            image = image.convert("RGB")
        _validate_image_pixels(image)
    except Exception as e:  # 图片解析失败统一返回 400
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=f"无法解析图片: {e!s}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    result = await run_in_threadpool(
        _run_model_prediction,
        image,
        model_name,
        enable_uncertainty,
        mc_samples,
    )
    timestamp = datetime.now(UTC).isoformat()

    # 保存到数据库
    record_id = insert_prediction({
        "timestamp": timestamp,
        "image_path": f"sha256:{file_hash}",
        "image_hash": file_hash,
        "model_name": model_name,
        "probability_normal": result["probability_normal"],
        "probability_pneumonia": result["probability_pneumonia"],
        "prediction": result["prediction"],
        "uncertainty_std": result["uncertainty_std"],
        "uncertain": result["uncertain"],
    })

    return PredictionResponse(
        id=record_id,
        timestamp=timestamp,
        model_name=model_name,
        prediction=result["prediction"],
        probability_normal=result["probability_normal"],
        probability_pneumonia=result["probability_pneumonia"],
        uncertainty_std=result["uncertainty_std"],
        uncertain=result["uncertain"],
        image_hash=file_hash,
    )


@app.get("/history", response_model=PredictionHistoryResponse)
async def prediction_history(
    limit: int = Query(50, ge=1, le=500),
    _: None = Depends(require_admin),
):
    """获取预测历史记录"""
    rows = get_history(limit)
    items = [
        PredictionHistoryItem(
            id=r["id"],
            timestamp=r["timestamp"],
            image_path=r["image_path"],
            prediction=r["prediction"],
            probability_normal=r["probability_normal"],
            probability_pneumonia=r["probability_pneumonia"],
            uncertain=bool(r.get("uncertain", False)) if r.get("uncertain") is not None else None,
        )
        for r in rows
    ]
    return PredictionHistoryResponse(total=len(items), items=items)


@app.get("/history/{pred_id}")
async def get_single_prediction(pred_id: int, _: None = Depends(require_admin)):
    """获取单条预测记录"""
    record = get_prediction(pred_id)
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")
    return record


@app.put("/history/{pred_id}/feedback")
async def submit_feedback(
    pred_id: int,
    feedback: str = Query(...),
    ground_truth: str | None = Query(None),
    _: None = Depends(require_admin),
):
    """提交反馈（用于主动学习）"""
    record = get_prediction(pred_id)
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")
    update_feedback(pred_id, feedback, ground_truth)
    return {"status": "ok", "id": pred_id}


@app.get("/statistics")
async def statistics(_: None = Depends(require_admin)):
    """获取预测统计"""
    return get_statistics()
