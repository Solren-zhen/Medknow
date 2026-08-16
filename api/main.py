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
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from config import CLASS_NAMES, DEVICE, NUM_CLASSES, OUTPUT_DIR, cfg
from models.model_factory import (
    enable_dropout,
    list_available_models,
    load_trained_model,
)
from utils.dicom_utils import read_dicom

# ── 应用初始化 ──
app = FastAPI(
    title="肺炎X光AI辅助诊断 API",
    description="基于深度学习的胸部X光肺炎检测 REST API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 模型管理 ──
_model_cache = {}


def get_model(model_name: str = "resnet18"):
    """获取缓存的模型实例"""
    if model_name not in _model_cache:
        model = load_trained_model(name=model_name, num_classes=NUM_CLASSES, device=DEVICE)
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


def predict_with_uncertainty(model, input_tensor, n_samples=30, temperature=None):
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
        model_name=cfg.get("training.model_name", "resnet18"),
        device=str(DEVICE),
        class_names=CLASS_NAMES,
        available_models=list(list_available_models().keys()),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),  # noqa: B008 - FastAPI 必需的依赖注入写法
    model_name: str = Query("resnet18", description="模型架构"),
    enable_uncertainty: bool = Query(True, description="启用不确定性量化"),
    mc_samples: int = Query(30, ge=10, le=200, description="MC Dropout 采样次数"),
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
    content = await file.read()
    max_bytes = int(cfg.get("api.max_upload_size_mb", 20) * 1024 * 1024)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{len(content)} 字节），上限 {max_bytes} 字节 "
                   f"({cfg.get('api.max_upload_size_mb', 20)} MB)",
        )
    file_hash = hashlib.sha256(content).hexdigest()[:16]

    # 判断文件类型
    is_dcm = file.filename.lower().endswith((".dcm", ".dicom"))

    try:
        if is_dcm:
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            image = Image.fromarray(read_dicom(tmp_path))
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        else:
            image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:  # noqa: BLE001 - 图片解析失败统一返回 400
        raise HTTPException(status_code=400, detail=f"无法解析图片: {e!s}")

    # 推理
    try:
        model = get_model(model_name)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    input_tensor = preprocess(image)
    temperature = _load_temperature()

    if enable_uncertainty:
        mean_probs, std_probs = predict_with_uncertainty(model, input_tensor, mc_samples, temperature)
        prob_normal, prob_pneumonia = float(mean_probs[0]), float(mean_probs[1])
        std_pneumonia = float(std_probs[1])
        uncertain = std_pneumonia > cfg.get("uncertainty.std_threshold", 0.05)
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
    prediction = CLASS_NAMES[pred_idx]

    # 保存到数据库
    record_id = insert_prediction({
        "timestamp": datetime.now(UTC).isoformat(),
        "image_path": file.filename,
        "image_hash": file_hash,
        "model_name": model_name,
        "probability_normal": prob_normal,
        "probability_pneumonia": prob_pneumonia,
        "prediction": prediction,
        "uncertainty_std": std_pneumonia,
        "uncertain": uncertain,
    })

    return PredictionResponse(
        id=record_id,
        timestamp=datetime.now(UTC).isoformat(),
        model_name=model_name,
        prediction=prediction,
        probability_normal=prob_normal,
        probability_pneumonia=prob_pneumonia,
        uncertainty_std=std_pneumonia,
        uncertain=uncertain,
        image_hash=file_hash,
    )


@app.get("/history", response_model=PredictionHistoryResponse)
async def prediction_history(limit: int = Query(50, ge=1, le=500)):
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
async def get_single_prediction(pred_id: int):
    """获取单条预测记录"""
    record = get_prediction(pred_id)
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")
    return record


@app.put("/history/{pred_id}/feedback")
async def submit_feedback(pred_id: int, feedback: str = Query(...), ground_truth: str | None = Query(None)):
    """提交反馈（用于主动学习）"""
    record = get_prediction(pred_id)
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")
    update_feedback(pred_id, feedback, ground_truth)
    return {"status": "ok", "id": pred_id}


@app.get("/statistics")
async def statistics():
    """获取预测统计"""
    return get_statistics()
