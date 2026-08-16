#!/usr/bin/env python3
"""
Pydantic 请求/响应模型
"""


from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """单张图片预测请求"""
    image_base64: str = Field(..., description="Base64 编码的图片 (JPG/PNG/DICOM)")


class PredictionResponse(BaseModel):
    """预测结果"""
    id: int
    timestamp: str
    model_name: str
    prediction: str
    probability_normal: float
    probability_pneumonia: float
    uncertainty_std: float | None = None
    uncertain: bool | None = None
    image_hash: str


class BatchPredictionResponse(BaseModel):
    """批量预测结果"""
    total: int
    results: list[PredictionResponse]


class ModelInfoResponse(BaseModel):
    """模型信息"""
    model_name: str
    device: str
    class_names: list[str]
    available_models: list[str]


class PredictionHistoryItem(BaseModel):
    """预测历史条目"""
    id: int
    timestamp: str
    image_path: str
    prediction: str
    probability_normal: float
    probability_pneumonia: float
    uncertain: bool | None = None


class PredictionHistoryResponse(BaseModel):
    """预测历史"""
    total: int
    items: list[PredictionHistoryItem]


class HealthResponse(BaseModel):
    """健康检查"""
    status: str = "healthy"
    model_loaded: bool
    device: str
    timestamp: str
