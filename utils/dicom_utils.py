#!/usr/bin/env python3
"""
DICOM 医学影像格式读取工具
支持 .dcm 文件读取，提取像素数据和元信息
"""

from pathlib import Path

import numpy as np
from PIL import Image


def is_dicom_file(filepath: str) -> bool:
    """判断文件是否为 DICOM 格式"""
    ext = Path(filepath).suffix.lower()
    if ext in (".dcm", ".dicom"):
        return True
    # 尝试读取文件头判断（DICOM 文件头 128 字节 + 'DICM'）
    try:
        with open(filepath, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def read_dicom(filepath: str) -> np.ndarray:
    """
    读取 DICOM 文件，返回归一化的 RGB 图像数组

    Args:
        filepath: .dcm 文件路径

    Returns:
        numpy array, shape (H, W, 3), dtype uint8, 值域 0-255
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("请安装 pydicom: pip install pydicom")

    ds = pydicom.dcmread(filepath)
    img = ds.pixel_array.astype(np.float32)

    # 归一化到 0-255
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255.0
    else:
        img = np.zeros_like(img)
    img = img.astype(np.uint8)

    # 单通道 → RGB 三通道
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    return img


def extract_dicom_metadata(filepath: str) -> dict[str, str]:
    """
    提取 DICOM 元信息（患者/检查相关字段）

    Args:
        filepath: .dcm 文件路径

    Returns:
        字典：{标签: 值}
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("请安装 pydicom: pip install pydicom")

    ds = pydicom.dcmread(filepath)
    fields = {
        "PatientID": "患者ID",
        "PatientName": "患者姓名",
        "PatientAge": "患者年龄",
        "PatientSex": "患者性别",
        "StudyDate": "检查日期",
        "Modality": "检查类型",
        "BodyPartExamined": "检查部位",
        "StudyDescription": "检查描述",
        "InstitutionName": "机构名称",
    }

    metadata = {}
    for tag, label in fields.items():
        if hasattr(ds, tag):
            val = getattr(ds, tag)
            metadata[label] = str(val) if val else "未知"

    return metadata


def dicom_to_pil(filepath: str) -> Image.Image:
    """读取 DICOM 并转为 PIL Image"""
    arr = read_dicom(filepath)
    return Image.fromarray(arr)
