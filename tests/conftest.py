#!/usr/bin/env python3
"""pytest 共享 fixtures"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 测试使用独立的临时数据库，避免污染 outputs/predictions.db
_TEST_DB_DIR = tempfile.mkdtemp(prefix="pneumonia_test_")
os.environ["PNEUMONIA_DB_PATH"] = os.path.join(_TEST_DB_DIR, "test_predictions.db")


@pytest.fixture
def device():
    """测试用的设备"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def sample_tensor():
    """模拟的输入 tensor (1, 3, 224, 224)"""
    return torch.randn(1, 3, 224, 224)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
