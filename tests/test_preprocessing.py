#!/usr/bin/env python3
"""图像预处理测试"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestImagePreprocessing:
    """测试图像预处理管道"""

    def test_tensor_shape(self):
        """验证输出形状"""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # 创建测试图片
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_batch_shape(self):
        """验证批次维度"""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        tensor = transform(img).unsqueeze(0)
        assert tensor.shape == (1, 3, 224, 224)

    def test_rgb_conversion(self):
        """验证灰度图转 RGB"""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        # 灰度图 (H, W)
        img = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8))
        img_rgb = img.convert("RGB")
        tensor = transform(img_rgb)
        assert tensor.shape[0] == 3  # RGB 三通道

    def test_normalization_range(self):
        """验证归一化后的值域"""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        tensor = transform(img)
        # 归一化后值应该在 -2.5 ~ 2.5 左右
        assert tensor.min() > -4.0
        assert tensor.max() < 4.0


class TestDICOMUtils:
    """测试 DICOM 工具"""

    def test_is_dicom_by_extension(self):
        from utils.dicom_utils import is_dicom_file
        assert is_dicom_file("test.dcm")
        assert is_dicom_file("test.dicom")
        assert not is_dicom_file("test.jpg")
        assert not is_dicom_file("test.png")


class TestDataAugmentation:
    """测试数据增强"""

    def test_random_flip(self):
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
        ])
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        t1 = transform(img)
        # 多次翻转应产生不同结果（概率性）
        # 只是确认增强管道不崩溃
        assert t1.shape == (3, 100, 100)
