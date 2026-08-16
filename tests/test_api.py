#!/usr/bin/env python3
"""API 端点测试（需要运行中的服务或使用 TestClient）"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestAPISchemas:
    """测试 Pydantic 模型"""

    def test_prediction_response_creation(self):
        from api.schemas import PredictionResponse
        resp = PredictionResponse(
            id=1,
            timestamp="2026-01-01T00:00:00",
            model_name="resnet18",
            prediction="肺炎 (PNEUMONIA)",
            probability_normal=0.15,
            probability_pneumonia=0.85,
            uncertainty_std=0.03,
            uncertain=False,
            image_hash="abc123",
        )
        assert resp.id == 1
        assert resp.probability_pneumonia == 0.85
        assert not resp.uncertain

    def test_health_response(self):
        from api.schemas import HealthResponse
        resp = HealthResponse(
            status="healthy",
            model_loaded=True,
            device="mps",
            timestamp="2026-01-01T00:00:00",
        )
        assert resp.status == "healthy"
        assert resp.model_loaded

    def test_model_info_response(self):
        from api.schemas import ModelInfoResponse
        resp = ModelInfoResponse(
            model_name="resnet18",
            device="cpu",
            class_names=["NORMAL", "PNEUMONIA"],
            available_models=["resnet18", "efficientnet_b0"],
        )
        assert "resnet18" in resp.available_models
        assert len(resp.class_names) == 2


class TestDatabase:
    """测试数据库操作"""

    def test_init_and_insert(self):
        from api.database import get_history, get_prediction, init_db, insert_prediction

        init_db()
        record_id = insert_prediction({
            "image_path": "test.jpg",
            "image_hash": "test123",
            "model_name": "resnet18",
            "probability_normal": 0.2,
            "probability_pneumonia": 0.8,
            "prediction": "肺炎",
            "uncertainty_std": 0.04,
            "uncertain": False,
        })
        assert record_id > 0

        record = get_prediction(record_id)
        assert record is not None
        assert record["prediction"] == "肺炎"
        assert record["probability_pneumonia"] == 0.8

        history = get_history(limit=10)
        assert len(history) > 0

    def test_feedback(self):
        from api.database import (
            get_prediction,
            init_db,
            insert_prediction,
            update_feedback,
        )

        init_db()
        record_id = insert_prediction({
            "image_hash": "feedback_test",
            "model_name": "resnet18",
            "probability_normal": 0.9,
            "probability_pneumonia": 0.1,
            "prediction": "正常",
        })
        update_feedback(record_id, "诊断正确", "NORMAL")
        record = get_prediction(record_id)
        assert record["user_feedback"] == "诊断正确"
        assert record["ground_truth"] == "NORMAL"

    def test_statistics(self):
        from api.database import get_statistics
        stats = get_statistics()
        assert "total_predictions" in stats
        assert isinstance(stats["total_predictions"], int)

    def test_insert_with_none_uncertain(self):
        """关闭不确定性量化时 uncertain=None，插入不应报错"""
        from api.database import get_prediction, init_db, insert_prediction

        init_db()
        record_id = insert_prediction({
            "image_hash": "none_uncertain",
            "model_name": "resnet18",
            "probability_normal": 0.9,
            "probability_pneumonia": 0.1,
            "prediction": "NORMAL",
            "uncertainty_std": None,
            "uncertain": None,
        })
        record = get_prediction(record_id)
        assert record["uncertain"] is None

    def test_statistics_counts_english_labels(self):
        """预测标签存为英文 (PNEUMONIA/NORMAL) 时统计应正确"""
        from api.database import get_connection, get_statistics, insert_prediction

        # 清空表，保证本测试确定性
        conn = get_connection()
        conn.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()

        insert_prediction({
            "image_hash": "stat_pneu",
            "model_name": "resnet18",
            "probability_normal": 0.2,
            "probability_pneumonia": 0.8,
            "prediction": "PNEUMONIA",
            "uncertain": True,
        })
        insert_prediction({
            "image_hash": "stat_norm",
            "model_name": "resnet18",
            "probability_normal": 0.9,
            "probability_pneumonia": 0.1,
            "prediction": "NORMAL",
            "uncertain": False,
        })
        stats = get_statistics()
        assert stats["pneumonia_count"] == 1
        assert stats["normal_count"] == 1
        assert stats["uncertain_count"] == 1
