#!/usr/bin/env python3
"""
SQLite 数据库操作 — 预测历史管理
"""

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_db_path() -> str:
    """获取数据库路径（可用环境变量 PNEUMONIA_DB_PATH 覆盖，便于测试隔离）"""
    override = os.environ.get("PNEUMONIA_DB_PATH")
    if override:
        db_path = Path(override)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return str(db_path)

    db_path = PROJECT_ROOT / "outputs" / "predictions.db"
    os.makedirs(db_path.parent, exist_ok=True)
    return str(db_path)


def get_connection() -> sqlite3.Connection:
    """获取数据库连接"""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表"""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT,
            image_hash TEXT,
            model_name TEXT NOT NULL,
            probability_normal REAL,
            probability_pneumonia REAL,
            prediction TEXT,
            uncertainty_std REAL,
            uncertain INTEGER,
            user_feedback TEXT,
            ground_truth TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_prediction(data: dict[str, Any]) -> int:
    """插入一条预测记录，返回 ID"""
    uncertain = data.get("uncertain")
    conn = get_connection()
    cursor = conn.execute("""
        INSERT INTO predictions
        (timestamp, image_path, image_hash, model_name,
         probability_normal, probability_pneumonia, prediction,
         uncertainty_std, uncertain)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("timestamp", datetime.now(UTC).isoformat()),
        data.get("image_path", ""),
        data.get("image_hash", ""),
        data.get("model_name", "unknown"),
        data.get("probability_normal"),
        data.get("probability_pneumonia"),
        data.get("prediction"),
        data.get("uncertainty_std"),
        int(uncertain) if uncertain is not None else None,
    ))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_history(limit: int = 50) -> list[dict]:
    """获取预测历史"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_prediction(pred_id: int) -> dict | None:
    """获取单条预测记录"""
    conn = get_connection()
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_feedback(pred_id: int, feedback: str, ground_truth: str | None = None):
    """更新用户反馈"""
    conn = get_connection()
    conn.execute(
        "UPDATE predictions SET user_feedback = ?, ground_truth = ? WHERE id = ?",
        (feedback, ground_truth, pred_id),
    )
    conn.commit()
    conn.close()


def get_statistics() -> dict:
    """获取统计信息"""
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    pneumonia_count = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction = 'PNEUMONIA' OR prediction LIKE '%肺炎%'"
    ).fetchone()[0]
    uncertain_count = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE uncertain = 1"
    ).fetchone()[0]
    conn.close()
    return {
        "total_predictions": total,
        "pneumonia_count": pneumonia_count,
        "uncertain_count": uncertain_count,
        "normal_count": total - pneumonia_count,
    }


# 启动时初始化
init_db()
