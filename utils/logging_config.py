#!/usr/bin/env python3
"""
统一日志配置：控制台 + 文件双输出，带时间戳
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    log_file: str | None = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    配置日志系统

    Args:
        log_file: 日志文件路径，None 则只输出到控制台
        level: 日志级别
        console: 是否输出到控制台

    Returns:
        配置完成的 root logger
    """
    logger = logging.getLogger("pneumonia")
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# 快捷获取 logger
def get_logger(name: str = "pneumonia") -> logging.Logger:
    return logging.getLogger(name)
