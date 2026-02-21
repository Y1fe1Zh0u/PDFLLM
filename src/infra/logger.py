"""统一日志模块

替代print，提供一致的日志格式和配置。
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: str | None = None,
) -> logging.Logger:
    """设置日志器

    Args:
        name: 日志器名称（通常使用 __name__）
        level: 日志级别
        log_file: 日志文件路径（可选）

    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """获取日志器，不存在则用默认配置创建"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
