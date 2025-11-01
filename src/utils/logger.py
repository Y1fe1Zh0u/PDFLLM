"""统一的日志系统

提供一致的日志格式和配置，替代print语句
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """设置日志器

    Args:
        name: 日志器名称（通常使用 __name__）
        level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        log_file: 日志文件路径（可选）
        console: 是否输出到控制台

    Returns:
        配置好的日志器

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("开始处理PDF文件")
        >>> logger.error("处理失败", exc_info=True)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """获取已配置的日志器

    如果日志器不存在，则使用默认配置创建

    Args:
        name: 日志器名称

    Returns:
        日志器实例
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# 为常用模块预配置日志器
def setup_script_logger(script_name: str, log_dir: str = "logs") -> logging.Logger:
    """为脚本设置日志器

    Args:
        script_name: 脚本名称（如 "extraction", "merge"）
        log_dir: 日志目录

    Returns:
        配置好的日志器
    """
    from datetime import datetime

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True)

    log_file = log_dir_path / f"{script_name}_{datetime.now().strftime('%Y%m%d')}.log"

    return setup_logger(
        name=script_name,
        level=logging.INFO,
        log_file=str(log_file),
        console=True
    )
