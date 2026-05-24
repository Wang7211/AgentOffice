"""Loguru 日志配置。"""

import sys
from pathlib import Path

from loguru import logger


def configure_logger() -> None:
    """配置控制台日志和按日归档文件日志。

    返回:
        无。

    异常:
        OSError: 日志目录创建失败时抛出。
    """
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | "
            "{file}:{line} | {message}"
        ),
    )
    logger.add(
        "logs/agent_{time:YYYYMMDD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | "
            "{file}:{line} | {message}"
        ),
    )
