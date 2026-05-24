"""结构化日志辅助工具。"""

import json
from typing import Any

from loguru import logger


def preview_text(value: object, max_length: int = 200) -> str:
    """生成适合日志展示的短文本。

    参数:
        value: 任意待展示对象。
        max_length: 最大保留字符数。
    返回:
        去除换行并截断后的文本。
    异常:
        无。
    """
    text = str(value or "").replace("\n", "\\n").replace("\r", "\\r")
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}..."


def log_agent_event(event: str, **fields: Any) -> None:
    """输出 Agent 链路结构化日志。

    参数:
        event: 事件名称。
        fields: 事件字段。
    返回:
        无。
    异常:
        无。
    """
    payload = {"event": event, **fields}
    logger.info(
        "AGENT_EVENT {}",
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True),
    )
