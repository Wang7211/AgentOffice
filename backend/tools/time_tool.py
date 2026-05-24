"""时间辅助工具。"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

from tools.base import BaseTool
from tools.base import ToolResult


class TimeTool(BaseTool):
    """用于北京时间、时间戳和日期偏移计算的工具。"""

    name = "time"
    description = "获取北京时间、时间戳、日期偏移计算。"
    input_schema = {
        "offset_days": "可选，日期偏移天数，整数。",
    }

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        """返回北京时间和可选偏移日期。

        参数:
            tool_input: 可选的 `offset_days` 整数。

        返回:
            当前时间信息。

        异常:
            ValueError: 偏移值无法转换为整数时抛出。
        """
        offset_days = int(tool_input.get("offset_days", 0))
        beijing_timezone = timezone(timedelta(hours=8))
        current_time = datetime.now(beijing_timezone)
        target_date = current_time + timedelta(days=offset_days)
        content = (
            f"北京时间：{current_time:%Y-%m-%d %H:%M:%S}，"
            f"目标日期：{target_date:%Y-%m-%d}，"
            f"Unix时间戳：{int(current_time.timestamp())}"
        )
        return ToolResult(
            content=content,
            metadata={"offset_days": offset_days},
        )
