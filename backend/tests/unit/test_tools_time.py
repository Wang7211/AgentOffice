"""TimeTool 测试用例。"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import pytest

from tools.time_tool import TimeTool


class TestTimeTool:
    def setup_method(self) -> None:
        self._tool = TimeTool()

    def test_spec(self) -> None:
        spec = self._tool.spec()
        assert spec.name == "time"
        assert "offset_days" in spec.input_schema

    def test_run_no_offset(self) -> None:
        before = datetime.now(timezone(timedelta(hours=8)))
        result = self._tool.run({})
        after = datetime.now(timezone(timedelta(hours=8)))
        # 验证结果包含北京时间
        assert "北京时间" in result.content
        assert "Unix时间戳" in result.content
        assert result.metadata["offset_days"] == 0

    def test_run_with_positive_offset(self) -> None:
        result = self._tool.run({"offset_days": "7"})
        assert "北京时间" in result.content
        assert result.metadata["offset_days"] == 7

    def test_run_with_negative_offset(self) -> None:
        result = self._tool.run({"offset_days": "-1"})
        assert "北京时间" in result.content
        assert result.metadata["offset_days"] == -1
