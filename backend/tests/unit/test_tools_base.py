"""ToolRegistry、BaseTool、ToolSpec、ToolResult 基础组件测试。"""

from typing import Any

import pytest

from tools.base import BaseTool
from tools.base import ToolRegistry
from tools.base import ToolResult
from tools.base import ToolSpec
from utils.exception import ToolException


# ---------------------------------------------------------------------------
# ToolSpec / ToolResult 值对象
# ---------------------------------------------------------------------------

class TestToolSpec:
    def test_frozen(self) -> None:
        spec = ToolSpec(name="test", description="测试工具", input_schema={"q": "str"})
        assert spec.name == "test"
        assert spec.description == "测试工具"
        assert spec.input_schema == {"q": "str"}
        with pytest.raises(AttributeError):
            spec.name = "changed"  # type: ignore[misc]


class TestToolResult:
    def test_value_object(self) -> None:
        result = ToolResult(content="hello", metadata={"key": "val"})
        assert result.content == "hello"
        assert result.metadata == {"key": "val"}


# ---------------------------------------------------------------------------
# BaseTool 抽象基类
# ---------------------------------------------------------------------------

class _ConcreteTool(BaseTool):
    name = "concrete"
    description = "具体工具"
    input_schema = {"x": "一个数字"}

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        x = tool_input.get("x", 0)
        return ToolResult(content=str(x * 2), metadata={"input": x})


class TestBaseTool:
    def test_spec_returns_metadata(self) -> None:
        tool = _ConcreteTool()
        spec = tool.spec()
        assert spec.name == "concrete"
        assert spec.description == "具体工具"
        assert spec.input_schema == {"x": "一个数字"}

    def test_concrete_tool_runs(self) -> None:
        tool = _ConcreteTool()
        result = tool.run({"x": 21})
        assert result.content == "42"
        assert result.metadata["input"] == 21


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        tool = _ConcreteTool()
        registry.register(tool)
        assert registry.get("concrete") is tool

    def test_register_empty_name_raises(self) -> None:
        registry = ToolRegistry()
        tool = _ConcreteTool()
        tool.name = ""
        with pytest.raises(ToolException, match="工具名称不能为空"):
            registry.register(tool)

    def test_register_duplicate_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(_ConcreteTool())
        with pytest.raises(ToolException, match="工具已存在"):
            registry.register(_ConcreteTool())

    def test_get_nonexistent_raises(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolException, match="工具不存在"):
            registry.get("nonexistent")

    def test_list_specs(self) -> None:
        registry = ToolRegistry()
        registry.register(_ConcreteTool())
        specs = registry.list_specs()
        assert len(specs) == 1
        assert specs[0].name == "concrete"

    def test_list_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(_ConcreteTool())
        tools = registry.list_tools()
        assert len(tools) == 1
        assert isinstance(tools[0], _ConcreteTool)

    def test_list_langchain_tools_empty_when_not_installed(self, monkeypatch) -> None:
        registry = ToolRegistry()
        registry.register(_ConcreteTool())
        # 模拟未安装 langchain-core
        import builtins

        original_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name.startswith("langchain"):
                raise ImportError
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        assert registry.list_langchain_tools() == []
