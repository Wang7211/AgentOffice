"""基础工具协议与注册表。"""

import json
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable

from utils.exception import ToolException


@dataclass(frozen=True)
class ToolSpec:
    """可执行工具的公开元数据。"""

    name: str
    description: str
    input_schema: dict[str, str]
    required_permissions: tuple[str, ...] = ()
    context_schema: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """标准工具执行结果。"""

    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ToolExecutionContext:
    """Runtime context supplied by the agent executor for tool calls."""

    user_id: int
    session_id: str
    permissions: frozenset[str] = field(default_factory=frozenset)
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_permissions(self, required_permissions: set[str] | frozenset[str]) -> bool:
        return set(required_permissions).issubset(self.permissions)


class BaseTool(ABC):
    """Agent 工具抽象基类。"""

    name: str
    description: str
    input_schema: dict[str, str]
    required_permissions: frozenset[str] = frozenset()
    context_schema: dict[str, str] = {}

    @abstractmethod
    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        """执行工具。

        参数:
            tool_input: 已校验的工具入参。

        返回:
            标准工具结果。

        异常:
            ToolException: 工具执行失败时抛出。
        """

    def run_with_context(
        self,
        tool_input: dict[str, Any],
        context: ToolExecutionContext | None,
    ) -> ToolResult:
        """Run a tool after permission checks and context injection."""
        self._validate_permissions(context)
        scoped_input = self._inject_context(dict(tool_input), context)
        return self.run(scoped_input)

    def _validate_permissions(self, context: ToolExecutionContext | None) -> None:
        if not self.required_permissions:
            return
        if context is None:
            raise ToolException(f"Tool {self.name} requires execution context")
        missing = sorted(self.required_permissions - context.permissions)
        if missing:
            raise ToolException(
                f"Tool {self.name} missing permissions: {', '.join(missing)}"
            )

    def _inject_context(
        self,
        tool_input: dict[str, Any],
        context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        if context is None:
            return tool_input
        for context_field, input_key in self.context_schema.items():
            if context_field == "user_id":
                tool_input[input_key] = context.user_id
            elif context_field == "session_id":
                tool_input[input_key] = context.session_id
            elif context_field in context.metadata:
                tool_input[input_key] = context.metadata[context_field]
        return tool_input

    def spec(self) -> ToolSpec:
        """返回工具公开元数据。

        返回:
            工具规格信息。

        异常:
            无。
        """
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema,
            required_permissions=tuple(sorted(self.required_permissions)),
            context_schema=dict(self.context_schema),
        )


class ToolRegistry:
    """保存全部已启用工具的内存注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """按工具名称注册工具。

        参数:
            tool: 工具实例。

        返回:
            无。

        异常:
            ToolException: 工具名称为空或重复时抛出。
        """
        if not tool.name:
            raise ToolException("工具名称不能为空")
        if tool.name in self._tools:
            raise ToolException(f"工具已存在：{tool.name}")
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> BaseTool:
        """获取已注册工具。

        参数:
            tool_name: 工具名称。

        返回:
            工具实例。

        异常:
            ToolException: 工具未注册时抛出。
        """
        tool = self._tools.get(tool_name)
        if not tool:
            raise ToolException(f"工具不存在：{tool_name}")
        return tool

    def list_specs(self) -> list[ToolSpec]:
        """列出全部工具的公开规格。

        返回:
            工具规格列表。

        异常:
            无。
        """
        return [tool.spec() for tool in self._tools.values()]

    def list_tools(self) -> list[BaseTool]:
        """列出全部内部工具实例。"""
        return list(self._tools.values())

    def list_langchain_tools(self) -> list[Any]:
        """将内部工具适配为 LangChain StructuredTool。

        返回:
            LangChain 工具列表。未安装 LangChain 时返回空列表。

        异常:
            无。适配失败的工具会被跳过。
        """
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            return []

        langchain_tools: list[Any] = []
        for tool in self._tools.values():
            runner = _build_langchain_runner(tool)
            langchain_tools.append(
                StructuredTool.from_function(
                    func=runner,
                    name=tool.name,
                    description=tool.description,
                    args_schema=_build_langchain_args_schema(tool),
                )
            )
        return langchain_tools

    def get_langchain_tool(self, tool_name: str) -> Any | None:
        """按名称返回 LangChain StructuredTool 适配器。"""
        tool = self._tools.get(tool_name)
        if not tool:
            return None
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            return None
        return StructuredTool.from_function(
            func=_build_langchain_runner(tool),
            name=tool.name,
            description=tool.description,
            args_schema=_build_langchain_args_schema(tool),
        )


def _build_langchain_runner(tool: BaseTool) -> Callable[..., str]:
    """创建 LangChain 可调用函数。"""

    def _runner(**kwargs: Any) -> str:
        if len(kwargs) == 1 and isinstance(kwargs.get("tool_input"), dict):
            tool_input = dict(kwargs["tool_input"])
        else:
            tool_input = {
                key: value
                for key, value in kwargs.items()
                if value is not None
            }
        result = tool.run(tool_input)
        return json.dumps(
            {"content": result.content, "metadata": result.metadata},
            ensure_ascii=False,
            default=str,
        )

    _runner.__name__ = f"run_{tool.name}"
    return _runner


def _build_langchain_args_schema(tool: BaseTool) -> Any:
    """Build a LangChain/Pydantic args schema from the internal tool schema."""
    try:
        from pydantic import Field
        from pydantic import create_model
    except ImportError:
        return None

    fields: dict[str, tuple[Any, Any]] = {}
    for field_name, description in tool.input_schema.items():
        normalized_name = str(field_name).strip()
        if not normalized_name or not normalized_name.isidentifier():
            continue
        fields[normalized_name] = (
            Any,
            Field(default=None, description=str(description)),
        )

    if not fields:
        fields["tool_input"] = (
            dict[str, Any],
            Field(default_factory=dict, description="Tool input parameters"),
        )

    model_name = "".join(part.capitalize() for part in tool.name.split("_"))
    return create_model(f"{model_name or 'InternalTool'}Input", **fields)
