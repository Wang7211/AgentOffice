"""MCP 工具发现与 Agent 工具适配服务。

职责边界：
- MCP 本身是协议；MCP Server 才会暴露一个或多个真实远程工具。
- 本文件属于 `services/` 服务层，负责发现远程 MCP 工具并包装为 Agent 可调用工具。
- 包装后的工具名会加 `mcp_` 前缀，例如远程 `calendar.create` 会暴露为
  `mcp_calendar_create`，避免和内置 `tools/` 中的工具重名。
- `tools/` 目录仍只放项目内置的具体能力工具；不要把 MCP 协议客户端放进 `tools/`。
- 未配置 `MCP_HTTP_ENDPOINT` 时不会发现任何远程工具，不影响内置工具运行。

调用关系：
services/tool_service.py -> discover_mcp_tools()
                       -> MCPHttpClient.tools/list
                       -> RemoteMCPTool.run()
                       -> MCPHttpClient.tools/call
"""

from typing import Any

from loguru import logger

from integrations.mcp_client import MCPHttpClient
from tools.base import BaseTool
from tools.base import ToolResult
from utils.exception import ToolException


class RemoteMCPTool(BaseTool):
    """MCP Server 暴露出的某一个真实工具的 Agent 适配器。"""

    def __init__(
        self,
        client: MCPHttpClient,
        exposed_name: str,
        remote_name: str,
        description: str,
        input_schema: dict[str, str],
    ) -> None:
        self._client = client
        self._remote_name = remote_name
        self.name = exposed_name
        self.description = description
        self.input_schema = input_schema
        self.required_permissions = frozenset({"mcp:call"})

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        """调用远端 MCP 工具。"""
        result = self._client.call_tool(self._remote_name, tool_input)
        return ToolResult(
            content=self._format_result(result),
            metadata={
                "protocol": "mcp",
                "remote_tool_name": self._remote_name,
                "exposed_tool_name": self.name,
            },
        )

    def _format_result(self, result: dict[str, Any]) -> str:
        """将 MCP result 规范化为 Agent 工具结果文本。"""
        content = result.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        if content is not None:
            return str(content)
        return str(result)


def discover_mcp_tools() -> list[RemoteMCPTool]:
    """从已配置的 MCP Server 发现并包装真实工具。"""
    client = MCPHttpClient.from_settings()
    if not client:
        return []
    try:
        remote_tools = client.list_tools()
    except ToolException as exc:
        logger.warning("MCP 工具发现失败：{}", exc.message)
        return []

    discovered_tools: list[RemoteMCPTool] = []
    used_names: set[str] = set()
    for remote_tool in remote_tools:
        remote_name = str(remote_tool.get("name") or "").strip()
        if not remote_name:
            continue
        exposed_name = _build_exposed_tool_name(remote_name)
        if exposed_name in used_names:
            logger.warning("跳过重复 MCP 工具名：{}", exposed_name)
            continue
        used_names.add(exposed_name)
        discovered_tools.append(
            RemoteMCPTool(
                client=client,
                exposed_name=exposed_name,
                remote_name=remote_name,
                description=_build_description(remote_tool),
                input_schema=_flatten_input_schema(remote_tool.get("inputSchema") or {}),
            )
        )
    return discovered_tools


def _build_exposed_tool_name(remote_name: str) -> str:
    """将远端工具名映射为本地安全工具名。"""
    normalized_name = "".join(
        character if character.isalnum() else "_"
        for character in remote_name.strip().lower()
    ).strip("_")
    return f"mcp_{normalized_name or 'remote_tool'}"


def _build_description(remote_tool: dict[str, Any]) -> str:
    """构造 Agent 可读的远端工具说明。"""
    remote_name = str(remote_tool.get("name") or "")
    description = str(remote_tool.get("description") or "MCP 远程工具")
    return f"MCP 远程工具 `{remote_name}`：{description}"


def _flatten_input_schema(input_schema: dict[str, Any]) -> dict[str, str]:
    """将 MCP JSON Schema 压平成 ToolSpec 需要的简单字段说明。"""
    properties = input_schema.get("properties") or {}
    required_fields = set(input_schema.get("required") or [])
    if not isinstance(properties, dict):
        return {}

    flattened_schema: dict[str, str] = {}
    for key, value in properties.items():
        if not isinstance(value, dict):
            flattened_schema[str(key)] = "参数。"
            continue
        field_type = value.get("type", "any")
        description = value.get("description", "参数。")
        required_text = "必填" if key in required_fields else "可选"
        flattened_schema[str(key)] = f"{required_text}，类型={field_type}，{description}"
    return flattened_schema
