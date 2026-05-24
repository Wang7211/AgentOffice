"""MCP HTTP JSON-RPC 客户端。

职责边界：
- MCP 是 Model Context Protocol，是外部工具服务的通信协议，不是一个具体工具。
- 本文件属于 `integrations/` 协议集成层，只负责和配置的 MCP HTTP 网关通信。
- 这里不会注册 Agent 工具，也不会承载搜索、文件、时间等业务能力。
- `MCP_HTTP_ENDPOINT` 未配置时，`from_settings()` 返回 None，系统继续只使用内置 `tools/`。

调用关系：
Agent -> services/tool_service.py -> services/mcp_service.py
      -> integrations/mcp_client.py -> 外部 MCP Server
"""

from typing import Any

import httpx

from config.settings import get_settings
from utils.common import generate_uuid
from utils.exception import ToolException


class MCPHttpClient:
    """通过 HTTP JSON-RPC 调用 MCP Server 的客户端。"""

    def __init__(self, endpoint: str, api_key: str = "") -> None:
        self._endpoint = endpoint
        self._api_key = api_key

    @classmethod
    def from_settings(cls) -> "MCPHttpClient | None":
        """从配置创建客户端；未配置端点时返回 None。"""
        settings = get_settings()
        if not settings.mcp_http_endpoint:
            return None
        return cls(
            endpoint=settings.mcp_http_endpoint,
            api_key=settings.mcp_api_key,
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """列出 MCP Server 暴露的真实工具。"""
        result = self._request("tools/list", {})
        tools = result.get("tools", result if isinstance(result, list) else [])
        if not isinstance(tools, list):
            raise ToolException("MCP tools/list 返回格式不合法")
        return [tool for tool in tools if isinstance(tool, dict)]

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """调用 MCP Server 中的指定工具。"""
        return self._request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """发送 JSON-RPC 请求并返回 result。"""
        settings = get_settings()
        payload = {
            "jsonrpc": "2.0",
            "id": generate_uuid(),
            "method": method,
            "params": params,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            with httpx.Client(timeout=settings.request_timeout) as client:
                response = client.post(self._endpoint, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolException(f"MCP 网关请求失败：{exc}") from exc

        response_data = response.json()
        if response_data.get("error"):
            raise ToolException(f"MCP 网关返回错误：{response_data['error']}")
        result = response_data.get("result", {})
        if isinstance(result, dict):
            return result
        return {"content": result}
