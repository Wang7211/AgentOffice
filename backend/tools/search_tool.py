"""带 Tavily 适配器的公网搜索工具。"""

from typing import Any

import httpx

from config.settings import get_settings
from tools.base import BaseTool
from tools.base import ToolResult
from utils.exception import ToolException


class SearchTool(BaseTool):
    """用于公开信息检索的工具。"""

    name = "search"
    description = "基于Tavily查询公开网络信息，需要配置TAVILY_API_KEY。"
    input_schema = {"query": "必填，搜索关键词。"}

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        """检索公网公开信息。

        参数:
            tool_input: 包含 `query` 的字典。

        返回:
            搜索摘要结果。

        异常:
            ToolException: 查询为空或 HTTP 请求失败时抛出。
        """
        query = str(tool_input.get("query", "")).strip()
        if not query:
            raise ToolException("搜索关键词不能为空")
        settings = get_settings()
        if not settings.tavily_api_key:
            return ToolResult(
                content="未配置TAVILY_API_KEY，无法执行联网搜索。",
                metadata={"query": query, "enabled": False},
            )
        return self._request_tavily(query, settings.tavily_api_key)

    def _request_tavily(self, query: str, api_key: str) -> ToolResult:
        """请求 Tavily 搜索接口。"""
        settings = get_settings()
        payload = {
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 5,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            with httpx.Client(timeout=settings.request_timeout) as client:
                response = client.post(
                    "https://api.tavily.com/search",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error_text = exc.response.text[:300]
            raise ToolException(f"联网搜索请求失败：{error_text}") from exc
        except httpx.HTTPError as exc:
            raise ToolException(f"联网搜索请求失败：{exc}") from exc
        response_data = response.json()
        answer = response_data.get("answer") or "未返回直接答案。"
        results = response_data.get("results", [])
        source_lines = [
            f"{item.get('title', '无标题')} - {item.get('url', '')}"
            for item in results
        ]
        content = "\n".join([answer, "来源：", *source_lines])
        return ToolResult(content=content, metadata={"query": query})
