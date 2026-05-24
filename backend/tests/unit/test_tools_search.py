"""SearchTool 测试用例（mock httpx 避免真实网络调用）。"""

from unittest import mock

import httpx
import pytest
from pytest import MonkeyPatch

from tools.search_tool import SearchTool
from utils.exception import ToolException


def _mock_http_response(status_code: int, **kwargs) -> httpx.Response:
    """创建带有 request 属性的 httpx.Response，避免 raise_for_status 报错。"""
    req = mock.MagicMock()
    return httpx.Response(status_code=status_code, request=req, **kwargs)


def _mock_tavily_success(*args, **kwargs) -> httpx.Response:
    """模拟 Tavily 搜索成功响应。"""
    return _mock_http_response(
        status_code=200,
        json={
            "answer": "Python 是一种高级编程语言。",
            "results": [
                {"title": "Python 官网", "url": "https://python.org"},
            ],
        },
    )


class TestSearchTool:
    def setup_method(self) -> None:
        self._tool = SearchTool()

    def test_spec(self) -> None:
        spec = self._tool.spec()
        assert spec.name == "search"
        assert "query" in spec.input_schema

    def test_run_without_api_key_returns_disabled_message(self) -> None:
        """未配置 API Key 时返回提示而非报错。"""
        result = self._tool.run({"query": "Python"})
        assert "未配置TAVILY_API_KEY" in result.content
        assert result.metadata["enabled"] is False

    @mock.patch("tools.search_tool.httpx.Client")
    def test_run_success(self, mock_client_class) -> None:
        """模拟 HTTP 调用成功场景。"""
        mock_client_instance = mock_client_class.return_value.__enter__.return_value
        mock_client_instance.post.return_value = _mock_tavily_success()

        from config.settings import get_settings

        settings = get_settings()
        settings.tavily_api_key = "test-key"

        result = self._tool.run({"query": "Python 是什么"})
        assert "Python 是一种高级编程语言" in result.content
        assert result.metadata["query"] == "Python 是什么"

    @mock.patch("tools.search_tool.httpx.Client")
    def test_http_error_raises_tool_exception(self, mock_client_class) -> None:
        """模拟 HTTP 错误时抛出 ToolException。"""
        mock_client_instance = mock_client_class.return_value.__enter__.return_value
        mock_client_instance.post.side_effect = httpx.HTTPStatusError(
            "401 Unauthorized",
            request=mock.MagicMock(),
            response=_mock_http_response(status_code=401, text="Unauthorized"),
        )

        from config.settings import get_settings

        settings = get_settings()
        settings.tavily_api_key = "bad-key"

        with pytest.raises(ToolException, match="联网搜索请求失败"):
            self._tool.run({"query": "Python"})

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ToolException, match="搜索关键词不能为空"):
            self._tool.run({"query": ""})
