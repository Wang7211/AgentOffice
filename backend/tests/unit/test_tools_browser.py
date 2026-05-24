"""BrowserTool 测试用例（mock HTTP 请求）。"""

from unittest import mock

import httpx
import pytest

from tools.browser_tool import BrowserTool
from utils.exception import ToolException


class TestBrowserTool:
    def setup_method(self) -> None:
        self._tool = BrowserTool()

    def test_spec(self) -> None:
        spec = self._tool.spec()
        assert spec.name == "browser"
        assert "url" in spec.input_schema

    def test_empty_url_raises(self) -> None:
        with pytest.raises(ToolException, match="需要明确的 url 参数"):
            self._tool.run({"url": ""})

    def test_invalid_scheme_raises(self) -> None:
        with pytest.raises(ToolException, match="仅支持 http 或 https"):
            self._tool.run({"url": "ftp://example.com"})

    def test_unsupported_action_raises(self) -> None:
        with pytest.raises(ToolException, match="仅支持 read/open"):
            self._tool.run({"url": "https://example.com", "action": "write"})

    @mock.patch("tools.browser_tool.httpx.Client")
    def test_read_with_http_fallback(self, mock_client_class) -> None:
        """模拟 Playwright 不可用时的 HTTP 降级读取。"""
        mock_client_instance = mock_client_class.return_value.__enter__.return_value
        mock_request = mock.MagicMock()
        mock_response = httpx.Response(
            status_code=200,
            text="<html><body><p>Hello World</p></body></html>",
            request=mock_request,
        )
        mock_client_instance.get.return_value = mock_response

        result = self._tool.run({"url": "https://example.com"})
        assert "页面标题" in result.content
        assert result.metadata["browser_engine"] == "http"

    @mock.patch("tools.browser_tool.httpx.Client")
    def test_http_error_raises(self, mock_client_class) -> None:
        mock_client_instance = mock_client_class.return_value.__enter__.return_value
        mock_client_instance.get.side_effect = httpx.HTTPError("Connection failed")

        from config.settings import get_settings

        get_settings().request_timeout = 5.0

        with pytest.raises(ToolException, match="网页读取失败"):
            self._tool.run({"url": "https://example.com"})

    def test_truncates_long_content(self) -> None:
        """超过 5000 字符的内容应截断。"""
        long_text = "Hello\n" * 3000  # ~18000 字符
        result = self._tool._format_content(title="Test", text=long_text)
        assert "...(已截断)" in result
        assert len(result) < 5500  # 标题 + 截断后内容
