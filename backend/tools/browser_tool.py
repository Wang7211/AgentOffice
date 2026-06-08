"""网页浏览与读取工具。"""

from html.parser import HTMLParser
from typing import Any

import httpx

from config.settings import get_settings
from tools.base import BaseTool
from tools.base import ToolResult
from utils.exception import ToolException


class _TextExtractor(HTMLParser):
    """从 HTML 中抽取可读文本。"""

    ignored_tags = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.ignored_tags:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self.ignored_tags and self._ignored_depth > 0:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        text = " ".join(data.split())
        if text:
            self._parts.append(text)

    def text(self) -> str:
        """返回合并后的正文文本。"""
        return "\n".join(self._parts)


class BrowserTool(BaseTool):
    """读取网页正文的浏览器工具。"""

    name = "browser"
    description = "打开并读取网页内容；优先使用 Playwright 渲染，未安装时降级为 HTTP HTML 抽取。"
    input_schema = {
        "url": "必填，待打开的网页 URL。",
        "action": "可选，默认 read，当前支持 read。",
    }

    required_permissions = frozenset({"network:read"})

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        """读取指定网页内容。"""
        url = str(tool_input.get("url") or "").strip()
        action = str(tool_input.get("action") or "read").strip().lower()
        if not url:
            raise ToolException("浏览器工具需要明确的 url 参数")
        if not url.startswith(("http://", "https://")):
            raise ToolException("浏览器工具仅支持 http 或 https URL")
        if action not in {"read", "open"}:
            raise ToolException("浏览器工具当前仅支持 read/open 操作")
        try:
            return self._read_with_playwright(url)
        except ImportError:
            return self._read_with_http(url)
        except Exception as exc:
            fallback_result = self._read_with_http(url)
            return ToolResult(
                content=fallback_result.content,
                metadata={
                    **fallback_result.metadata,
                    "browser_engine": "http_fallback",
                    "playwright_error": str(exc),
                },
            )

    def _read_with_playwright(self, url: str) -> ToolResult:
        """使用 Playwright 渲染并读取网页正文。"""
        from playwright.sync_api import sync_playwright

        settings = get_settings()
        timeout_ms = int(settings.request_timeout * 1000)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            title = page.title()
            body_text = page.locator("body").inner_text(timeout=timeout_ms)
            browser.close()
        content = self._format_content(title=title, text=body_text)
        return ToolResult(
            content=content,
            metadata={"url": url, "title": title, "browser_engine": "playwright"},
        )

    def _read_with_http(self, url: str) -> ToolResult:
        """使用 HTTP 抓取网页并抽取正文。"""
        settings = get_settings()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AgentOfficeBrowser/1.0"
            )
        }
        try:
            with httpx.Client(
                timeout=settings.request_timeout,
                follow_redirects=True,
                headers=headers,
            ) as client:
                response = client.get(url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolException(f"网页读取失败：{exc}") from exc
        extractor = _TextExtractor()
        extractor.feed(response.text)
        content = self._format_content(title=url, text=extractor.text())
        return ToolResult(
            content=content,
            metadata={"url": url, "browser_engine": "http", "status_code": response.status_code},
        )

    def _format_content(self, title: str, text: str) -> str:
        """格式化网页读取结果。"""
        cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if len(cleaned_text) > 5000:
            cleaned_text = f"{cleaned_text[:5000]}\n...(已截断)"
        return f"页面标题：{title}\n\n页面正文：\n{cleaned_text or '未提取到可读正文'}"
