"""PDF、TXT 和 DOCX 文件解析工具。"""

from pathlib import Path
from typing import Any

import fitz
from docx import Document

from tools.base import BaseTool
from tools.base import ToolResult
from utils.common import normalize_text
from utils.exception import ToolException


class FileTool(BaseTool):
    """从办公文档中抽取文本的工具。"""

    name = "file"
    description = "解析PDF、TXT、DOCX文件文本内容。"
    input_schema = {"file_path": "必填，本地文件路径。"}
    allowed_suffixes = {".pdf", ".txt", ".docx"}
    required_permissions = frozenset({"file:read"})

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        """从本地文件中抽取规范化文本。

        参数:
            tool_input: 包含 `file_path` 的字典。

        返回:
            抽取文本和文件元数据。

        异常:
            ToolException: 文件类型不支持或读取失败时抛出。
        """
        file_path = Path(str(tool_input.get("file_path", ""))).resolve()
        if not file_path.exists() or not file_path.is_file():
            raise ToolException("文件不存在")
        suffix = file_path.suffix.lower()
        if suffix not in self.allowed_suffixes:
            raise ToolException("仅支持PDF、TXT、DOCX文件")
        content = self.extract_text(file_path)
        return ToolResult(
            content=content,
            metadata={
                "file_name": file_path.name,
                "file_suffix": suffix.replace(".", ""),
                "text_length": len(content),
            },
        )

    def extract_text(self, file_path: Path) -> str:
        """从支持的文件中抽取文本。

        参数:
            file_path: 本地文件路径。

        返回:
            规范化文本。

        异常:
            ToolException: 文本抽取失败时抛出。
        """
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(file_path)
        if suffix == ".txt":
            return normalize_text(file_path.read_text(encoding="utf-8"))
        if suffix == ".docx":
            return self._extract_docx(file_path)
        raise ToolException("不支持的文件类型")

    def _extract_pdf(self, file_path: Path) -> str:
        """从 PDF 中抽取文本。"""
        try:
            document = fitz.open(file_path)
            page_text = [page.get_text("text") for page in document]
        except RuntimeError as exc:
            raise ToolException("PDF解析失败") from exc
        return normalize_text("\n".join(page_text))

    def _extract_docx(self, file_path: Path) -> str:
        """从 DOCX 中抽取文本。"""
        try:
            document = Document(str(file_path))
        except Exception as exc:
            raise ToolException("DOCX解析失败") from exc
        paragraph_text = [paragraph.text for paragraph in document.paragraphs]
        return normalize_text("\n".join(paragraph_text))
