"""FileTool 测试用例（使用真实临时文件）。"""

import pytest

from tools.file_tool import FileTool
from utils.exception import ToolException


class TestFileTool:
    def setup_method(self) -> None:
        self._tool = FileTool()

    def test_spec(self) -> None:
        spec = self._tool.spec()
        assert spec.name == "file"
        assert "file_path" in spec.input_schema

    def test_run_with_txt(self, sample_txt) -> None:
        result = self._tool.run({"file_path": str(sample_txt)})
        assert "Hello AgentOffice TXT Test" in result.content
        assert result.metadata["file_suffix"] == "txt"
        assert result.metadata["text_length"] > 0

    def test_run_with_pdf(self, sample_pdf) -> None:
        result = self._tool.run({"file_path": str(sample_pdf)})
        assert "Hello AgentOffice PDF Test" in result.content
        assert result.metadata["file_suffix"] == "pdf"

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(ToolException, match="文件不存在"):
            self._tool.run({"file_path": "/nonexistent/path/file.pdf"})

    def test_unsupported_suffix_raises(self, tmp_path) -> None:
        png_file = tmp_path / "test.png"
        png_file.write_text("fake png content")
        with pytest.raises(ToolException, match="仅支持PDF、TXT、DOCX"):
            self._tool.run({"file_path": str(png_file)})

    def test_text_metadata(self, sample_txt) -> None:
        result = self._tool.run({"file_path": str(sample_txt)})
        assert result.metadata["file_name"] == "test.txt"
        assert result.metadata["text_length"] == len("Hello AgentOffice TXT Test\n第二行内容")
