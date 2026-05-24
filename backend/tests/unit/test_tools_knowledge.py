"""KnowledgeTool 测试用例（mock 向量记忆）。"""

from typing import Any
from unittest import mock

import pytest

from tools.knowledge_tool import KnowledgeTool
from utils.exception import ToolException


def _mock_compliance_results() -> list[dict[str, Any]]:
    """模拟知识库搜索返回合规类结果。"""
    return [
        {
            "vector_id": "vec_001",
            "score": 0.45,
            "text": "企业碳足迹管理包括范围一、范围二和范围三排放。",
            "metadata": {"file_name": "碳足迹手册.pdf", "document_category": "compliance"},
        },
    ]


class TestKnowledgeTool:
    def setup_method(self) -> None:
        self._tool = KnowledgeTool()

    def test_spec(self) -> None:
        spec = self._tool.spec()
        assert spec.name == "knowledge"
        assert "query" in spec.input_schema

    @mock.patch("tools.knowledge_tool.vector_memory.search_filtered")
    def test_run_with_matches(self, mock_search) -> None:
        mock_search.return_value = _mock_compliance_results()
        result = self._tool.run({"query": "碳足迹管理规定"})
        assert "企业碳足迹管理" in result.content
        assert result.metadata["query_category"] == "compliance"

    @mock.patch("tools.knowledge_tool.vector_memory.search_filtered")
    def test_run_no_matches(self, mock_search) -> None:
        mock_search.return_value = []
        result = self._tool.run({"query": "不存在的查询"})
        assert "没有找到" in result.content
        assert result.metadata["matches"] == []

    def test_filter_by_category_rejects_incompatible(self) -> None:
        """测试类别不匹配的文档被过滤掉。"""
        results = [
            {
                "vector_id": "vec_001",
                "score": 0.5,
                "text": "旅游景点介绍",
                "metadata": {"file_name": "旅游.pdf", "document_category": "travel"},
            },
        ]
        filtered, rejected = self._tool._filter_by_category(results, "compliance")
        assert len(filtered) == 0
        assert len(rejected) == 1
        assert rejected[0]["reason"] == "document_category_mismatch"

    def test_filter_by_category_accepts_compatible(self) -> None:
        results = [
            {
                "vector_id": "vec_002",
                "score": 0.6,
                "text": "公司规章制度",
                "metadata": {"file_name": "制度.pdf", "document_category": "policy"},
            },
        ]
        filtered, rejected = self._tool._filter_by_category(results, "compliance")
        assert len(filtered) == 1
        assert len(rejected) == 0

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ToolException, match="不能为空"):
            self._tool.run({"query": ""})

    def test_format_results(self) -> None:
        results = [
            {
                "vector_id": "v1",
                "score": 0.85,
                "text": "测试内容",
                "metadata": {"file_name": "test.pdf", "document_category": "general"},
            },
        ]
        formatted = self._tool._format_results(results)
        assert "测试内容" in formatted
        assert "test.pdf" in formatted
