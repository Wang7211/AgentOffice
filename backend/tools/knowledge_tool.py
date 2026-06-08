"""知识库检索工具。"""

from typing import Any

from config.settings import get_settings
from memory.store import vector_memory
from tools.base import BaseTool
from tools.base import ToolResult
from utils.document_classifier import categories_compatible
from utils.document_classifier import classify_document
from utils.exception import ToolException


class KnowledgeTool(BaseTool):
    """本地知识库语义检索工具。"""

    name = "knowledge"
    description = "检索企业本地知识库，返回相似文档分片。"
    input_schema = {"query": "必填，知识库检索问题。"}

    required_permissions = frozenset({"knowledge:read"})
    context_schema = {"user_id": "upload_user_id"}

    def run(self, tool_input: dict[str, Any]) -> ToolResult:
        """检索知识库分片。

        参数:
            tool_input: 包含 `query` 和可选 `top_k` 的字典。

        返回:
            匹配到的知识分片。

        异常:
            ToolException: 查询为空时抛出。
        """
        query = str(tool_input.get("query", "")).strip()
        top_k = int(tool_input.get("top_k", 5))
        if not query:
            raise ToolException("知识库检索问题不能为空")
        settings = get_settings()
        query_category = classify_document("", query)
        upload_user_id = tool_input.get("upload_user_id") or tool_input.get("user_id")
        metadata_filter = (
            {"upload_user_id": int(upload_user_id)}
            if upload_user_id is not None
            else None
        )
        raw_results = vector_memory.search_filtered(
            query=query,
            top_k=top_k * 3,
            min_score=float(
                tool_input.get(
                    "min_score",
                    settings.knowledge_similarity_threshold,
                )
            ),
            metadata_filter=metadata_filter,
        )
        results, rejected_results = self._filter_by_category(
            raw_results,
            query_category,
        )
        results = results[:top_k]
        if not results:
            return ToolResult(
                content=(
                    "知识库没有找到达到相似度阈值且类别匹配的内容。"
                    "已过滤低相关或类别不匹配的分片。"
                    "请换一个更具体的问题、上传相关文档，或允许跳过知识库。"
                ),
                metadata={
                    "query": query,
                    "query_category": query_category,
                    "min_score": settings.knowledge_similarity_threshold,
                    "matches": [],
                    "filtered_count": len(raw_results),
                    "rejected": rejected_results[:5],
                },
            )
        content = self._format_results(results)
        return ToolResult(
            content=content,
            metadata={
                "query": query,
                "query_category": query_category,
                "matches": results,
                "rejected": rejected_results[:5],
            },
        )

    def _filter_by_category(
        self,
        results: list[dict[str, Any]],
        query_category: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """按查询类别过滤明显不匹配的文档。"""
        filtered_results: list[dict[str, Any]] = []
        rejected_results: list[dict[str, Any]] = []
        for item in results:
            metadata = dict(item.get("metadata") or {})
            document_category = str(metadata.get("document_category") or "")
            if not document_category:
                document_category = classify_document(
                    str(metadata.get("file_name") or ""),
                    str(item.get("text") or ""),
                )
                metadata["document_category"] = document_category
                item["metadata"] = metadata
            if categories_compatible(query_category, document_category):
                filtered_results.append(item)
                continue
            rejected_results.append(
                {
                    "file_name": str(metadata.get("file_name") or "未知文件"),
                    "document_category": document_category,
                    "score": float(item.get("score") or 0),
                    "reason": "document_category_mismatch",
                }
            )
        return filtered_results, rejected_results

    def _format_results(self, results: list[dict[str, Any]]) -> str:
        """将检索结果格式化为模型上下文。"""
        lines: list[str] = []
        for index, item in enumerate(results, start=1):
            metadata = item.get("metadata", {})
            file_name = metadata.get("file_name", "未知文件")
            document_category = metadata.get("document_category", "general")
            score = item.get("score", 0)
            text = str(item.get("text", "")).strip()
            prefix = (
                f"[{index}] {file_name} 类别={document_category} 相似度={score:.3f}"
            )
            if score < 0.3:
                text = f"[低置信度，仅供参考]\n{text}"
            lines.append(f"{prefix}\n{text}")
        return "\n\n".join(lines)
