"""本地向量记忆，用作轻量 Milvus 兼容适配器。"""

import json
import math
from typing import Any

from config.settings import get_settings
from utils.common import text_hash


try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    class Embeddings:  # type: ignore[no-redef]
        """LangChain 未安装时的占位基类。"""


class LocalVectorMemory:
    """面向本地开发的 JSON 向量存储。"""

    vector_dimension = 1024

    def __init__(self, index_name: str = "knowledge_index.json") -> None:
        settings = get_settings()
        self._index_path = settings.vector_store_dir / index_name
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

    def add_text(
        self,
        vector_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """新增或更新一个文本向量。

        参数:
            vector_id: 稳定向量标识。
            text: 原始文本分片。
            metadata: 与分片关联的元数据。

        返回:
            无。

        异常:
            OSError: 索引文件无法写入时抛出。
        """
        index_data = self._load_index()
        index_data[vector_id] = {
            "text": text,
            "metadata": metadata,
            "vector": self._embed_text(text),
        }
        self._save_index(index_data)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """检索最相似的文本分片。

        参数:
            query: 查询文本。
            top_k: 最大返回分片数量。

        返回:
            按相似度排序的检索结果字典列表。

        异常:
            OSError: 索引文件无法读取时抛出。
        """
        return self.search_filtered(query=query, top_k=top_k)

    def search_filtered(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """按相似度阈值和元数据过滤检索结果。"""
        index_data = self._load_index()
        langchain_results = self._search_with_langchain_faiss(
            index_data=index_data,
            query=query,
            top_k=top_k,
            min_score=min_score,
            metadata_filter=metadata_filter,
        )
        if langchain_results is not None:
            return langchain_results
        return self._search_with_local_vectors(
            index_data=index_data,
            query=query,
            top_k=top_k,
            min_score=min_score,
            metadata_filter=metadata_filter,
        )

    def _search_with_local_vectors(
        self,
        index_data: dict[str, dict[str, Any]],
        query: str,
        top_k: int,
        min_score: float,
        metadata_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """使用本地 JSON 向量索引检索。"""
        query_vector = self._embed_text(query)
        scored_items = []
        for vector_id, item in index_data.items():
            metadata = dict(item.get("metadata") or {})
            if metadata_filter and not self._metadata_matches(metadata, metadata_filter):
                continue
            score = self._cosine_similarity(query_vector, item["vector"])
            if score < min_score:
                continue
            scored_items.append(
                {
                    "vector_id": vector_id,
                    "score": score,
                    "text": item["text"],
                    "metadata": metadata,
                }
            )
        scored_items.sort(key=lambda item: item["score"], reverse=True)
        return scored_items[:top_k]

    def _search_with_langchain_faiss(
        self,
        index_data: dict[str, dict[str, Any]],
        query: str,
        top_k: int,
        min_score: float,
        metadata_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]] | None:
        """使用 LangChain FAISS VectorStore 检索，依赖缺失时返回 None。"""
        if not index_data:
            return []
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document
        except ImportError:
            return None

        documents = [
            Document(
                page_content=str(item.get("text") or ""),
                metadata={
                    **dict(item.get("metadata") or {}),
                    "vector_id": vector_id,
                },
            )
            for vector_id, item in index_data.items()
        ]
        vector_store = FAISS.from_documents(
            documents,
            _LocalHashEmbeddings(self),
        )
        candidate_count = min(len(documents), max(top_k * 5, top_k))
        candidates = vector_store.similarity_search(query, k=candidate_count)
        query_vector = self._embed_text(query)
        scored_items: list[dict[str, Any]] = []
        for document in candidates:
            metadata = dict(document.metadata)
            vector_id = str(metadata.pop("vector_id", ""))
            if metadata_filter and not self._metadata_matches(metadata, metadata_filter):
                continue
            score = self._cosine_similarity(
                query_vector,
                self._embed_text(document.page_content),
            )
            if score < min_score:
                continue
            scored_items.append(
                {
                    "vector_id": vector_id,
                    "score": score,
                    "text": document.page_content,
                    "metadata": metadata,
                }
            )
        scored_items.sort(key=lambda item: item["score"], reverse=True)
        return scored_items[:top_k]

    def delete(self, vector_ids: list[str]) -> None:
        """按标识删除向量。

        参数:
            vector_ids: 待删除的向量标识列表。

        返回:
            无。

        异常:
            OSError: 索引文件无法写入时抛出。
        """
        index_data = self._load_index()
        for vector_id in vector_ids:
            index_data.pop(vector_id, None)
        self._save_index(index_data)

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """从磁盘加载向量索引。"""
        if not self._index_path.exists():
            return {}
        content = self._index_path.read_text(encoding="utf-8")
        if not content.strip():
            return {}
        return json.loads(content)

    def _save_index(self, index_data: dict[str, dict[str, Any]]) -> None:
        """将向量索引持久化到磁盘。"""
        self._index_path.write_text(
            json.dumps(index_data, ensure_ascii=False),
            encoding="utf-8",
        )

    def _metadata_matches(
        self,
        metadata: dict[str, Any],
        metadata_filter: dict[str, Any],
    ) -> bool:
        """判断元数据是否满足过滤条件。"""
        for key, expected_value in metadata_filter.items():
            actual_value = metadata.get(key)
            if isinstance(expected_value, set | list | tuple):
                if actual_value not in expected_value:
                    return False
            elif actual_value != expected_value:
                return False
        return True

    def _embed_text(self, text: str) -> list[float]:
        """创建确定性的稀疏哈希向量。"""
        vector = [0.0] * self.vector_dimension
        normalized_text = text.lower()
        if not normalized_text:
            return vector
        for token in self._tokenize(normalized_text):
            index = int(text_hash(token), 16) % self.vector_dimension
            vector[index] += 1.0
        return self._normalize_vector(vector)

    def _tokenize(self, text: str) -> list[str]:
        """将中英文文本切分为哈希词元，中文使用 unigram+bigram。"""
        tokens: list[str] = []
        current_word = ""
        cjk_chars: list[str] = []
        for char in text:
            if char.isascii() and char.isalnum():
                if cjk_chars:
                    tokens.extend(self._cjk_tokens(cjk_chars))
                    cjk_chars = []
                current_word += char
            elif char.strip():
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
                cjk_chars.append(char)
            else:
                if current_word:
                    tokens.append(current_word)
                    current_word = ""
                if cjk_chars:
                    tokens.extend(self._cjk_tokens(cjk_chars))
                    cjk_chars = []
        if current_word:
            tokens.append(current_word)
        if cjk_chars:
            tokens.extend(self._cjk_tokens(cjk_chars))
        return tokens

    def _cjk_tokens(self, chars: list[str]) -> list[str]:
        """为 CJK 字符生成 unigram + bigram 特征。"""
        if not chars:
            return []
        result = list(chars)
        for i in range(len(chars) - 1):
            result.append(chars[i] + chars[i + 1])
        return result

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        """将向量归一化为单位长度。"""
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _cosine_similarity(
        self,
        first_vector: list[float],
        second_vector: list[float],
    ) -> float:
        """计算归一化向量的余弦相似度。"""
        return sum(
            first_value * second_value
            for first_value, second_value in zip(first_vector, second_vector)
        )


class _LocalHashEmbeddings(Embeddings):
    """LangChain Embeddings 兼容适配器，复用本地确定性哈希向量。"""

    def __init__(self, vector_memory: LocalVectorMemory) -> None:
        self._vector_memory = vector_memory

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为文档列表生成向量。"""
        return [self._vector_memory._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """为查询生成向量。"""
        return self._vector_memory._embed_text(text)
