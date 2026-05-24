"""LocalVectorMemory 本地向量存储测试。"""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from memory.vector_memory import LocalVectorMemory


class TestLocalVectorMemory:
    def test_add_and_search(self, temp_data_dir: Path) -> None:
        mem = LocalVectorMemory("test_index.json")
        mem.add_text(
            vector_id="v1",
            text="Python 是一种编程语言",
            metadata={"source": "manual"},
        )
        mem.add_text(
            vector_id="v2",
            text="Java 也是一种编程语言",
            metadata={"source": "manual"},
        )
        mem.add_text(
            vector_id="v3",
            text="今天的天气非常好",
            metadata={"source": "manual"},
        )
        results = mem.search("编程语言", top_k=2)
        assert len(results) == 2
        assert results[0]["score"] >= 0.0
        # "编程语言" 应比 "天气" 更匹配
        v1_score = next(r["score"] for r in results if r["vector_id"] == "v1")
        v3 = next((r for r in results if r["vector_id"] == "v3"), None)
        assert v1_score >= (v3["score"] if v3 else 0.0)

    def test_search_filtered_with_min_score(self, temp_data_dir: Path) -> None:
        mem = LocalVectorMemory("test_min_score.json")
        mem.add_text("v1", "苹果是一种水果", {})
        mem.add_text("v2", "苹果公司发布新款手机", {})
        results = mem.search_filtered("苹果", top_k=5, min_score=0.5)
        assert all(r["score"] >= 0.5 for r in results)

    def test_search_filtered_with_metadata_filter(self, temp_data_dir: Path) -> None:
        mem = LocalVectorMemory("test_metadata.json")
        mem.add_text("v1", "合规文档内容", {"category": "compliance"})
        mem.add_text("v2", "普通文档内容", {"category": "general"})
        results = mem.search_filtered(
            "文档", top_k=5, metadata_filter={"category": "compliance"}
        )
        assert all(r["metadata"]["category"] == "compliance" for r in results)

    def test_delete(self, temp_data_dir: Path) -> None:
        mem = LocalVectorMemory("test_delete.json")
        mem.add_text("v1", "内容1", {})
        mem.add_text("v2", "内容2", {})
        assert len(mem.search("内容", top_k=10)) == 2
        mem.delete(["v1"])
        results = mem.search("内容", top_k=10)
        assert len(results) == 1
        assert results[0]["vector_id"] == "v2"

    def test_empty_index_returns_empty(self, temp_data_dir: Path) -> None:
        mem = LocalVectorMemory("empty.json")
        assert mem.search("任何内容") == []

    def test_empty_text_returns_zero_vector(self) -> None:
        mem = LocalVectorMemory("_dummy.json")
        vector = mem._embed_text("")
        assert all(v == 0.0 for v in vector)

    def test_embed_text_normalized(self) -> None:
        mem = LocalVectorMemory("_dummy2.json")
        v1 = mem._embed_text("hello world")
        v2 = mem._embed_text("HELLO WORLD")
        # 相同文本因小写化应产生相同向量
        assert v1 == v2

    def test_cosine_similarity_identical(self) -> None:
        mem = LocalVectorMemory("_dummy3.json")
        vector = [0.6, 0.8]
        similarity = mem._cosine_similarity(vector, vector)
        assert abs(similarity - 1.0) < 1e-10

    def test_cosine_similarity_orthogonal(self) -> None:
        mem = LocalVectorMemory("_dummy4.json")
        similarity = mem._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(similarity) < 1e-10

    def test_tokenize_cjk_generates_unigram_bigram(self) -> None:
        mem = LocalVectorMemory("_dummy5.json")
        tokens = mem._tokenize("你好世界")
        assert "你" in tokens
        assert "好" in tokens
        assert "你好" in tokens
        assert "好世" in tokens
        assert len(tokens) == 7  # 4 unigram + 3 bigram

    def test_tokenize_mixed_content(self) -> None:
        mem = LocalVectorMemory("_dummy6.json")
        tokens = mem._tokenize("hello 你好 world")
        assert "hello" in tokens
        assert "world" in tokens
        assert "你" in tokens
        assert "你好" in tokens

    def test_normalize_vector(self) -> None:
        mem = LocalVectorMemory("_dummy7.json")
        result = mem._normalize_vector([3.0, 4.0])
        assert abs(result[0] - 0.6) < 1e-10
        assert abs(result[1] - 0.8) < 1e-10

    def test_normalize_zero_vector(self) -> None:
        mem = LocalVectorMemory("_dummy8.json")
        result = mem._normalize_vector([0.0, 0.0])
        assert result == [0.0, 0.0]
