"""MilvusMemory + RedisKV 存储层测试（mock 外部依赖）。"""

from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from memory.vector_memory import MilvusMemory
from memory.vector_memory import RedisKV


class TestMilvusMemory:
    """MilvusMemory 接口测试（mock MilvusClient）。"""

    @mock.patch("memory.vector_memory.MilvusClient")
    def test_add_text(self, mock_client_cls: mock.MagicMock) -> None:
        mem = MilvusMemory("test_coll")
        mem.add_text("v1", "你好世界", {"source": "manual"})
        call_kwargs = mock_client_cls.return_value.insert.call_args
        assert call_kwargs is not None
        args, _ = call_kwargs
        assert args[0] == "test_coll"
        data = args[1]
        assert data["id"] == "v1"
        assert data["text"] == "你好世界"
        assert data["source"] == "manual"
        assert "vector" in data

    @mock.patch("memory.vector_memory.MilvusClient")
    def test_search_delegates_to_search_filtered(
        self, mock_client_cls: mock.MagicMock
    ) -> None:
        mem = MilvusMemory("test_coll")
        mock_client_cls.return_value.search.return_value = [
            [
                {
                    "id": "v1",
                    "distance": 0.92,
                    "entity": {"text": "test", "source": "manual"},
                }
            ]
        ]
        results = mem.search("hello")
        assert len(results) == 1
        assert results[0]["vector_id"] == "v1"

    @mock.patch("memory.vector_memory.MilvusClient")
    def test_search_filtered_with_min_score(
        self, mock_client_cls: mock.MagicMock
    ) -> None:
        mem = MilvusMemory("test_coll")
        mock_client_cls.return_value.search.return_value = [
            [
                {
                    "id": "v1",
                    "distance": 0.92,
                    "entity": {"text": "high score", "cat": "a"},
                },
                {
                    "id": "v2",
                    "distance": 0.30,
                    "entity": {"text": "low score", "cat": "b"},
                },
            ]
        ]
        results = mem.search_filtered("query", min_score=0.5)
        assert len(results) == 1
        assert results[0]["vector_id"] == "v1"

    @mock.patch("memory.vector_memory.MilvusClient")
    def test_search_filtered_with_metadata(
        self, mock_client_cls: mock.MagicMock
    ) -> None:
        mem = MilvusMemory("test_coll")
        mock_client_cls.return_value.search.return_value = [
            [
                {
                    "id": "v1",
                    "distance": 0.85,
                    "entity": {"text": "doc", "category": "compliance"},
                }
            ]
        ]
        results = mem.search_filtered(
            "query", metadata_filter={"category": "compliance"}
        )
        # 验证 filter 表达式被传入
        call_args = mock_client_cls.return_value.search.call_args
        assert call_args is not None
        kwargs = call_args[1]
        assert "category" in kwargs.get("filter", "")

    @mock.patch("memory.vector_memory.MilvusClient")
    def test_delete(self, mock_client_cls: mock.MagicMock) -> None:
        mem = MilvusMemory("test_coll")
        mem.delete(["v1", "v2"])
        mock_client_cls.return_value.delete.assert_called_with(
            "test_coll", ["v1", "v2"]
        )

    @mock.patch("memory.vector_memory.MilvusClient")
    def test_empty_index_returns_empty(
        self, mock_client_cls: mock.MagicMock
    ) -> None:
        mem = MilvusMemory("empty_coll")
        mock_client_cls.return_value.search.return_value = [[]]
        assert mem.search("anything") == []

    def test_embed_identical_texts(self) -> None:
        """同一文本应产生相同向量。"""
        mem = MilvusMemory("_dummy")
        v1 = mem._embed("hello world")
        v2 = mem._embed("HELLO WORLD")
        assert v1 == v2

    def test_embed_empty_string(self) -> None:
        mem = MilvusMemory("_dummy")
        vector = mem._embed("")
        assert all(v == 0.0 for v in vector)

    def test_normalize_zero_vector(self) -> None:
        mem = MilvusMemory("_dummy")
        result = mem._normalize_vector([0.0, 0.0])
        assert result == [0.0, 0.0]

    def test_tokenize_cjk(self) -> None:
        mem = MilvusMemory("_dummy")
        tokens = mem._tokenize("你好世界")
        assert "你" in tokens
        assert "你好" in tokens
        assert "好世" in tokens
        assert len(tokens) == 7

    def test_tokenize_mixed(self) -> None:
        mem = MilvusMemory("_dummy")
        tokens = mem._tokenize("hello 你好 world")
        assert "hello" in tokens
        assert "你" in tokens

    def test_build_filter_expr_simple(self) -> None:
        mem = MilvusMemory("_dummy")
        expr = mem._build_filter_expr({"cat": "a"})
        assert 'cat ==' in expr
        assert "a" in expr

    def test_build_filter_expr_list(self) -> None:
        mem = MilvusMemory("_dummy")
        expr = mem._build_filter_expr({"status": ["ok", "pending"]})
        assert "in" in expr
        assert "ok" in expr or "'ok'" in expr or '"ok"' in expr


class TestRedisKV:
    """RedisKV 接口测试（mock Redis 不可用场景）。"""

    def test_redis_unavailable_returns_none(self) -> None:
        kv = RedisKV()
        # Redis 不可用时各方法静默降级
        assert kv.get("any_key") is None
        assert kv.exists("any_key") is False
        # set / delete 不应抛出
        kv.set("k", "v")
        kv.delete("k")
