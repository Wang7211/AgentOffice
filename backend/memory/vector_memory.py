"""基于 Milvus + Redis 的混合记忆存储。

架构：
  - Milvus（Milvus Lite 本地文件 / 远程服务）
    存储非结构化文本的语义向量，负责相似性检索。
  - Redis
    存储用户 ID、会话 ID 等低延迟访问的元数据。
"""

import json
import math
from typing import Any

import redis as redis_py
from loguru import logger
from pymilvus import DataType
from pymilvus import MilvusClient

from config.settings import get_settings
from utils.common import text_hash

VECTOR_DIMENSION = 1024


class MilvusMemory:
    """基于 Milvus 的向量记忆，支持语义检索与元数据过滤。

    使用 Milvus Lite（本地文件）或连接远程 Milvus 服务。
    """

    def __init__(self, collection_name: str = "knowledge") -> None:
        settings = get_settings()
        self._collection_name = collection_name
        milvus_uri = (
            settings.milvus_uri
            or str(settings.vector_store_dir / "milvus.db")
        )
        self._client = MilvusClient(
            milvus_uri,
            grpc_options=[("grpc.keepalive_time_ms", 60000)],
        )
        self._ensure_collection()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def add_text(
        self,
        vector_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """新增或更新文本向量。"""
        vector = self._embed(text)
        self._client.insert(
            self._collection_name,
            {
                "id": vector_id,
                "vector": vector,
                "text": text,
                **metadata,
            },
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """检索最相似的文本分片。"""
        return self.search_filtered(query=query, top_k=top_k)

    def search_filtered(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """按相似度阈值和元数据过滤检索。"""
        query_vector = self._embed(query)
        filter_expr = self._build_filter_expr(metadata_filter) if metadata_filter else ""
        self._client.load_collection(self._collection_name)

        results = self._client.search(
            collection_name=self._collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "*"],
            filter=filter_expr,
        )

        parsed: list[dict[str, Any]] = []
        for result in results[0]:
            score = round(float(result["distance"]), 4)
            if score < min_score:
                continue
            entity = dict(result.get("entity", {}))
            parsed.append(
                {
                    "vector_id": result["id"],
                    "score": score,
                    "text": entity.pop("text", ""),
                    "metadata": entity,
                }
            )
        return parsed

    def delete(self, vector_ids: list[str]) -> None:
        """按标识删除向量。"""
        self._client.delete(self._collection_name, vector_ids)

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """确保 Milvus 集合存在，不存在则创建。"""
        if self._client.has_collection(self._collection_name):
            return
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )
        schema.add_field("id", DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        self._client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params,
        )

    # ------------------------------------------------------------------
    # 过滤表达式
    # ------------------------------------------------------------------

    def _build_filter_expr(self, metadata_filter: dict[str, Any]) -> str:
        """将元数据过滤字典转为 Milvus 过滤表达式。"""
        expr_parts: list[str] = []
        for key, val in metadata_filter.items():
            if isinstance(val, set | list | tuple):
                formatted = ", ".join(repr(v) for v in val)
                expr_parts.append(f"{key} in [{formatted}]")
            elif isinstance(val, str):
                expr_parts.append(f"{key} == {repr(val)}")
            else:
                expr_parts.append(f"{key} == {val}")
        return " and ".join(expr_parts)

    # ------------------------------------------------------------------
    # 本地哈希向量化
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """将文本转为确定性归一化向量。"""
        vector = [0.0] * VECTOR_DIMENSION
        normalized_text = text.lower()
        if not normalized_text:
            return vector
        for token in self._tokenize(normalized_text):
            index = int(text_hash(token), 16) % VECTOR_DIMENSION
            vector[index] += 1.0
        return self._normalize_vector(vector)

    def _tokenize(self, text: str) -> list[str]:
        """中英文文本切分，中文使用 unigram+bigram。"""
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
        """CJK 字符 unigram + bigram。"""
        if not chars:
            return []
        result = list(chars)
        for i in range(len(chars) - 1):
            result.append(chars[i] + chars[i + 1])
        return result

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        """L2 归一化。"""
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]


class RedisKV:
    """基于 Redis 的键值元数据存储。

    用于存储用户 ID、会话 ID 等低延迟访问的元数据。
    Redis 不可用时静默降级（各方法返回 None / False）。
    """

    def __init__(self) -> None:
        settings = get_settings()
        redis_url = settings.redis_url
        self._client: redis_py.Redis | None = (
            redis_py.from_url(redis_url) if redis_url else None
        )

    def _call(self, op_name: str, *args: Any, **kwargs: Any) -> Any:
        """执行 Redis 操作，连接异常时静默降级。"""
        if not self._client:
            return None
        try:
            method = getattr(self._client, op_name)
            return method(*args, **kwargs)
        except redis_py.exceptions.ConnectionError:
            logger.warning("[RedisKV] {} 失败：Redis 不可用", op_name)
            return None
        except redis_py.exceptions.RedisError as exc:
            logger.warning("[RedisKV] {} 失败：{}", op_name, exc)
            return None

    def get(self, key: str) -> Any:
        """读取键值。"""
        value = self._call("get", key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value.decode("utf-8") if isinstance(value, bytes) else value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """写入键值（可选 TTL 秒）。"""
        serialized = (
            json.dumps(value, ensure_ascii=False)
            if not isinstance(value, (str, bytes))
            else value
        )
        self._call("set", key, serialized, ex=ttl)

    def delete(self, key: str) -> None:
        """删除键。"""
        self._call("delete", key)

    def exists(self, key: str) -> bool:
        """检查键是否存在。"""
        result = self._call("exists", key)
        return bool(result) if result is not None else False
