"""Shared memory stores with lazy initialization.

Short-term memory uses an in-process sliding window plus Redis summary.
Long-term memory uses Milvus vectors for cross-session durable facts.
"""

from typing import Any

from memory.chat_memory import ChatMemory

chat_memory = ChatMemory()


class _LazyMilvus:
    """MilvusMemory 惰性代理：首次属性访问时初始化。"""

    def __init__(self, collection_name: str) -> None:
        self._collection_name = collection_name
        self._instance: Any = None

    def _get(self) -> Any:
        if self._instance is None:
            from memory.vector_memory import MilvusMemory

            self._instance = MilvusMemory(self._collection_name)
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)


class _LazyRedis:
    """RedisKV 惰性代理。"""

    def __init__(self) -> None:
        self._instance: Any = None

    def _get(self) -> Any:
        if self._instance is None:
            from memory.vector_memory import RedisKV

            self._instance = RedisKV()
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)


vector_memory = _LazyMilvus("knowledge")
agent_memory = _LazyMilvus("agent_memory")
redis_kv = _LazyRedis()
