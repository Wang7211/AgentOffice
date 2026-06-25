"""Short-term session memory with a sliding window and compact summary."""

import re
from collections import defaultdict
from typing import Any
from typing import DefaultDict

from config.settings import get_settings


class ChatMemory:
    """Keep recent session messages for conversation continuity."""

    _SUMMARY_TTL = 86400
    _SUMMARY_KEY_PREFIX = "summary:"
    _TOPIC_STOPWORDS = frozenset(
        {
            "可以",
            "什么",
            "怎么",
            "这个",
            "那个",
            "一个",
            "没有",
            "不是",
            "就是",
            "如果",
            "因为",
            "所以",
            "然后",
            "之后",
            "但是",
            "还是",
        }
    )

    def __init__(self) -> None:
        self._messages: DefaultDict[str, list[dict[str, str]]] = defaultdict(list)
        self._observations: DefaultDict[str, list[dict[str, Any]]] = defaultdict(list)

    def append(self, session_id: str, role: str, content: str) -> None:
        """Append one message and compact overflow into the session summary."""
        settings = get_settings()
        self._messages[session_id].append({"role": role, "content": content})
        max_messages = settings.chat_window_size * 2
        if len(self._messages[session_id]) > max_messages:
            truncated = self._messages[session_id][:-max_messages]
            self._update_summary(session_id, truncated)
        self._messages[session_id] = self._messages[session_id][-max_messages:]

    def _update_summary(self, session_id: str, truncated: list[dict[str, str]]) -> None:
        """Merge compacted overflow messages into the session summary."""
        from memory.store import redis_kv

        new_part = self._compress_messages(truncated)
        existing = redis_kv.get(self._summary_key(session_id)) or ""
        combined = f"{existing}\n{new_part}" if existing else new_part
        redis_kv.set(self._summary_key(session_id), combined, ttl=self._SUMMARY_TTL)

    @staticmethod
    def _compress_messages(messages: list[dict[str, str]]) -> str:
        """Extract a compact session summary without recording tool details."""
        user_requests: list[str] = []
        assistant_points: list[str] = []
        topics: set[str] = set()

        for msg in messages:
            role = msg.get("role", "")
            content = re.sub(r"\s+", " ", msg.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                user_requests.append(content[:80])
                words = re.findall(r"[一-鿿]{2,4}", content)
                for w in words[:5]:
                    if w not in ChatMemory._TOPIC_STOPWORDS:
                        topics.add(w)
            elif role == "assistant":
                assistant_points.append(content[:100])

        parts: list[str] = []
        if user_requests:
            parts.append(f"用户诉求：{' | '.join(user_requests[:5])}")
        if assistant_points:
            parts.append(f"助理回应：{' | '.join(assistant_points[:3])}")
        if topics:
            parts.append(f"涉及话题：{', '.join(list(topics)[:8])}")
        return " | ".join(parts) if parts else "（历史对话）"

    def get_recent(self, session_id: str) -> list[dict[str, str]]:
        """Return recent raw messages for a session."""
        return list(self._messages.get(session_id, []))

    def append_observations(
        self,
        session_id: str,
        observations: list[dict[str, Any]],
    ) -> None:
        """Append reusable tool observations for follow-up references."""
        reusable = [
            dict(item)
            for item in observations
            if isinstance(item, dict)
            and item.get("type") == "tool_result"
            and item.get("status") == "success"
        ]
        if not reusable:
            return
        self._observations[session_id].extend(reusable)
        self._observations[session_id] = self._observations[session_id][-10:]

    def get_recent_observations(self, session_id: str) -> list[dict[str, Any]]:
        """Return recent reusable tool observations for a session."""
        return [dict(item) for item in self._observations.get(session_id, [])]

    def get_summary(self, session_id: str) -> str:
        """Return the compacted short-term summary for a session."""
        from memory.store import redis_kv

        return str(redis_kv.get(self._summary_key(session_id)) or "")

    def clear(self, session_id: str) -> None:
        """Clear recent messages and compacted summary for a session."""
        from memory.store import redis_kv

        self._messages.pop(session_id, None)
        self._observations.pop(session_id, None)
        redis_kv.delete(self._summary_key(session_id))

    @classmethod
    def _summary_key(cls, session_id: str) -> str:
        return f"{cls._SUMMARY_KEY_PREFIX}{session_id}"
