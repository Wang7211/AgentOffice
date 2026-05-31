"""采用滑动窗口策略的短期聊天记忆。"""

from collections import defaultdict
from typing import DefaultDict

from config.settings import get_settings


class ChatMemory:
    """保存最近对话轮次的内存短期记忆。"""

    def __init__(self) -> None:
        self._messages: DefaultDict[str, list[dict[str, str]]] = defaultdict(list)

    def append(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息并执行滑动窗口截断（截断部分自动压缩并缓存到 Redis）。

        参数:
            session_id: 聊天会话标识。
            role: 消息角色。
            content: 消息内容。

        返回:
            无。

        异常:
            无。
        """
        settings = get_settings()
        self._messages[session_id].append({"role": role, "content": content})
        max_messages = settings.chat_window_size * 2
        if len(self._messages[session_id]) > max_messages:
            truncated = self._messages[session_id][:-max_messages]
            self._update_summary(session_id, truncated)
        self._messages[session_id] = self._messages[session_id][-max_messages:]

    # ------------------------------------------------------------------
    # 上下文压缩：截断消息 → 提取摘要 → Redis 缓存
    # ------------------------------------------------------------------

    _SUMMARY_TTL = 86400  # 24 小时

    def _update_summary(self, session_id: str, truncated: list[dict[str, str]]) -> None:
        """将截断消息压缩后合并到会话的历史摘要中。"""
        from memory.store import redis_kv

        new_part = self._compress_messages(truncated)
        existing = redis_kv.get(f"summary:{session_id}") or ""
        combined = f"{existing}\n{new_part}" if existing else new_part
        redis_kv.set(f"summary:{session_id}", combined, ttl=self._SUMMARY_TTL)

    @staticmethod
    def _compress_messages(messages: list[dict[str, str]]) -> str:
        """提取式压缩消息列表为简短摘要（不调用 LLM）。"""
        user_requests: list[str] = []
        tools_used: set[str] = set()
        topics: set[str] = set()

        tool_keywords: dict[str, tuple[str, ...]] = {
            "weather": ("天气", "°C", "温度", "湿度", "风力", "晴", "雨", "雪", "预报"),
            "email": ("邮件已发送", "已发送邮件", "邮件发送成功", "发送给"),
            "code": ("计算结果", "计算", "="),
            "time": ("北京时间", "当前时间", "日期", "今天是"),
            "knowledge": ("知识库", "文档内容", "根据文档", "规章制度"),
            "browser": ("网页内容", "页面内容", "网页访问"),
        }

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user":
                user_requests.append(content[:80])
                # 提取话题关键词：2-4 字中文名词
                import re
                words = re.findall(r"[一-鿿]{2,4}", content)
                for w in words[:5]:
                    if w not in ("可以", "什么", "怎么", "这个", "那个", "一个", "没有", "不是", "就是", "如果", "因为", "所以", "然后", "之后", "但是", "还是"):
                        topics.add(w)
            elif role == "assistant":
                for tool, kws in tool_keywords.items():
                    if any(kw in content for kw in kws):
                        tools_used.add(tool)

        parts: list[str] = []
        if user_requests:
            parts.append(f"用户询问：{' | '.join(user_requests[:5])}")
        if tools_used:
            parts.append(f"使用工具：{', '.join(sorted(tools_used))}")
        if topics:
            parts.append(f"涉及话题：{', '.join(list(topics)[:8])}")
        return " | ".join(parts) if parts else "（历史对话）"

    def get_recent(self, session_id: str) -> list[dict[str, str]]:
        """返回指定会话的最近消息。

        参数:
            session_id: 聊天会话标识。

        返回:
            最近消息字典列表。

        异常:
            无。
        """
        return list(self._messages.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """清理指定会话的短期记忆。

        参数:
            session_id: 聊天会话标识。

        返回:
            无。

        异常:
            无。
        """
        self._messages.pop(session_id, None)
