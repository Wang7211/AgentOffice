"""采用滑动窗口策略的短期聊天记忆。"""

from collections import defaultdict
from typing import DefaultDict

from config.settings import get_settings


class ChatMemory:
    """保存最近对话轮次的内存短期记忆。"""

    def __init__(self) -> None:
        self._messages: DefaultDict[str, list[dict[str, str]]] = defaultdict(list)

    def append(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息并执行滑动窗口截断。

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
        self._messages[session_id] = self._messages[session_id][-max_messages:]

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
