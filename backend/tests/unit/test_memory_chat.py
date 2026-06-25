"""ChatMemory 滑动窗口短期记忆测试。"""

from memory.chat_memory import ChatMemory
from config.settings import get_settings


class TestChatMemory:
    def setup_method(self) -> None:
        self._memory = ChatMemory()
        self._session_id = "test-session"

    def test_append_and_get_recent(self) -> None:
        self._memory.append(self._session_id, "user", "你好")
        self._memory.append(self._session_id, "assistant", "你好，有什么可以帮你？")
        messages = self._memory.get_recent(self._session_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "你好"
        assert messages[1]["role"] == "assistant"

    def test_get_recent_empty_session(self) -> None:
        messages = self._memory.get_recent("nonexistent")
        assert messages == []

    def test_sliding_window_truncation(self) -> None:
        """超过 chat_window_size * 2 条后应截断。"""
        settings = get_settings()
        max_messages = settings.chat_window_size * 2
        for i in range(max_messages + 10):
            self._memory.append(self._session_id, "user", f"msg_{i}")
            self._memory.append(self._session_id, "assistant", f"resp_{i}")
        messages = self._memory.get_recent(self._session_id)
        assert len(messages) <= max_messages

    def test_compressed_summary_does_not_record_tool_usage(self) -> None:
        summary = self._memory._compress_messages(
            [
                {"role": "user", "content": "请查询北京天气"},
                {"role": "assistant", "content": "天气工具返回北京 20 度"},
            ]
        )

        assert "用户诉求" in summary
        assert "使用工具" not in summary

    def test_sessions_isolated(self) -> None:
        self._memory.append("session-a", "user", "A的消息")
        self._memory.append("session-b", "user", "B的消息")
        assert len(self._memory.get_recent("session-a")) == 1
        assert len(self._memory.get_recent("session-b")) == 1
        assert self._memory.get_recent("session-a")[0]["content"] == "A的消息"

    def test_clear_session(self) -> None:
        self._memory.append(self._session_id, "user", "数据")
        self._memory.clear(self._session_id)
        assert self._memory.get_recent(self._session_id) == []

    def test_clear_nonexistent_session_does_not_raise(self) -> None:
        self._memory.clear("nonexistent")  # should not raise
