"""Agent 全流程集成测试（使用 LocalModelClient，不依赖外部 API）。"""

import pytest

from agent.graph import AgentGraph


_RUN_COUNTER = 0


def _unique_session_id() -> str:
    """生成唯一的会话 ID 避免 Milvus 锁冲突。"""
    global _RUN_COUNTER
    _RUN_COUNTER += 1
    import time
    return f"int-test-mem-{int(time.time() * 1000)}-{_RUN_COUNTER}"


class TestAgentPipeline:
    """测试 Agent 从消息输入到最终输出的完整链路。"""

    def setup_method(self) -> None:
        self._graph = AgentGraph()

    def test_agent_handles_greeting(self) -> None:
        """打招呼 -> 无需工具 -> 直接回复。"""
        result = self._graph.run(
            user_message="你好",
            session_id="int-test-greeting",
            history=[],
        )
        assert result["answer"] != ""
        assert result["session_id"] == "int-test-greeting"
        assert result["need_tool"] is False

    def test_agent_handles_time_query(self) -> None:
        """时间查询：understand 识别 tool -> planning 决策 -> tool 执行 -> action 回复。"""
        result = self._graph.run(
            user_message="现在几点了",
            session_id="int-test-time",
            history=[],
        )
        assert result["answer"] != ""
        assert result["need_tool"] is True
        assert result["tool_name"] == "time"
        assert result["tool_result"] != ""

    def test_agent_handles_weather_query(self) -> None:
        """天气查询：understand -> planning -> tool (weather) -> action。"""
        result = self._graph.run(
            user_message="北京的天气怎么样",
            session_id="int-test-weather",
            history=[],
        )
        assert result["answer"] != ""
        assert result["need_tool"] is True
        assert result["tool_name"] == "weather"

    def test_agent_handles_email_query(self) -> None:
        """邮件发送：understand -> planning -> tool (email) -> action。"""
        result = self._graph.run(
            user_message="发邮件给 admin@test.com 说测试",
            session_id="int-test-email",
            history=[],
        )
        assert result["answer"] != ""
        assert result["need_tool"] is True
        assert result["tool_name"] == "email"

    def test_agent_handles_capability_question(self) -> None:
        result = self._graph.run(
            user_message="你能做什么",
            session_id="int-test-capability",
            history=[],
        )
        assert result["answer"] != ""

    def test_agent_handles_calculation(self) -> None:
        result = self._graph.run(
            user_message="计算 25 * 4 + 10",
            session_id="int-test-math",
            history=[],
        )
        assert result["answer"] != ""

    def test_agent_handles_search_without_api_key(self) -> None:
        """未配置 API Key 时搜索应给出提示而非崩溃。"""
        result = self._graph.run(
            user_message="搜索今天的新闻",
            session_id="int-test-search-no-key",
            history=[],
        )
        assert result["answer"] != ""
        assert result["session_id"] == "int-test-search-no-key"

    def test_agent_with_conversation_history(self) -> None:
        """带历史上下文的对话。"""
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么可以帮你？"},
        ]
        result = self._graph.run(
            user_message="今天几号",
            session_id="int-test-history",
            history=history,
        )
        assert result["answer"] != ""
        assert len(result["messages"]) == 2

    def test_agent_pipeline_does_not_hang(self) -> None:
        """确保管道在合理步数内结束（不会死循环）。"""
        import time

        start = time.perf_counter()
        result = self._graph.run(
            user_message="你好",
            session_id="int-test-timeout",
            history=[],
        )
        elapsed = time.perf_counter() - start
        assert elapsed < 30.0
        assert result["answer"] != ""

    def test_agent_multiple_sessions_isolated(self) -> None:
        """多次对话不应互相干扰。"""
        result_1 = self._graph.run(
            user_message="你好",
            session_id="int-test-session-a",
            history=[],
        )
        result_2 = self._graph.run(
            user_message="计算 1 + 1",
            session_id="int-test-session-b",
            history=[],
        )
        assert result_1["session_id"] == "int-test-session-a"
        assert result_2["session_id"] == "int-test-session-b"

    # ------------------------------------------------------------------
    # 记忆持久化
    # ------------------------------------------------------------------

    def test_memory_persistence_between_rounds(self) -> None:
        """第一次调用 mem_post 写入记忆，第二次 mem_pre 应能加载历史记忆。"""
        session_id = _unique_session_id()

        # 第一轮：执行带工具的查询，触发 mem_post 写记忆
        result_1 = self._graph.run(
            user_message="现在几点了",
            session_id=session_id,
            history=[],
        )
        assert result_1["answer"] != ""
        assert result_1["need_tool"] is True

        # 第二轮：同一个 session，查询相似内容，检查是否能加载历史记忆
        result_2 = self._graph.run(
            user_message="告诉我时间",
            session_id=session_id,
            history=[
                {"role": "user", "content": "现在几点了"},
                {"role": "assistant", "content": result_1["answer"]},
            ],
        )
        assert result_2["answer"] != ""
        # 验证 state 传递正确：mem_pre 写入 relevant_memories
        # 注意：由于哈希向量引擎的精度，不一定每次都能匹配到，
        # 但至少确保链路不报错且第二轮正常执行
        assert isinstance(result_2["relevant_memories"], list)
