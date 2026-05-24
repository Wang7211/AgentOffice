"""Agent 全流程集成测试（使用 LocalModelClient，不依赖外部 API）。"""

import pytest

from agent.graph import AgentGraph
from agent.state import AgentState


class TestAgentPipeline:
    """测试 Agent 从消息输入到最终输出的完整链路。"""

    def setup_method(self) -> None:
        self._graph = AgentGraph()

    def test_agent_handles_greeting(self) -> None:
        """测试完整链路：打招呼 -> 无需工具 -> 直接回复。"""
        result = self._graph.run(
            user_message="你好",
            session_id="int-test-greeting",
            history=[],
        )
        assert result["task_status"] in ("completed", "needs_clarification")
        assert result["answer"] != ""
        assert result["session_id"] == "int-test-greeting"

    def test_agent_handles_thanks(self) -> None:
        result = self._graph.run(
            user_message="谢谢",
            session_id="int-test-thanks",
            history=[],
        )
        assert result["answer"] == "不客气。"

    def test_agent_handles_farewell(self) -> None:
        result = self._graph.run(
            user_message="再见",
            session_id="int-test-farewell",
            history=[],
        )
        assert result["answer"] == "再见。"

    def test_agent_handles_time_query(self) -> None:
        """时间查询：走规则路由 -> tool 执行 -> 回复。"""
        result = self._graph.run(
            user_message="现在几点了",
            session_id="int-test-time",
            history=[],
        )
        assert result["task_status"] in ("completed", "failed", "needs_clarification")
        if result["task_status"] == "completed":
            assert "北京时间" in result["answer"]

    def test_agent_handles_capability_question(self) -> None:
        result = self._graph.run(
            user_message="你能做什么",
            session_id="int-test-capability",
            history=[],
        )
        assert result["task_status"] == "completed"
        assert len(result["answer"]) > 50

    def test_agent_handles_calculation(self) -> None:
        result = self._graph.run(
            user_message="计算 25 * 4 + 10",
            session_id="int-test-math",
            history=[],
        )
        assert result["task_status"] in ("completed", "failed")
        if result["task_status"] == "completed":
            assert "110" in result["answer"]

    def test_agent_handles_joke_request(self) -> None:
        result = self._graph.run(
            user_message="讲个笑话",
            session_id="int-test-joke",
            history=[],
        )
        assert result["task_status"] == "completed"
        assert len(result["answer"]) > 30

    def test_agent_handles_search_without_api_key(self) -> None:
        """未配置 API Key 时搜索应给出提示而非崩溃。"""
        result = self._graph.run(
            user_message="搜索今天的新闻",
            session_id="int-test-search-no-key",
            history=[],
        )
        assert result["task_status"] in ("completed", "failed")
        if result["task_status"] == "completed":
            assert result["answer"] != ""

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
        assert result["task_status"] in ("completed", "failed", "needs_clarification")
        # 验证历史消息被传递
        assert len(result["messages"]) == 2

    def test_agent_handles_missing_capability_gracefully(self) -> None:
        """对于能力缺失的任务给出友好提示而非崩溃。"""
        result = self._graph.run(
            user_message="查一下明天北京到上海的高铁票",
            session_id="int-test-missing-cap",
            history=[],
        )
        # 可能被检测为 capability_missing 或进入其他状态
        assert result["answer"] != ""
        assert "error_info" in result

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
        assert elapsed < 30.0  # 30 秒内必须完成
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
