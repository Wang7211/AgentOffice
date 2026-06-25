"""Agent 全流程集成测试（使用 LocalModelClient，不依赖外部 API）。"""

import pytest

from agent.graph import AgentGraph
from services.llm_service import LocalModelClient


_RUN_COUNTER = 0


class _ScriptedPlanner(LocalModelClient):
    """Test-only planner that keeps integration tests offline and deterministic."""

    def create_plan(
        self,
        analysis,
        memories,
        tool_specs,
        observations=None,
        memory_ctx=None,
        replan_context=None,
        capability_context=None,
    ):
        _ = memories
        _ = observations
        _ = memory_ctx
        _ = replan_context
        allowed = set((capability_context or {}).get("allowed_tools", []))
        steps = []
        for fact in analysis.get("semantic_facts", []):
            predicate = str(fact.get("predicate") or "")
            obj = str(fact.get("object") or "")
            qualifiers = dict(fact.get("qualifiers") or {})
            tool_name = ""
            tool_input = {}
            if predicate == "query" and obj == "weather":
                tool_name = "weather"
                tool_input = {"city": str(qualifiers.get("location") or "")}
            elif predicate == "query" and obj == "time":
                tool_name = "time"
            elif predicate == "calculate" and obj == "expression":
                tool_name = "code"
                tool_input = {"expression": str(qualifiers.get("expression") or "")}
            elif predicate == "send" and qualifiers.get("channel") == "email":
                tool_name = "email"
                tool_input = {
                    "to": str(qualifiers.get("recipient") or ""),
                    "subject": str(qualifiers.get("subject") or "test"),
                    "body": str(qualifiers.get("body") or "test"),
                }
            if not tool_name or tool_name not in allowed:
                continue
            step_id = f"tool_{len(steps) + 1}"
            steps.append(
                {
                    "id": step_id,
                    "kind": "tool",
                    "phase": "tools",
                    "name": f"Call {tool_name}",
                    "goal": f"Execute {tool_name}",
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "depends_on": [steps[-1]["id"]] if steps else [],
                    "status": "pending",
                }
            )
        steps.append(
            {
                "id": "respond",
                "kind": "compose" if steps else "respond",
                "phase": "action",
                "name": "Respond",
                "goal": "Return the task outcome.",
                "depends_on": [steps[-1]["id"]] if steps else [],
                "status": "pending",
            }
        )
        return steps


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
        self._graph._nodes._model_client = _ScriptedPlanner()

    def test_agent_handles_greeting(self) -> None:
        """打招呼 -> 无需工具 -> 直接回复。"""
        result = self._graph.run(
            user_message="你好",
            session_id="int-test-greeting",
            history=[],
        )
        assert result["answer"] != ""
        assert result["session_id"] == "int-test-greeting"
        assert result["tool_calls"] == []

    def test_agent_handles_time_query(self) -> None:
        """时间查询：understand 识别 tool -> planning 决策 -> tool 执行 -> action 回复。"""
        result = self._graph.run(
            user_message="现在几点了",
            session_id="int-test-time",
            history=[],
        )
        assert result["answer"] != ""
        assert result["tool_calls"][0]["tool_name"] == "time"
        assert result["observations"] != []

    def test_agent_handles_weather_query(self) -> None:
        """天气查询：understand -> planning -> tool (weather) -> action。"""
        result = self._graph.run(
            user_message="北京的天气怎么样",
            session_id="int-test-weather",
            history=[],
        )
        assert result["answer"] != ""
        assert result["tool_calls"][0]["tool_name"] == "weather"

    def test_agent_handles_email_query(self) -> None:
        """邮件发送：understand -> planning -> tool (email) -> action。"""
        result = self._graph.run(
            user_message="发邮件给 admin@test.com 说测试",
            session_id="int-test-email",
            history=[],
        )
        assert result["answer"] != ""
        assert result["tool_calls"][0]["tool_name"] == "email"

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

    def test_memory_context_between_rounds(self) -> None:
        """Second round should load short-term context and long-term retrieval safely."""
        session_id = _unique_session_id()

        # 第一轮：执行带工具的查询。
        result_1 = self._graph.run(
            user_message="现在几点了",
            session_id=session_id,
            history=[],
        )
        assert result_1["answer"] != ""
        assert result_1["tool_calls"] != []

        # 第二轮：同一个 session，检查记忆上下文链路不报错。
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
