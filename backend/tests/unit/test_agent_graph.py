"""AgentGraph 路由逻辑和状态管理测试。"""

from agent.graph import AgentGraph


class TestAgentGraphRouting:
    def setup_method(self) -> None:
        self._graph = AgentGraph()

    # ------------------------------------------------------------------
    # _build_initial_state
    # ------------------------------------------------------------------

    def test_build_initial_state(self) -> None:
        state = self._graph._build_initial_state(
            user_message="你好",
            session_id="s1",
            history=[{"role": "user", "content": "之前的问题"}],
        )
        assert state["task_desc"] == "你好"
        assert state["session_id"] == "s1"
        assert len(state["messages"]) == 1
        assert state["need_tool"] is False
        assert state["step_count"] == 0
        assert state["answer"] == ""

    # ------------------------------------------------------------------
    # _route_after_planning
    # ------------------------------------------------------------------

    def test_route_after_planning_need_tool(self) -> None:
        state = {"need_tool": True}
        route = self._graph._route_after_planning(state)
        assert route == "tool"

    def test_route_after_planning_no_tool(self) -> None:
        state = {"need_tool": False}
        route = self._graph._route_after_planning(state)
        assert route == "action"

    def test_route_after_planning_default_no_tool(self) -> None:
        """need_tool 初始化为 False 时走 action。"""
        from agent.state import AgentState

        state = AgentState(
            messages=[],
            task_desc="",
            normalized_task="",
            intent={},
            need_tool=False,
            relevant_memories=[],
            tool_name="",
            tool_input={},
            tool_result="",
            tool_calls=[],
            tool_results=[],
            step_count=0,
            error_info="",
            answer="",
            session_id="",
        )
        route = self._graph._route_after_planning(state)
        assert route == "action"

    # ------------------------------------------------------------------
    # run 方法基本行为
    # ------------------------------------------------------------------

    def test_run_returns_expected_keys(self) -> None:
        result = self._graph.run(
            user_message="你好",
            session_id="run-test",
            history=[],
        )
        expected_keys = {
            "messages", "task_desc", "normalized_task", "intent",
            "need_tool", "relevant_memories", "tool_name", "tool_input",
            "tool_result", "tool_calls", "tool_results", "step_count",
            "error_info", "answer", "session_id", "memory_context",
        }
        assert set(result.keys()) == expected_keys
