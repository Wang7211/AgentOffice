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
            user_id=42,
        )
        assert state["task_desc"] == "你好"
        assert state["session_id"] == "s1"
        assert state["user_id"] == 42
        assert len(state["messages"]) == 1
        assert state["need_tool"] is False
        assert state["plan"] == []
        assert state["step_count"] == 0
        assert state["max_steps"] == 6
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
            plan=[],
            tool_calls=[],
            tool_results=[],
            step_count=0,
            max_steps=6,
            error_info="",
            answer="",
            session_id="",
            user_id=1,
            memory_context=None,
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
            "tool_result", "plan", "tool_calls", "tool_results", "step_count",
            "max_steps",
            "error_info", "answer", "session_id", "user_id", "memory_context",
        }
        assert set(result.keys()) == expected_keys

    def test_route_after_observe_continues_for_runnable_step(self) -> None:
        state = {
            "step_count": 1,
            "max_steps": 6,
            "tool_calls": [
                {"step_id": "weather", "status": "completed"},
                {
                    "step_id": "email",
                    "status": "pending",
                    "depends_on": ["weather"],
                },
            ],
        }
        assert self._graph._route_after_observe(state) == "tool"

    def test_route_after_observe_stops_without_runnable_step(self) -> None:
        state = {
            "step_count": 1,
            "max_steps": 6,
            "tool_calls": [
                {"step_id": "weather", "status": "failed"},
                {
                    "step_id": "email",
                    "status": "pending",
                    "depends_on": ["weather"],
                },
            ],
        }
        assert self._graph._route_after_observe(state) == "action"
