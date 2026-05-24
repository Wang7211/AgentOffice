"""AgentGraph 路由逻辑和状态管理测试。"""

import pytest

from agent.graph import AgentGraph
from agent.state import AgentState


def _make_state(overrides: dict | None = None) -> AgentState:
    """构造最小测试用 AgentState。"""
    defaults: AgentState = {
        "messages": [],
        "task_desc": "test",
        "normalized_task": "",
        "intent": {},
        "constraints": {},
        "task_status": "received",
        "boundary": {},
        "clarification_question": "",
        "relevant_memories": [],
        "plan": [],
        "tool_name": "",
        "tool_input": {},
        "tool_result": "",
        "tool_calls": [],
        "tool_results": [],
        "step_count": 0,
        "error_info": "",
        "answer": "",
        "reflection": {},
        "reflection_retry_count": 0,
        "archived_memory_ids": [],
        "session_id": "test-graph",
    }
    if overrides:
        defaults.update(overrides)
    return defaults


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
        assert state["task_status"] == "received"
        assert state["step_count"] == 0

    # ------------------------------------------------------------------
    # _route_after_understand
    # ------------------------------------------------------------------

    def test_route_after_understand_chat_greeting(self) -> None:
        """闲聊问候路由到 action。"""
        state = _make_state({
            "intent": {"intent_category": "interaction_chat", "intent_subtype": "greeting"},
        })
        route = self._graph._route_after_understand(state)
        assert route == "action"

    def test_route_after_understand_chat_farewell(self) -> None:
        state = _make_state({
            "intent": {"intent_category": "interaction_chat", "intent_subtype": "farewell"},
        })
        route = self._graph._route_after_understand(state)
        assert route == "action"

    def test_route_after_understand_capability_question(self) -> None:
        state = _make_state({
            "intent": {"intent_subtype": "capability_question"},
        })
        route = self._graph._route_after_understand(state)
        assert route == "action"

    def test_route_after_understand_task(self) -> None:
        """普通任务继续到 memory。"""
        state = _make_state({
            "intent": {"intent_category": "task_execution"},
        })
        route = self._graph._route_after_understand(state)
        assert route == "memory"

    # ------------------------------------------------------------------
    # _route_after_validate
    # ------------------------------------------------------------------

    def test_route_after_validate_not_ready(self) -> None:
        state = _make_state({"task_status": "needs_clarification"})
        route = self._graph._route_after_validate(state)
        assert route == "action"

    def test_route_after_validate_task_execution(self) -> None:
        state = _make_state({
            "task_status": "ready",
            "intent": {"intent_category": "task_execution"},
            "tool_name": "search",
        })
        route = self._graph._route_after_validate(state)
        assert route == "planning"

    def test_route_after_validate_inquiry_with_tool(self) -> None:
        state = _make_state({
            "task_status": "ready",
            "intent": {"intent_category": "information_inquiry"},
            "tool_name": "search",
        })
        route = self._graph._route_after_validate(state)
        assert route == "tools"

    def test_route_after_validate_inquiry_no_tool(self) -> None:
        state = _make_state({
            "task_status": "ready",
            "intent": {"intent_category": "information_inquiry"},
            "tool_name": "",
        })
        route = self._graph._route_after_validate(state)
        assert route == "action"

    # ------------------------------------------------------------------
    # _route_after_planning
    # ------------------------------------------------------------------

    def test_route_after_planning_needs_clarification(self) -> None:
        state = _make_state({"task_status": "needs_clarification"})
        route = self._graph._route_after_planning(state)
        assert route == "action"

    def test_route_after_planning_has_tool(self) -> None:
        state = _make_state({
            "task_status": "ready",
            "tool_name": "search",
            "step_count": 0,
        })
        route = self._graph._route_after_planning(state)
        assert route == "tools"

    def test_route_after_planning_no_tool(self) -> None:
        state = _make_state({
            "task_status": "ready",
            "tool_name": "",
            "step_count": 0,
        })
        route = self._graph._route_after_planning(state)
        assert route == "action"

    # ------------------------------------------------------------------
    # _route_after_action
    # ------------------------------------------------------------------

    def test_route_after_action_task_execution(self) -> None:
        state = _make_state({
            "task_status": "completed",
            "intent": {"intent_category": "task_execution"},
        })
        route = self._graph._route_after_action(state)
        assert route == "reflection"

    def test_route_after_action_inquiry(self) -> None:
        state = _make_state({
            "task_status": "completed",
            "intent": {"intent_category": "information_inquiry"},
        })
        route = self._graph._route_after_action(state)
        assert route == "end"

    def test_route_after_action_not_completed(self) -> None:
        state = _make_state({
            "task_status": "failed",
            "intent": {"intent_category": "task_execution"},
        })
        route = self._graph._route_after_action(state)
        assert route == "end"

    # ------------------------------------------------------------------
    # _route_after_reflection
    # ------------------------------------------------------------------

    def test_route_after_reflection_completed(self) -> None:
        state = _make_state({"task_status": "completed"})
        route = self._graph._route_after_reflection(state)
        assert route == "archive"

    def test_route_after_reflection_not_completed(self) -> None:
        state = _make_state({"task_status": "failed"})
        route = self._graph._route_after_reflection(state)
        assert route == "end"

    # ------------------------------------------------------------------
    # _should_call_tool
    # ------------------------------------------------------------------

    def test_should_call_tool_true(self) -> None:
        state = _make_state({"tool_name": "search", "step_count": 0})
        assert self._graph._should_call_tool(state) is True

    def test_should_call_tool_false_no_tool(self) -> None:
        state = _make_state({"tool_name": "", "step_count": 0})
        assert self._graph._should_call_tool(state) is False

    def test_should_call_tool_false_max_steps(self) -> None:
        state = _make_state({"tool_name": "search", "step_count": 3})
        assert self._graph._should_call_tool(state) is False

    # ------------------------------------------------------------------
    # _node_flow
    # ------------------------------------------------------------------

    def test_node_flow_has_all_nodes(self) -> None:
        flow = self._graph._node_flow()
        node_names = [name for name, _ in flow]
        assert node_names == [
            "understand", "memory", "validate", "planning",
            "tools", "action", "reflection", "archive",
        ]
