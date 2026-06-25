"""AgentGraph routing and state management tests."""

from agent.graph import AgentGraph
from agent.state import AgentState


class TestAgentGraphRouting:
    def setup_method(self) -> None:
        self._graph = AgentGraph()

    def test_build_initial_state(self) -> None:
        state = self._graph._build_initial_state(
            user_message="hello",
            session_id="s1",
            history=[{"role": "user", "content": "previous question"}],
            user_id=42,
        )

        assert state["task_desc"] == "hello"
        assert state["session_id"] == "s1"
        assert state["user_id"] == 42
        assert len(state["messages"]) == 1
        assert state["understanding"] == {}
        assert state["capability_context"] == {}
        assert state["task_contract"] == {}
        assert state["task_evaluation"] == {}
        assert state["short_term_summary"] == ""
        assert state["plan"] == []
        assert state["current_step_id"] == ""
        assert state["tool_calls"] == []
        assert state["observations"] == []
        assert state["replan_requested"] is False
        assert state["replan_context"] == {}
        assert state["replan_count"] == 0
        assert state["max_replans"] == 1
        assert state["step_count"] == 0
        assert state["max_steps"] == 6
        assert state["answer"] == ""

    def test_route_after_planning_executes_next_pending_step(self) -> None:
        state = {
            "plan": [
                {
                    "id": "weather",
                    "kind": "tool",
                    "phase": "tools",
                    "tool_name": "weather",
                    "depends_on": [],
                    "status": "pending",
                },
            ]
        }

        assert self._graph._route_after_planning(state) == "execute"

    def test_route_accepts_string_dependency(self) -> None:
        state = {
            "plan": [
                {"id": "weather", "kind": "tool", "status": "completed"},
                {
                    "id": "email",
                    "kind": "tool",
                    "phase": "tools",
                    "tool_name": "email",
                    "depends_on": "weather",
                    "status": "pending",
                },
            ]
        }

        assert self._graph._route_after_planning(state) == "execute"

    def test_route_after_planning_finalizes_without_runnable_step(self) -> None:
        assert self._graph._route_after_planning({"plan": []}) == "finalize"

    def test_route_after_planning_default_state_finalizes(self) -> None:
        state = AgentState(
            messages=[],
            task_desc="",
            normalized_task="",
            understanding={},
            capability_context={},
            task_contract={},
            task_evaluation={},
            short_term_summary="",
            relevant_memories=[],
            plan=[],
            current_step_id="",
            tool_calls=[],
            observations=[],
            replan_requested=False,
            replan_context={},
            replan_count=0,
            max_replans=1,
            step_count=0,
            max_steps=6,
            error_info="",
            answer="",
            session_id="",
            user_id=1,
            memory_context=None,
        )

        assert self._graph._route_after_planning(state) == "finalize"

    def test_run_returns_expected_keys(self) -> None:
        result = self._graph.run(
            user_message="hello",
            session_id="run-test",
            history=[],
        )
        expected_keys = {
            "messages",
            "task_desc",
            "normalized_task",
            "understanding",
            "capability_context",
            "task_contract",
            "task_evaluation",
            "short_term_summary",
            "relevant_memories",
            "plan",
            "current_step_id",
            "tool_calls",
            "observations",
            "replan_requested",
            "replan_context",
            "replan_count",
            "max_replans",
            "step_count",
            "max_steps",
            "error_info",
            "answer",
            "session_id",
            "user_id",
            "memory_context",
        }

        assert set(result.keys()) == expected_keys

    def test_route_after_observe_replans_when_requested(self) -> None:
        state = {"replan_requested": True}
        assert self._graph._route_after_observe(state) == "planning"

    def test_route_after_observe_finalizes_terminal_task_evaluation(self) -> None:
        state = {
            "task_evaluation": {"status": "success"},
            "replan_requested": True,
        }

        assert self._graph._route_after_observe(state) == "finalize"

    def test_route_after_observe_continues_for_runnable_step(self) -> None:
        state = {
            "step_count": 1,
            "max_steps": 6,
            "plan": [
                {"id": "weather", "kind": "tool", "status": "completed"},
                {
                    "id": "email",
                    "kind": "tool",
                    "tool_name": "email",
                    "depends_on": ["weather"],
                    "status": "pending",
                },
            ],
        }

        assert self._graph._route_after_observe(state) == "execute"

    def test_route_after_observe_finalizes_without_runnable_step(self) -> None:
        state = {
            "step_count": 1,
            "max_steps": 6,
            "plan": [
                {"id": "weather", "kind": "tool", "status": "failed"},
                {
                    "id": "email",
                    "kind": "tool",
                    "tool_name": "email",
                    "depends_on": ["weather"],
                    "status": "pending",
                },
            ],
        }

        assert self._graph._route_after_observe(state) == "finalize"

    def test_route_after_observe_finalizes_at_max_steps(self) -> None:
        state = {"step_count": 6, "max_steps": 6, "plan": []}
        assert self._graph._route_after_observe(state) == "finalize"
