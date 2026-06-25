"""Agent LangGraph 状态图。"""

import time
from typing import Any

from agent.nodes import AgentNodes
from agent.state import AgentState
from memory.memory_context import MemoryContext
from schemas.agent_contract import normalize_depends_on
from utils.structured_log import log_agent_event
from utils.structured_log import preview_text


class AgentGraph:
    """基于 LangGraph 的 Agent 状态图。"""

    def __init__(self) -> None:
        self._nodes = AgentNodes()
        self._compiled_graph = self._compile_langgraph()

    def run(
        self,
        user_message: str,
        session_id: str,
        history: list[dict[str, str]],
        user_id: int = 1,
        recent_observations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """执行一次 Agent 任务。"""
        start_time = time.perf_counter()
        state = self._build_initial_state(
            user_message,
            session_id,
            history,
            user_id,
            recent_observations=recent_observations or [],
        )
        state = self._compiled_graph.invoke(state)
        tool_observations = [
            item for item in state.get("observations", [])
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ]
        tool_names = []
        for item in tool_observations:
            name = str(item.get("tool_name") or "")
            if name and name not in tool_names:
                tool_names.append(name)
        log_agent_event(
            "agent_finished",
            session_id=session_id,
            tool_count=len(tool_observations),
            tool_names=tool_names,
            has_error=bool(state["error_info"]),
            answer_preview=preview_text(state["answer"]),
            duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )
        return dict(state)

    def _compile_langgraph(self) -> Any:
        """编译 LangGraph StateGraph。"""
        from langgraph.graph import END
        from langgraph.graph import StateGraph

        graph = StateGraph(AgentState)
        graph.add_node("mem_pre", self._nodes.mem_pre_node)
        graph.add_node("understand", self._nodes.understand_node)
        graph.add_node("planning", self._nodes.planning_node)
        graph.add_node("execute", self._nodes.execute_node)
        graph.add_node("observe", self._nodes.observe_node)
        graph.add_node("finalize", self._nodes.finalize_node)
        graph.add_node("mem_post", self._nodes.mem_post_node)

        graph.set_entry_point("mem_pre")
        graph.add_edge("mem_pre", "understand")
        graph.add_edge("understand", "planning")
        graph.add_conditional_edges(
            "planning",
            self._route_after_planning,
            {"execute": "execute", "finalize": "finalize"},
        )
        graph.add_edge("execute", "observe")
        graph.add_conditional_edges(
            "observe",
            self._route_after_observe,
            {"planning": "planning", "execute": "execute", "finalize": "finalize"},
        )
        graph.add_edge("finalize", "mem_post")
        graph.add_edge("mem_post", END)

        return graph.compile()

    def _route_after_planning(self, state: AgentState) -> str:
        """Route by the next executable plan step."""
        return self._route_by_next_plan_step(state)

    def _route_after_observe(self, state: AgentState) -> str:
        """Continue execution, replan after failures when possible."""
        evaluation_status = str(
            (state.get("task_evaluation") or {}).get("status") or ""
        )
        if evaluation_status in {"success", "partial", "blocked", "failed"}:
            return "finalize"
        if state.get("replan_requested"):
            return "planning"
        if state.get("step_count", 0) >= state.get("max_steps", 0):
            return "finalize"
        return self._route_by_next_plan_step(state)

    def _route_by_next_plan_step(self, state: AgentState) -> str:
        next_step = self._next_executable_plan_step(state)
        return "execute" if next_step else "finalize"

    @classmethod
    def _next_executable_plan_step(cls, state: AgentState) -> dict[str, Any] | None:
        plan = list(state.get("plan", []))
        for step in plan:
            if step.get("status") != "pending":
                continue
            if cls._plan_dependencies_satisfied(step, plan):
                return step
        return None

    @staticmethod
    def _plan_dependencies_satisfied(
        step: dict[str, Any],
        plan: list[dict[str, Any]],
    ) -> bool:
        completed_ids = {
            str(item.get("id") or "")
            for item in plan
            if item.get("status") == "completed"
        }
        deps = normalize_depends_on(step.get("depends_on"))
        return all(dep in completed_ids for dep in deps)

    def _build_initial_state(
        self,
        user_message: str,
        session_id: str,
        history: list[dict[str, str]],
        user_id: int,
        recent_observations: list[dict[str, Any]] | None = None,
    ) -> AgentState:
        """构造状态图运行的初始状态。"""
        recent_observations = list(recent_observations or [])
        return AgentState(
            messages=history,
            task_desc=user_message,
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
            recent_observations=recent_observations,
            resolved_references=[],
            replan_requested=False,
            replan_context={},
            replan_count=0,
            max_replans=1,
            step_count=0,
            max_steps=6,
            error_info="",
            answer="",
            session_id=session_id,
            user_id=user_id,
            memory_context=MemoryContext.build(
                messages=history,
                relevant_memories=[],
                short_term_summary="",
                recent_observations=recent_observations,
            ),
        )
