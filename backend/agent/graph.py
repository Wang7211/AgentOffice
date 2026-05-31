"""Agent LangGraph 状态图。"""

import time
from typing import Any

from agent.nodes import AgentNodes
from agent.state import AgentState
from memory.memory_context import MemoryContext
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
    ) -> dict[str, Any]:
        """执行一次 Agent 任务。"""
        start_time = time.perf_counter()
        state = self._build_initial_state(user_message, session_id, history)
        state = self._compiled_graph.invoke(state)
        log_agent_event(
            "agent_finished",
            session_id=session_id,
            need_tool=state["need_tool"],
            tool_name=state["tool_name"] or "none",
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
        graph.add_node("tool", self._nodes.tool_node)
        graph.add_node("action", self._nodes.action_node)
        graph.add_node("mem_post", self._nodes.mem_post_node)

        graph.set_entry_point("mem_pre")
        graph.add_edge("mem_pre", "understand")
        graph.add_edge("understand", "planning")
        graph.add_conditional_edges(
            "planning",
            self._route_after_planning,
            {"tool": "tool", "action": "action"},
        )
        graph.add_edge("tool", "action")
        graph.add_edge("action", "mem_post")
        graph.add_edge("mem_post", END)

        return graph.compile()

    def _route_after_planning(self, state: AgentState) -> str:
        """按 need_tool 路由：需要工具走 tool，否则直接 action。"""
        return "tool" if state["need_tool"] else "action"

    def _build_initial_state(
        self,
        user_message: str,
        session_id: str,
        history: list[dict[str, str]],
    ) -> AgentState:
        """构造状态图运行的初始状态。"""
        return AgentState(
            messages=history,
            task_desc=user_message,
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
            session_id=session_id,
            memory_context=MemoryContext.build(
                messages=history,
                relevant_memories=[],
            ),
        )
