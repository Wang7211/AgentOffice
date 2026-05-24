"""Agent 状态图组装。"""

import time
from typing import Any
from typing import Callable

from agent.nodes import AgentNodes
from agent.state import AgentState
from utils.structured_log import log_agent_event
from utils.structured_log import preview_text


AgentNode = Callable[[AgentState], AgentState]


class AgentGraph:
    """基于 LangGraph 的 Agent 状态图。

    LangGraph 依赖未安装时，会自动回退到本地顺序执行器，便于开发环境渐进迁移。
    安装 `langgraph` 后，运行路径会使用真实 StateGraph、节点调度和条件路由。
    """

    max_steps = 3

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
        log_agent_event(
            "agent_started",
            session_id=session_id,
            message_preview=preview_text(user_message),
            history_count=len(history),
            graph_runtime="langgraph" if self._compiled_graph else "local_fallback",
        )
        state = self._build_initial_state(user_message, session_id, history)

        if self._compiled_graph:
            state = self._compiled_graph.invoke(state)
        else:
            state = self._run_local_fallback(state, session_id)

        log_agent_event(
            "agent_finished",
            session_id=session_id,
            task_status=state["task_status"],
            tool_name=state["tool_name"] or "direct_answer",
            step_count=state["step_count"],
            plan_count=len(state["plan"]),
            tool_call_count=len(state["tool_calls"]),
            has_error=bool(state["error_info"]),
            answer_preview=preview_text(state["answer"]),
            duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )
        return dict(state)

    def _compile_langgraph(self) -> Any | None:
        """编译真实 LangGraph StateGraph。"""
        try:
            from langgraph.graph import END
            from langgraph.graph import StateGraph
        except ImportError:
            log_agent_event("langgraph_unavailable", reason="dependency_not_installed")
            return None

        graph = StateGraph(AgentState)
        for node_name, node_func in self._node_flow():
            graph.add_node(node_name, self._wrap_node(node_name, node_func))
        graph.set_entry_point("understand")
        graph.add_conditional_edges(
            "understand",
            self._route_after_understand,
            {"memory": "memory", "action": "action"},
        )
        graph.add_edge("memory", "validate")
        graph.add_conditional_edges(
            "validate",
            self._route_after_validate,
            {"planning": "planning", "tools": "tools", "action": "action"},
        )
        graph.add_conditional_edges(
            "planning",
            self._route_after_planning,
            {"tools": "tools", "action": "action"},
        )
        graph.add_edge("tools", "action")
        graph.add_conditional_edges(
            "action",
            self._route_after_action,
            {"reflection": "reflection", "end": END},
        )
        graph.add_conditional_edges(
            "reflection",
            self._route_after_reflection,
            {"archive": "archive", "end": END},
        )
        graph.add_edge("archive", END)
        return graph.compile()

    def _run_local_fallback(
        self,
        state: AgentState,
        session_id: str,
    ) -> AgentState:
        """LangGraph 未安装时的本地顺序执行回退。"""
        state = self._run_local_node(
            state,
            session_id,
            "understand",
            self._nodes.understand_node,
        )
        if self._route_after_understand(state) == "action":
            state = self._run_local_node(
                state,
                session_id,
                "action",
                self._nodes.action_node,
            )
            return state

        state = self._run_local_node(
            state,
            session_id,
            "memory",
            self._nodes.memory_node,
        )
        state = self._run_local_node(
            state,
            session_id,
            "validate",
            self._nodes.validate_node,
        )
        route = self._route_after_validate(state)
        if route == "planning":
            state = self._run_local_node(
                state,
                session_id,
                "planning",
                self._nodes.planning_node,
            )
            if self._route_after_planning(state) == "tools":
                state = self._run_local_node(
                    state,
                    session_id,
                    "tools",
                    self._nodes.tool_node,
                )
        elif route == "tools":
            state = self._run_local_node(
                state,
                session_id,
                "tools",
                self._nodes.tool_node,
            )

        state = self._run_local_node(
            state,
            session_id,
            "action",
            self._nodes.action_node,
        )
        if self._route_after_action(state) != "reflection":
            return state
        state = self._run_local_node(
            state,
            session_id,
            "reflection",
            self._nodes.reflection_node,
        )
        if self._route_after_reflection(state) == "archive":
            state = self._run_local_node(
                state,
                session_id,
                "archive",
                self._nodes.archive_node,
            )
        return state

    def _run_local_node(
        self,
        state: AgentState,
        session_id: str,
        node_name: str,
        node_func: AgentNode,
    ) -> AgentState:
        """执行单个本地回退节点并记录统一日志。"""
        node_start = time.perf_counter()
        state = node_func(state)
        log_agent_event(
            "agent_node_completed",
            session_id=session_id,
            node=node_name,
            task_status=state["task_status"],
            graph_runtime="local_fallback",
            duration_ms=round((time.perf_counter() - node_start) * 1000, 2),
        )
        return state

    def _wrap_node(self, node_name: str, node_func: AgentNode) -> AgentNode:
        """为 LangGraph 节点增加统一链路日志。"""

        def _wrapped(state: AgentState) -> AgentState:
            node_start = time.perf_counter()
            new_state = node_func(state)
            log_agent_event(
                "agent_node_completed",
                session_id=new_state["session_id"],
                node=node_name,
                task_status=new_state["task_status"],
                graph_runtime="langgraph",
                duration_ms=round((time.perf_counter() - node_start) * 1000, 2),
            )
            return new_state

        return _wrapped

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
            constraints={},
            task_status="received",
            boundary={},
            clarification_question="",
            relevant_memories=[],
            plan=[],
            tool_name="",
            tool_input={},
            tool_result="",
            tool_calls=[],
            tool_results=[],
            step_count=0,
            error_info="",
            answer="",
            reflection={},
            reflection_retry_count=0,
            archived_memory_ids=[],
            session_id=session_id,
        )

    def _node_flow(self) -> list[tuple[str, AgentNode]]:
        """返回单次任务闭环的节点顺序。"""
        return [
            ("understand", self._nodes.understand_node),
            ("memory", self._nodes.memory_node),
            ("validate", self._nodes.validate_node),
            ("planning", self._nodes.planning_node),
            ("tools", self._nodes.tool_node),
            ("action", self._nodes.action_node),
            ("reflection", self._nodes.reflection_node),
            ("archive", self._nodes.archive_node),
        ]

    def _route_after_understand(self, state: AgentState) -> str:
        """闲聊不进入任务处理链路。"""
        analysis = state.get("intent") or {}
        intent_cat = str(
            analysis.get("intent_category")
            or analysis.get("intent_type")
            or "task_execution",
        )
        if intent_cat == "interaction_chat" and str(
            analysis.get("intent_subtype") or "",
        ) in {"greeting", "social_ack", "farewell"}:
            return "action"
        if analysis.get("intent_subtype") == "capability_question":
            return "action"
        return "memory"

    def _route_after_validate(self, state: AgentState) -> str:
        """校验后按任务状态进行条件路由。"""
        if state["task_status"] != "ready":
            return "action"
        intent_category = str(
            state.get("intent", {}).get("intent_category")
            or state.get("intent", {}).get("intent_type")
            or "task_execution",
        )
        if intent_category == "task_execution":
            return "planning"
        if intent_category == "information_inquiry" and state["tool_name"]:
            return "tools"
        return "action"

    def _route_after_planning(self, state: AgentState) -> str:
        """规划后按是否需要工具或澄清进行条件路由。"""
        if state["task_status"] != "ready":
            return "action"
        return "tools" if self._should_call_tool(state) else "action"

    def _route_after_reflection(self, state: AgentState) -> str:
        """反思后只归档可靠完成的任务。"""
        return "archive" if state["task_status"] == "completed" else "end"

    def _route_after_action(self, state: AgentState) -> str:
        """只有任务执行类进入反思闭环。"""
        if state["task_status"] != "completed":
            return "end"
        intent_category = str(
            state.get("intent", {}).get("intent_category")
            or state.get("intent", {}).get("intent_type")
            or "task_execution",
        )
        return "reflection" if intent_category == "task_execution" else "end"

    def _should_call_tool(self, state: AgentState) -> bool:
        """判断状态图是否需要调用工具。"""
        return bool(state["tool_name"]) and state["step_count"] < self.max_steps
