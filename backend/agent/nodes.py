"""Agent graph node implementations."""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

from loguru import logger

from agent.state import AgentState
from config.settings import get_settings
from memory.memory_context import MemoryContext
from memory.store import agent_memory
from services.llm_service import get_model_client
from services.tool_service import get_tool_registry
from tools.base import ToolExecutionContext
from utils.common import normalize_text
from utils.exception import ToolException


_IMPLICIT_COMPLETED_STEPS = {"understand", "plan"}
_RETRIEVABLE_MEMORY_KINDS = ["semantic", "episodic"]
_SEMANTIC_MARKERS = (
    "请记住",
    "记住",
    "以后",
    "默认",
    "我喜欢",
    "我偏好",
    "我习惯",
    "我的",
    "我们公司",
    "本公司",
    "规则",
    "制度",
    "流程",
    "偏好",
    "不要",
    "必须",
    "总是",
    "always",
    "prefer",
    "preference",
    "remember",
)


class AgentNodes:
    """Nodes used by the Agent LangGraph runtime."""

    def __init__(self) -> None:
        self._model_client = get_model_client()
        self._tool_registry = get_tool_registry()

    # ------------------------------------------------------------------
    # Memory nodes
    # ------------------------------------------------------------------

    def mem_pre_node(self, state: AgentState) -> AgentState:
        """Load user-scoped long-term memories relevant to this task."""
        query = state["task_desc"]
        memories: list[dict[str, Any]] = []
        if query:
            settings = get_settings()
            user_id = int(state.get("user_id") or 1)
            for item in agent_memory.search_filtered(
                query=query,
                top_k=5,
                min_score=settings.agent_memory_similarity_threshold,
                metadata_filter={
                    "user_id": user_id,
                    "memory_kind": _RETRIEVABLE_MEMORY_KINDS,
                },
            ):
                score = float(item.get("score", 0))
                if score <= 0:
                    continue
                metadata = dict(item.get("metadata") or {})
                if (
                    metadata.get("memory_kind") == "episodic"
                    and metadata.get("status") == "failed"
                ):
                    continue
                memories.append(
                    {
                        "score": round(score, 4),
                        "text": str(item.get("text") or ""),
                        "metadata": metadata,
                    }
                )
        state["relevant_memories"] = memories
        state["memory_context"] = MemoryContext.build(
            messages=state.get("messages", []),
            relevant_memories=memories,
        )
        logger.debug("[memory] mem_pre loaded {} memories", len(memories))
        return state

    def mem_post_node(self, state: AgentState) -> AgentState:
        """Archive durable semantic notes and execution episodes."""
        records = self._build_memory_records(state)
        if not records:
            logger.debug("[memory] mem_post skipped: no durable memory")
            return state

        session_id = state.get("session_id", "")
        normalized_task = state.get("normalized_task") or state["task_desc"]
        timestamp = time.time()

        for index, record in enumerate(records, start=1):
            memory_kind = str(record["metadata"]["memory_kind"])
            memory_id = hashlib.md5(
                f"{session_id}{normalized_task}{memory_kind}{index}{time.time_ns()}".encode()
            ).hexdigest()[:16]

            try:
                metadata = dict(record["metadata"])
                metadata["timestamp"] = timestamp
                agent_memory.add_text(
                    vector_id=f"mem_{memory_kind}_{memory_id}",
                    text=record["text"],
                    metadata=metadata,
                )
                logger.debug("[memory] mem_post saved kind={}", memory_kind)
            except Exception as exc:
                logger.warning("[memory] mem_post save failed: {}", exc)

        return state

    def _build_memory_records(
        self,
        state: AgentState,
    ) -> list[dict[str, Any]]:
        """Build long-term memory records for the current turn."""
        normalized_task = state.get("normalized_task") or state["task_desc"]
        intent = state.get("intent", {})
        tool_name = str(intent.get("tool_name") or "")
        tool_input = dict(intent.get("tool_input") or {})
        tool_result = state.get("tool_result", "")
        error_info = state.get("error_info", "")
        answer = state.get("answer", "")
        session_id = state.get("session_id", "")
        user_id = int(state.get("user_id") or 1)
        tool_results = list(state.get("tool_results") or [])
        tool_calls = list(state.get("tool_calls") or [])
        status = self._execution_status(tool_results, error_info, tool_result)

        base_metadata = {
            "user_id": user_id,
            "session_id": session_id,
            "normalized_task": normalized_task,
            "type": "agent_memory",
        }

        records: list[dict[str, Any]] = []
        if self._should_archive_episodic(state):
            records.append(
                {
                    "text": self._build_episodic_memory_text(
                        normalized_task=normalized_task,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_result=tool_result,
                        error_info=error_info,
                        answer=answer,
                        tool_results=tool_results,
                        tool_calls=tool_calls,
                        status=status,
                    ),
                    "metadata": {
                        **base_metadata,
                        "memory_kind": "episodic",
                        "status": status,
                        "reusable": status == "success",
                        "tool_names": ",".join(
                            self._collect_tool_names(tool_name, tool_results, tool_calls)
                        ),
                        "tool_count": len(tool_results or tool_calls),
                        "plan_step_count": len(state.get("plan") or []),
                    },
                }
            )

        semantic_text = self._extract_semantic_memory(normalized_task)
        if semantic_text:
            records.append(
                {
                    "text": self._build_semantic_memory_text(semantic_text),
                    "metadata": {
                        **base_metadata,
                        "memory_kind": "semantic",
                        "status": "active",
                        "reusable": True,
                        "source": "user_explicit",
                    },
                }
            )

        return records

    @staticmethod
    def _should_archive_episodic(state: AgentState) -> bool:
        """Only tool executions and execution errors become episodic memory."""
        return bool(
            state.get("tool_results")
            or state.get("tool_calls")
            or state.get("tool_result")
            or state.get("error_info")
            or (state.get("intent") or {}).get("tool_name")
        )

    @staticmethod
    def _execution_status(
        tool_results: list[dict[str, Any]],
        error_info: str,
        tool_result: str,
    ) -> str:
        if any(item.get("status") == "failed" for item in tool_results):
            return "failed"
        if error_info and not tool_result:
            return "failed"
        return "success"

    @staticmethod
    def _collect_tool_names(
        intent_tool_name: str,
        tool_results: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
    ) -> list[str]:
        names: list[str] = []
        for item in tool_results:
            name = str(item.get("tool_name") or "")
            if name and name not in names:
                names.append(name)
        for item in tool_calls:
            name = str(item.get("tool_name") or "")
            if name and name not in names:
                names.append(name)
        if intent_tool_name and intent_tool_name not in names:
            names.append(intent_tool_name)
        return names

    @staticmethod
    def _extract_semantic_memory(normalized_task: str) -> str:
        text = normalize_text(normalized_task)
        if not text:
            return ""
        lowered = text.lower()
        if any(marker.lower() in lowered for marker in _SEMANTIC_MARKERS):
            return text[:500]
        if re.search(r"(我的|我们|本公司|公司).{0,24}(是|为|叫|使用|默认|偏好)", text):
            return text[:500]
        return ""

    @staticmethod
    def _build_episodic_memory_text(
        *,
        normalized_task: str,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_result: str,
        error_info: str,
        answer: str,
        tool_results: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        status: str,
    ) -> str:
        memory_lines = [
            "Memory kind: episodic",
            f"Task: {normalized_task}",
            f"Intent: {tool_name or 'direct_answer'}",
            f"Execution status: {status}",
        ]
        if tool_input:
            memory_lines.append(
                f"Input: {json.dumps(tool_input, ensure_ascii=False)}"
            )
        if tool_calls:
            plan_status = [
                f"{step.get('step_id') or step.get('id')}:{step.get('status', '')}"
                for step in tool_calls
            ]
            memory_lines.append(f"Plan status: {', '.join(plan_status)}")
        if tool_results:
            outcomes = []
            for item in tool_results:
                name = str(item.get("tool_name") or "")
                item_status = str(item.get("status") or "")
                content = str(item.get("content") or "")[:160]
                error = str(item.get("error_msg") or "")[:160]
                preview = content if item_status == "success" else f"error={error}"
                outcomes.append(f"{name}({item_status}): {preview}")
            memory_lines.append("Tool outcomes: " + " | ".join(outcomes))
        elif tool_result:
            memory_lines.append(f"Tool result: {tool_result[:200]}")
        if error_info:
            memory_lines.append(f"Error: {error_info[:200]}")
        if answer:
            memory_lines.append(f"Answer: {answer[:200]}")
        return "\n".join(memory_lines)

    @staticmethod
    def _build_semantic_memory_text(semantic_text: str) -> str:
        return "\n".join(
            [
                "Memory kind: semantic",
                f"Durable user note: {semantic_text}",
                "Scope: user preference, user fact, or business rule",
            ]
        )

    # ------------------------------------------------------------------
    # Understanding and planning
    # ------------------------------------------------------------------

    def understand_node(self, state: AgentState) -> AgentState:
        """Identify the task intent and normalize the user request."""
        analysis = self._model_client.analyze_task(
            message=state["task_desc"],
            context=state["messages"],
            memories=state.get("relevant_memories", []),
            memory_ctx=state.get("memory_context"),
        )
        state["normalized_task"] = str(
            analysis.get("normalized_task") or state["task_desc"],
        )
        state["intent"] = analysis
        return state

    def planning_node(self, state: AgentState) -> AgentState:
        """Create a plan and normalize tool steps into executable records."""
        analysis = state["intent"]
        tool_name = str(analysis.get("tool_name") or "")
        tool_input = dict(analysis.get("tool_input") or {})

        if not tool_name:
            state["need_tool"] = False
            state["plan"] = []
            state["tool_calls"] = []
            return state

        tool_specs = self._tool_registry.list_specs()
        specs_dict = [{"name": s.name, "description": s.description} for s in tool_specs]
        plan = self._model_client.create_plan(
            analysis=analysis,
            memories=state["relevant_memories"],
            tool_specs=specs_dict,
            memory_ctx=state.get("memory_context"),
        )
        state["plan"] = list(plan)

        tool_steps = [s for s in plan if str(s.get("tool_name") or "").strip()]
        if not tool_steps:
            tool_steps = [
                {
                    "id": "tool_1",
                    "phase": "tools",
                    "name": f"Call {tool_name}",
                    "goal": "Execute required tool",
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "depends_on": ["plan"],
                    "status": "pending",
                }
            ]
            state["plan"] = [
                {"id": "understand", "phase": "memory", "status": "completed"},
                {"id": "plan", "phase": "planning", "status": "completed"},
                dict(tool_steps[0]),
            ]

        state["need_tool"] = True
        state["tool_name"] = str(tool_steps[0]["tool_name"])
        state["tool_input"] = dict(tool_steps[0].get("tool_input") or {})
        state["tool_calls"] = [
            self._build_tool_step(step, index)
            for index, step in enumerate(tool_steps, start=1)
        ]
        self._sync_plan_status(state)
        return state

    # ------------------------------------------------------------------
    # Execution and observation
    # ------------------------------------------------------------------

    def tool_node(self, state: AgentState) -> AgentState:
        """Execute one runnable tool step from the current plan."""
        if not state.get("need_tool") or not state.get("tool_calls"):
            return state

        max_steps = int(state.get("max_steps") or 6)
        if state.get("step_count", 0) >= max_steps:
            state["error_info"] = "Reached max tool execution steps"
            self._skip_pending_steps(state, reason="max_steps_reached")
            self._update_tool_summary(state)
            self._sync_plan_status(state)
            return state

        completed = state.get("tool_results", [])
        step = self._next_runnable_step(state.get("tool_calls", []))
        if step is None:
            return state

        tool_name = str(step["tool_name"])
        tool_input = dict(step.get("tool_input") or {})
        tool_input = self._resolve_tool_input(tool_name, tool_input, completed)
        tool_context = self._build_tool_context(state)
        start_time = time.perf_counter()

        step["status"] = "running"
        step["tool_input"] = tool_input
        try:
            tool = self._tool_registry.get(tool_name)
            result = self._run_tool_with_context(tool, tool_name, tool_input, tool_context)
            entry = self._build_tool_result_entry(
                step=step,
                tool_input=self._redact_context_input(tool_input, tool),
                content=normalize_text(result.content),
                metadata={
                    **dict(result.metadata or {}),
                    "execution_context": {
                        "user_id": tool_context.user_id,
                        "session_id": tool_context.session_id,
                        "permissions": sorted(tool_context.permissions),
                    },
                },
                status="success",
                error_msg="",
                start_time=start_time,
            )
            step["tool_input"] = entry["tool_input"]
            step["status"] = "completed"
            step["error_msg"] = ""
            logger.info("[tool] {} completed step={}", tool_name, step["step_id"])
        except (ToolException, ValueError) as exc:
            message = exc.message if isinstance(exc, ToolException) else str(exc)
            entry = self._build_tool_result_entry(
                step=step,
                tool_input=tool_input,
                content="",
                metadata={},
                status="failed",
                error_msg=message,
                start_time=start_time,
            )
            step["status"] = "failed"
            step["error_msg"] = message
            logger.warning("[tool] {} failed step={}: {}", tool_name, step["step_id"], message)

        step["duration_ms"] = entry["duration_ms"]
        completed.append(entry)
        state["tool_results"] = completed
        state["step_count"] = int(state.get("step_count") or 0) + 1
        self._sync_plan_status(state)
        return state

    def observe_node(self, state: AgentState) -> AgentState:
        """Observe execution results and update aggregate plan state."""
        self._skip_blocked_steps(state)
        self._update_tool_summary(state)
        self._sync_plan_status(state)
        return state

    def action_node(self, state: AgentState) -> AgentState:
        """Generate the final answer from memory and tool observations."""
        if state.get("error_info") and not state.get("tool_result"):
            state["answer"] = f"执行过程中遇到问题：{state['error_info']}"
            return state

        state["answer"] = self._model_client.generate(
            user_message=state["normalized_task"] or state["task_desc"],
            tool_result=state["tool_result"] or None,
            context=state["messages"],
            memories=state["relevant_memories"],
            tool_results=state["tool_results"],
        )
        return state

    # ------------------------------------------------------------------
    # Tool execution helpers
    # ------------------------------------------------------------------

    def _run_tool_via_langchain(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> Any | None:
        """Execute a registered tool through its LangChain adapter."""
        langchain_tool = self._tool_registry.get_langchain_tool(tool_name)
        if not langchain_tool:
            return None
        raw_result = langchain_tool.invoke(tool_input)
        try:
            payload = json.loads(str(raw_result))
        except json.JSONDecodeError:
            payload = {"content": str(raw_result), "metadata": {}}
        from tools.base import ToolResult

        return ToolResult(
            content=str(payload.get("content") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )

    def _run_tool_with_context(
        self,
        tool: Any,
        tool_name: str,
        tool_input: dict[str, Any],
        context: ToolExecutionContext,
    ) -> Any:
        """Execute tools through the context-aware protocol when available."""
        required_permissions = getattr(tool, "required_permissions", frozenset())
        if required_permissions:
            return tool.run_with_context(tool_input, context)
        result = self._run_tool_via_langchain(tool_name, tool_input)
        if result is not None:
            return result
        if hasattr(tool, "run_with_context"):
            return tool.run_with_context(tool_input, context)
        return tool.run(tool_input)

    @staticmethod
    def _build_tool_context(state: AgentState) -> ToolExecutionContext:
        """Build the permission and identity context for one tool invocation."""
        user_id = int(state.get("user_id") or 0)
        permissions = {
            "knowledge:read",
            "network:read",
            "email:send",
            "file:read",
            "mcp:call",
        }
        if user_id <= 0:
            permissions = set()
        return ToolExecutionContext(
            user_id=user_id,
            session_id=str(state.get("session_id") or ""),
            permissions=frozenset(permissions),
            metadata={
                "task_desc": str(state.get("task_desc") or ""),
                "normalized_task": str(state.get("normalized_task") or ""),
            },
        )

    @staticmethod
    def _redact_context_input(
        tool_input: dict[str, Any],
        tool: Any,
    ) -> dict[str, Any]:
        """Hide protocol-injected fields from trace-visible tool input."""
        visible_input = dict(tool_input)
        context_schema = getattr(tool, "context_schema", {}) or {}
        for input_key in context_schema.values():
            visible_input.pop(str(input_key), None)
        return visible_input

    @staticmethod
    def _build_tool_step(step: dict[str, Any], index: int) -> dict[str, Any]:
        step_id = str(step.get("id") or f"tool_{index}")
        deps = step.get("depends_on") or []
        if isinstance(deps, str):
            deps = [deps]
        elif not isinstance(deps, list):
            deps = []
        return {
            "id": step_id,
            "step_id": step_id,
            "phase": str(step.get("phase") or "tools"),
            "name": str(step.get("name") or f"Tool step {index}"),
            "goal": str(step.get("goal") or ""),
            "depends_on": [str(dep) for dep in deps],
            "tool_name": str(step.get("tool_name") or ""),
            "tool_input": dict(step.get("tool_input") or {}),
            "status": "pending",
            "error_msg": "",
        }

    @staticmethod
    def _build_tool_result_entry(
        step: dict[str, Any],
        tool_input: dict[str, Any],
        content: str,
        metadata: dict[str, Any],
        status: str,
        error_msg: str,
        start_time: float,
    ) -> dict[str, Any]:
        return {
            "step_id": step.get("step_id") or step.get("id"),
            "phase": step.get("phase", "tools"),
            "name": step.get("name", ""),
            "goal": step.get("goal", ""),
            "depends_on": list(step.get("depends_on") or []),
            "tool_name": step.get("tool_name", ""),
            "tool_input": tool_input,
            "content": content,
            "metadata": metadata,
            "status": status,
            "error_msg": error_msg,
            "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }

    @classmethod
    def _next_runnable_step(
        cls,
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for step in tool_calls:
            if step.get("status") != "pending":
                continue
            if cls._dependencies_satisfied(step, tool_calls):
                return step
        return None

    @classmethod
    def _dependencies_satisfied(
        cls,
        step: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> bool:
        deps = [str(dep) for dep in step.get("depends_on", [])]
        return all(cls._dependency_completed(dep, tool_calls) for dep in deps)

    @staticmethod
    def _dependency_completed(dep: str, tool_calls: list[dict[str, Any]]) -> bool:
        if dep in _IMPLICIT_COMPLETED_STEPS:
            return True
        for step in tool_calls:
            if str(step.get("step_id") or step.get("id")) == dep:
                return step.get("status") == "completed"
        return False

    @staticmethod
    def _dependency_failed(dep: str, tool_calls: list[dict[str, Any]]) -> bool:
        if dep in _IMPLICIT_COMPLETED_STEPS:
            return False
        for step in tool_calls:
            if str(step.get("step_id") or step.get("id")) == dep:
                return step.get("status") in {"failed", "skipped"}
        return False

    @classmethod
    def _skip_blocked_steps(cls, state: AgentState) -> None:
        tool_calls = state.get("tool_calls", [])
        changed = True
        while changed:
            changed = False
            for step in tool_calls:
                if step.get("status") != "pending":
                    continue
                deps = [str(dep) for dep in step.get("depends_on", [])]
                failed_dep = next(
                    (dep for dep in deps if cls._dependency_failed(dep, tool_calls)),
                    "",
                )
                if failed_dep:
                    step["status"] = "skipped"
                    step["error_msg"] = f"dependency_failed:{failed_dep}"
                    changed = True

    @staticmethod
    def _skip_pending_steps(state: AgentState, reason: str) -> None:
        for step in state.get("tool_calls", []):
            if step.get("status") == "pending":
                step["status"] = "skipped"
                step["error_msg"] = reason

    @staticmethod
    def _sync_plan_status(state: AgentState) -> None:
        tool_status = {
            str(step.get("step_id") or step.get("id")): step
            for step in state.get("tool_calls", [])
        }
        synced: list[dict[str, Any]] = []
        for step in state.get("plan", []):
            item = dict(step)
            step_id = str(item.get("id") or "")
            tool_step = tool_status.get(step_id)
            if tool_step:
                item["status"] = tool_step.get("status", item.get("status", "pending"))
                item["tool_input"] = tool_step.get("tool_input", item.get("tool_input", {}))
                item["error_msg"] = tool_step.get("error_msg", "")
                if "duration_ms" in tool_step:
                    item["duration_ms"] = tool_step["duration_ms"]
            synced.append(item)
        state["plan"] = synced

    @staticmethod
    def _update_tool_summary(state: AgentState) -> None:
        successful = [
            r for r in state.get("tool_results", [])
            if r.get("status") == "success"
        ]
        failed = [
            r for r in state.get("tool_results", [])
            if r.get("status") == "failed"
        ]
        if successful:
            state["tool_result"] = "\n\n".join(
                f"[{r['tool_name']}]\n{r['content']}" for r in successful
            )
        else:
            state["tool_result"] = ""
        if failed:
            state["error_info"] = str(failed[-1].get("error_msg") or "")
        elif not state.get("error_info"):
            skipped = [
                s for s in state.get("tool_calls", [])
                if s.get("status") == "skipped"
            ]
            state["error_info"] = str(skipped[-1].get("error_msg") or "") if skipped else ""

    # ------------------------------------------------------------------
    # Cross-step input resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_tool_input(
        tool_name: str,
        tool_input: dict[str, Any],
        completed: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Resolve tool inputs from prior successful steps."""
        if tool_name != "email":
            return tool_input
        resolved = AgentNodes._resolve_email_body(tool_input, completed)
        resolved = AgentNodes._resolve_email_to(resolved)
        return resolved

    @staticmethod
    def _resolve_email_body(
        tool_input: dict[str, Any],
        completed: list[dict[str, Any]],
    ) -> dict[str, Any]:
        body = tool_input.get("body", "")
        if not body:
            return tool_input
        if any(kw in body for kw in ("°C", "温度", "湿度", "风力")):
            return tool_input
        if len(body) >= 50:
            return tool_input
        weather_result = next(
            (
                r for r in completed
                if r.get("tool_name") == "weather" and r.get("status") == "success"
            ),
            None,
        )
        if not weather_result:
            return tool_input
        weather_content = str(weather_result.get("content", ""))
        if not weather_content:
            return tool_input
        resolved = dict(tool_input)
        resolved["body"] = f"{body}\n\n{weather_content}"
        return resolved

    @staticmethod
    def _resolve_email_to(tool_input: dict[str, Any]) -> dict[str, Any]:
        import re

        to_val = tool_input.get("to", "")
        if to_val and str(to_val).strip():
            return tool_input
        resolved = dict(tool_input)
        for field in ("subject", "body"):
            val = str(resolved.get(field, ""))
            match = re.search(
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                val,
            )
            if match:
                resolved["to"] = match.group(0)
                cleaned = val.replace(match.group(0), "").strip()
                resolved[field] = cleaned
                if not resolved.get("subject"):
                    resolved["subject"] = cleaned
                break
        return resolved
