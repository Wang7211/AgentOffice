"""Agent graph node implementations."""

from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

from agent.state import AgentState
from config.settings import get_settings
from memory.memory_context import MemoryContext
from memory.store import agent_memory
from memory.store import chat_memory
from schemas.agent_contract import filter_task_steps
from schemas.agent_contract import normalize_depends_on
from schemas.agent_contract import safe_dict
from services.llm_service import PlanningError
from services.llm_service import get_model_client
from services.tool_service import get_tool_registry
from tools.base import ToolExecutionContext
from utils.common import normalize_text
from utils.exception import ToolException


_LONG_TERM_MEMORY_KIND = "long_term"
_MEMORY_WRITE_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="agent-memory",
)
_REFERENCE_MARKERS = (
    "\u8fd9\u4efd",
    "\u8fd9\u4e2a",
    "\u8fd9\u4e9b",
    "\u521a\u624d",
    "\u4e0a\u6b21",
    "\u524d\u9762",
    "\u5b83",
    "\u5176",
)
_WEATHER_MARKERS = (
    "\u5929\u6c14",
    "\u6e29\u5ea6",
    "\u6e7f\u5ea6",
    "\u51fa\u884c",
    "\u964d\u96e8",
)
_EMAIL_CONTRADICTION_MARKERS = (
    "\u65e0\u6cd5\u76f4\u63a5",
    "\u4e0d\u80fd\u76f4\u63a5",
    "\u65e0\u6cd5\u64cd\u4f5c",
    "\u4e0d\u80fd\u64cd\u4f5c",
    "\u624b\u52a8\u53d1\u9001",
    "\u590d\u5236\u4ee5\u4e0b",
)
_SIDE_EFFECT_TOOLS = {"email"}
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
        """Load short-term summary and user-scoped long-term memories."""
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
                    "memory_kind": _LONG_TERM_MEMORY_KIND,
                },
            ):
                score = float(item.get("score", 0))
                if score <= 0:
                    continue
                metadata = dict(item.get("metadata") or {})
                memories.append(
                    {
                        "score": round(score, 4),
                        "text": str(item.get("text") or ""),
                        "metadata": metadata,
                    }
                )
        short_term_summary = chat_memory.get_summary(state.get("session_id", ""))
        state["short_term_summary"] = short_term_summary
        state["relevant_memories"] = memories
        state["memory_context"] = MemoryContext.build(
            messages=state.get("messages", []),
            relevant_memories=memories,
            short_term_summary=short_term_summary,
            recent_observations=state.get("recent_observations", []),
        )
        logger.debug("[memory] mem_pre loaded {} memories", len(memories))
        return state

    def mem_post_node(self, state: AgentState) -> AgentState:
        """Schedule long-term memory writes for durable user facts."""
        records = self._build_memory_records(state)
        if not records:
            logger.debug("[memory] mem_post skipped: no durable memory")
            return state

        self._schedule_memory_write(records, state)
        logger.debug("[memory] mem_post scheduled {} memories", len(records))
        return state

    @staticmethod
    def _schedule_memory_write(
        records: list[dict[str, Any]],
        state: AgentState,
    ) -> None:
        _MEMORY_WRITE_EXECUTOR.submit(AgentNodes._save_memory_records, records, state)

    @staticmethod
    def _save_memory_records(
        records: list[dict[str, Any]],
        state: AgentState,
    ) -> None:
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

    def _build_memory_records(
        self,
        state: AgentState,
    ) -> list[dict[str, Any]]:
        """Build long-term memory records for durable user facts only."""
        normalized_task = state.get("normalized_task") or state["task_desc"]
        session_id = state.get("session_id", "")
        user_id = int(state.get("user_id") or 1)

        base_metadata = {
            "user_id": user_id,
            "session_id": session_id,
            "normalized_task": normalized_task,
            "type": "agent_memory",
            "memory_kind": _LONG_TERM_MEMORY_KIND,
            "status": "active",
            "reusable": True,
        }

        records: list[dict[str, Any]] = []
        long_term_text = self._extract_long_term_memory(normalized_task)
        if long_term_text:
            records.append(
                {
                    "text": self._build_long_term_memory_text(long_term_text),
                    "metadata": {
                        **base_metadata,
                        "source": "user_explicit",
                    },
                }
            )

        return records

    @staticmethod
    def _extract_long_term_memory(normalized_task: str) -> str:
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
    def _build_long_term_memory_text(memory_text: str) -> str:
        return "\n".join(
            [
                "Memory layer: long_term",
                f"Durable fact: {memory_text}",
                "Scope: user preference, user fact, business rule, or durable knowledge",
            ]
        )

    # ------------------------------------------------------------------
    # Understanding and planning
    # ------------------------------------------------------------------

    def understand_node(self, state: AgentState) -> AgentState:
        """Translate the user request into semantic facts for planning."""
        understanding = self._model_client.analyze_task(
            message=state["task_desc"],
            context=state["messages"],
            memories=state.get("relevant_memories", []),
            memory_ctx=state.get("memory_context"),
        )
        state["normalized_task"] = str(
            understanding.get("normalized_task") or state["task_desc"],
        )
        resolved_references = self._resolve_context_references(state)
        if resolved_references:
            understanding["resolved_references"] = resolved_references
            constraints = self._safe_dict(understanding.get("constraints"))
            constraints["resolved_references"] = resolved_references
            understanding["constraints"] = constraints
        state["resolved_references"] = resolved_references
        state["understanding"] = understanding
        state["task_contract"] = self._build_task_contract(understanding)
        state["task_evaluation"] = {}
        return state

    def planning_node(self, state: AgentState) -> AgentState:
        """Create a plan and normalize tool steps into executable records."""
        understanding = state["understanding"]
        replan_context: dict[str, Any] | None = None
        if state.get("replan_requested"):
            state["replan_count"] = int(state.get("replan_count") or 0) + 1
            replan_context = dict(
                state.get("replan_context")
                or self._build_replan_context(state)
            )
            replan_context["attempt"] = state["replan_count"]
            state["replan_context"] = replan_context
            state["replan_requested"] = False
        else:
            state["replan_context"] = {}
        state["current_step_id"] = ""

        tool_specs = self._tool_registry.list_specs()
        specs_dict = [
            {
                "name": s.name,
                "description": s.description,
                "input_schema": dict(getattr(s, "input_schema", {}) or {}),
                "required_permissions": list(getattr(s, "required_permissions", ()) or ()),
                "context_schema": dict(getattr(s, "context_schema", {}) or {}),
            }
            for s in tool_specs
        ]
        capability_context = self._build_capability_context(
            understanding=understanding,
            tool_specs=specs_dict,
            tool_context=self._build_tool_context(state),
        )
        state["capability_context"] = capability_context
        try:
            raw_plan = self._model_client.create_plan(
                analysis=understanding,
                memories=state["relevant_memories"],
                tool_specs=specs_dict,
                capability_context=capability_context,
                observations=state.get("observations", []),
                memory_ctx=state.get("memory_context"),
                replan_context=replan_context,
            )
        except PlanningError as exc:
            state["plan"] = []
            state["tool_calls"] = []
            state["error_info"] = f"planning_failed: {exc}"
            state["task_evaluation"] = {
                "status": "blocked",
                "satisfied_criteria": [],
                "unmet_criteria": [
                    str(item.get("id") or "")
                    for item in (state.get("task_contract") or {}).get("criteria", [])
                    if isinstance(item, dict) and str(item.get("id") or "")
                ],
                "criteria_results": [],
                "reason": "planning_failed",
            }
            return state
        plan = self._task_steps_only(list(raw_plan))
        state["plan"] = plan

        tool_steps = [s for s in plan if str(s.get("tool_name") or "").strip()]
        state["tool_calls"] = [
            self._build_tool_step(step, index)
            for index, step in enumerate(tool_steps, start=1)
        ]
        self._sync_plan_status(state)
        return state

    @staticmethod
    def _task_steps_only(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return filter_task_steps(plan)

    # ------------------------------------------------------------------
    # Execution and observation
    # ------------------------------------------------------------------

    def execute_node(self, state: AgentState) -> AgentState:
        """Execute exactly one runnable plan step."""
        step = self._current_or_next_plan_step(state)
        if step is None:
            state["current_step_id"] = ""
            return state

        if state.get("step_count", 0) >= int(state.get("max_steps") or 6):
            state["error_info"] = "Reached max execution steps"
            self._mark_plan_step(state, str(step.get("id") or ""), "failed", state["error_info"])
            return state

        step_id = str(step.get("id") or "")
        state["current_step_id"] = step_id
        self._mark_plan_step(state, step_id, "running")

        try:
            observation = self._execute_plan_step(step, state)
        except Exception as exc:
            observation = self._build_step_observation(
                step=step,
                observation_type="step_result",
                status="failed",
                content="",
                error_msg=str(exc),
            )

        self._append_observation(state, observation)
        status = self._plan_status_from_observation(observation)
        self._mark_plan_step(
            state,
            step_id,
            status,
            str(observation.get("error_msg") or ""),
        )
        if status == "failed":
            state["error_info"] = str(observation.get("error_msg") or "")
        state["step_count"] = int(state.get("step_count") or 0) + 1
        self._sync_plan_status(state)
        return state

    def finalize_node(self, state: AgentState) -> AgentState:
        """Finalize the run without executing additional plan steps."""
        if state.get("answer"):
            return state

        task_evaluation = self._safe_dict(state.get("task_evaluation"))
        if task_evaluation.get("reason") == "planning_failed":
            detail = str(state.get("error_info") or "planning service unavailable")
            state["answer"] = (
                "Unable to create a reliable execution plan. "
                f"The task was not executed. Details: {detail}"
            )
            return state

        observations = state.get("observations", [])
        observation_summary = self._build_observation_summary(observations) or None
        if state.get("error_info") and not observation_summary:
            state["answer"] = f"执行过程中遇到问题：{state['error_info']}"
            return state

        state["answer"] = self._model_client.generate(
            user_message=state["normalized_task"] or state["task_desc"],
            observation_summary=observation_summary,
            context=state["messages"],
            memories=state["relevant_memories"],
            plan=state.get("plan", []),
            observations=observations,
        )
        state["answer"] = self._sanitize_final_answer(state)
        return state

    def observe_node(self, state: AgentState) -> AgentState:
        """Observe execution results and update aggregate plan state."""
        self._skip_blocked_steps(state)
        self._update_observation_status(state)
        self._sync_plan_status(state)
        state["task_evaluation"] = self._evaluate_task_completion(state)
        self._request_replan_if_needed(state)
        state["current_step_id"] = ""
        return state

    @classmethod
    def _build_task_contract(
        cls,
        understanding: dict[str, Any],
    ) -> dict[str, Any]:
        """Build verifiable outcomes from semantic facts without planning execution."""
        criteria: list[dict[str, Any]] = []
        for index, fact in enumerate(cls._semantic_facts(understanding), start=1):
            criterion_id = f"fact_{index}"
            tool_name = cls._desired_tool_for_fact(fact)
            if tool_name:
                criteria.append(
                    {
                        "id": criterion_id,
                        "type": "observation",
                        "description": cls._criterion_description(fact),
                        "tool_name": tool_name,
                        "qualifiers": cls._safe_dict(fact.get("qualifiers")),
                        "required": True,
                    }
                )
            elif cls._requires_semantic_quality(fact):
                criteria.append(
                    {
                        "id": criterion_id,
                        "type": "semantic_quality",
                        "description": cls._criterion_description(fact),
                        "fact": dict(fact),
                        "required": True,
                    }
                )
            elif cls._requires_external_capability(fact):
                criteria.append(
                    {
                        "id": criterion_id,
                        "type": "external_action",
                        "description": cls._criterion_description(fact),
                        "fact": dict(fact),
                        "required": True,
                    }
                )

        return {
            "objective": str(understanding.get("normalized_task") or ""),
            "constraints": cls._safe_dict(understanding.get("constraints")),
            "criteria": criteria,
        }

    def _evaluate_task_completion(self, state: AgentState) -> dict[str, Any]:
        """Evaluate task completion with rules first and semantic review only when needed."""
        contract = self._safe_dict(state.get("task_contract"))
        criteria = [
            dict(item)
            for item in contract.get("criteria", [])
            if isinstance(item, dict) and item.get("required", True)
        ]
        if not criteria:
            return self._evaluate_without_contract(state)

        results: list[dict[str, Any]] = []
        semantic_criteria: list[dict[str, Any]] = []
        for criterion in criteria:
            criterion_type = str(criterion.get("type") or "")
            if criterion_type == "observation":
                evidence = self._matching_observation(criterion, state)
                results.append(
                    self._criterion_result(
                        criterion,
                        satisfied=evidence is not None,
                        evidence=evidence,
                        reason="matching_success_observation" if evidence else "observation_missing",
                    )
                )
            elif criterion_type == "semantic_quality":
                semantic_criteria.append(criterion)
            elif criterion_type == "external_action":
                results.append(
                    self._criterion_result(
                        criterion,
                        satisfied=False,
                        evidence=None,
                        reason="external_action_not_observed",
                    )
                )

        if semantic_criteria:
            results.extend(
                self._evaluate_semantic_criteria(
                    semantic_criteria,
                    contract,
                    state,
                )
            )

        satisfied = [
            str(item.get("id") or "")
            for item in results
            if item.get("satisfied")
        ]
        unmet = [
            str(item.get("id") or "")
            for item in results
            if not item.get("satisfied")
        ]
        status = self._completion_status(
            state=state,
            satisfied=satisfied,
            unmet=unmet,
        )
        return {
            "status": status,
            "satisfied_criteria": satisfied,
            "unmet_criteria": unmet,
            "criteria_results": results,
            "reason": self._completion_reason(status, unmet),
        }

    @staticmethod
    def _criterion_description(fact: dict[str, Any]) -> str:
        predicate = str(fact.get("predicate") or "requested outcome")
        obj = str(fact.get("object") or "task")
        return f"Satisfy requested outcome: {predicate} {obj}."

    @staticmethod
    def _requires_semantic_quality(fact: dict[str, Any]) -> bool:
        return str(fact.get("predicate") or "").lower() in {
            "analyze",
            "compare",
            "compose",
            "draft",
            "explain",
            "recommend",
            "rewrite",
            "summarize",
            "translate",
            "write",
        }

    @classmethod
    def _matching_observation(
        cls,
        criterion: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any] | None:
        tool_name = str(criterion.get("tool_name") or "")
        qualifiers = cls._safe_dict(criterion.get("qualifiers"))
        for observation in reversed(state.get("observations", [])):
            if observation.get("type") != "tool_result":
                continue
            if observation.get("status") != "success":
                continue
            if str(observation.get("tool_name") or "") != tool_name:
                continue
            if not cls._observation_matches_qualifiers(observation, qualifiers):
                continue
            return {
                "step_id": str(observation.get("step_id") or ""),
                "tool_name": tool_name,
                "content_preview": str(observation.get("content") or "")[:160],
            }
        return None

    @staticmethod
    def _observation_matches_qualifiers(
        observation: dict[str, Any],
        qualifiers: dict[str, Any],
    ) -> bool:
        tool_input = AgentNodes._safe_dict(observation.get("tool_input"))
        content = str(observation.get("content") or "").strip().lower()
        expected_location = str(qualifiers.get("location") or "").strip().lower()
        if expected_location:
            actual_city = str(tool_input.get("city") or "").strip().lower()
            if expected_location not in actual_city and expected_location not in content:
                return False
        expected_recipient = str(qualifiers.get("recipient") or "").strip().lower()
        if expected_recipient:
            actual_recipient = str(tool_input.get("to") or "").strip().lower()
            if actual_recipient != expected_recipient and expected_recipient not in content:
                return False
        return True

    @staticmethod
    def _criterion_result(
        criterion: dict[str, Any],
        *,
        satisfied: bool,
        evidence: dict[str, Any] | None,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "id": str(criterion.get("id") or ""),
            "type": str(criterion.get("type") or ""),
            "satisfied": satisfied,
            "evidence": evidence or {},
            "reason": reason,
        }

    def _evaluate_semantic_criteria(
        self,
        criteria: list[dict[str, Any]],
        contract: dict[str, Any],
        state: AgentState,
    ) -> list[dict[str, Any]]:
        answer = str(state.get("answer") or "").strip()
        if not answer:
            return [
                self._criterion_result(
                    criterion,
                    satisfied=False,
                    evidence=None,
                    reason="answer_missing",
                )
                for criterion in criteria
            ]
        review = self._model_client.evaluate_completion_quality(
            task_contract={**contract, "criteria": criteria},
            answer=answer,
            observations=list(state.get("observations") or []),
        )
        satisfied = bool(review.get("satisfied"))
        reason = str(review.get("reason") or "semantic_quality_checked")
        return [
            self._criterion_result(
                criterion,
                satisfied=satisfied,
                evidence={"review": reason},
                reason=reason,
            )
            for criterion in criteria
        ]

    @classmethod
    def _completion_status(
        cls,
        state: AgentState,
        satisfied: list[str],
        unmet: list[str],
    ) -> str:
        if not unmet:
            return "success"
        if cls._next_executable_plan_step(state) is not None:
            return "in_progress"

        capability_context = cls._safe_dict(state.get("capability_context"))
        missing_requirements = capability_context.get("missing_requirements", [])
        unavailable = capability_context.get("unavailable_capabilities", [])
        forbidden = capability_context.get("forbidden_actions", [])
        if missing_requirements:
            return "blocked"
        if unavailable or forbidden:
            return "partial" if satisfied else "blocked"
        if int(state.get("replan_count") or 0) < int(state.get("max_replans") or 0):
            return "in_progress"
        if satisfied:
            return "partial"
        return "failed"

    @staticmethod
    def _completion_reason(status: str, unmet: list[str]) -> str:
        if status == "success":
            return "all_required_criteria_satisfied"
        if status == "in_progress":
            return "criteria_remaining"
        if status == "blocked":
            return "task_blocked"
        if status == "partial":
            return "partially_satisfied"
        return "required_criteria_unsatisfied:" + ",".join(unmet)

    @classmethod
    def _evaluate_without_contract(cls, state: AgentState) -> dict[str, Any]:
        if cls._next_executable_plan_step(state) is not None:
            status = "in_progress"
        elif state.get("error_info"):
            can_replan = int(state.get("replan_count") or 0) < int(
                state.get("max_replans") or 0
            )
            status = "in_progress" if can_replan else "failed"
        elif state.get("answer"):
            status = "success"
        else:
            status = "in_progress"
        return {
            "status": status,
            "satisfied_criteria": [],
            "unmet_criteria": [],
            "criteria_results": [],
            "reason": "legacy_state_without_task_contract",
        }

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

    @classmethod
    def _build_capability_context(
        cls,
        understanding: dict[str, Any],
        tool_specs: list[dict[str, Any]],
        tool_context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Resolve semantic facts against currently executable capabilities."""
        tools_by_name = {
            str(spec.get("name") or ""): dict(spec)
            for spec in tool_specs
            if str(spec.get("name") or "")
        }
        allowed_tools: list[str] = []
        forbidden_actions: list[dict[str, Any]] = []
        for tool_name, spec in tools_by_name.items():
            required_permissions = {
                str(permission)
                for permission in spec.get("required_permissions", [])
                if str(permission)
            }
            if required_permissions and not tool_context.has_permissions(required_permissions):
                forbidden_actions.append(
                    {
                        "tool_name": tool_name,
                        "reason": "missing_permission",
                        "required_permissions": sorted(required_permissions),
                    }
                )
                continue
            allowed_tools.append(tool_name)

        unavailable_capabilities: list[dict[str, Any]] = []
        missing_requirements: list[dict[str, Any]] = []
        for fact in cls._semantic_facts(understanding):
            desired_tool = cls._desired_tool_for_fact(fact)
            if desired_tool:
                if desired_tool not in tools_by_name:
                    unavailable_capabilities.append(
                        cls._capability_gap(
                            fact=fact,
                            desired_tool=desired_tool,
                            reason="tool_not_registered",
                        )
                    )
                elif desired_tool not in allowed_tools:
                    unavailable_capabilities.append(
                        cls._capability_gap(
                            fact=fact,
                            desired_tool=desired_tool,
                            reason="tool_not_allowed",
                        )
                    )
                elif cls._fact_missing_required_info(fact, desired_tool):
                    missing_requirements.append(
                        cls._capability_gap(
                            fact=fact,
                            desired_tool=desired_tool,
                            reason="missing_required_info",
                        )
                    )
                continue

            if cls._requires_external_capability(fact):
                unavailable_capabilities.append(
                    cls._capability_gap(
                        fact=fact,
                        desired_tool="",
                        reason="unsupported_capability",
                    )
                )

        return {
            "allowed_tools": allowed_tools,
            "tools": [
                spec for spec in tool_specs if str(spec.get("name") or "") in allowed_tools
            ],
            "unavailable_capabilities": unavailable_capabilities,
            "forbidden_actions": forbidden_actions,
            "missing_requirements": missing_requirements,
        }

    @staticmethod
    def _semantic_facts(understanding: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            dict(fact)
            for fact in understanding.get("semantic_facts", [])
            if isinstance(fact, dict)
        ]

    @staticmethod
    def _desired_tool_for_fact(fact: dict[str, Any]) -> str:
        predicate = str(fact.get("predicate") or "")
        obj = str(fact.get("object") or "")
        qualifiers = AgentNodes._safe_dict(fact.get("qualifiers"))
        if predicate == "query" and obj == "weather":
            return "weather"
        if predicate == "query" and obj == "time":
            return "time"
        if predicate == "calculate" and obj == "expression":
            return "code"
        if predicate == "read" and obj == "file":
            return "file"
        if predicate == "search" and obj == "public_information":
            return "search"
        if predicate == "send" and str(qualifiers.get("channel") or "") == "email":
            return "email"
        return ""

    @staticmethod
    def _fact_missing_required_info(fact: dict[str, Any], desired_tool: str) -> bool:
        qualifiers = AgentNodes._safe_dict(fact.get("qualifiers"))
        if desired_tool == "file":
            return not str(qualifiers.get("file_path") or "").strip()
        return False

    @staticmethod
    def _requires_external_capability(fact: dict[str, Any]) -> bool:
        predicate = str(fact.get("predicate") or "").lower()
        obj = str(fact.get("object") or "").lower()
        if predicate in {
            "book",
            "reserve",
            "buy",
            "purchase",
            "order",
            "pay",
            "transfer",
            "call",
            "publish",
            "post",
            "upload",
            "download",
            "install",
            "delete",
            "update",
            "create",
            "send",
        }:
            return True
        return predicate == "query" and obj not in {
            "general_answer",
            "conversation",
            "agent_capability",
        }

    @staticmethod
    def _capability_gap(
        fact: dict[str, Any],
        desired_tool: str,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "fact": dict(fact),
            "desired_tool": desired_tool,
            "reason": reason,
            "predicate": str(fact.get("predicate") or ""),
            "object": str(fact.get("object") or ""),
        }

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
        return {
            "id": step_id,
            "step_id": step_id,
            "kind": str(step.get("kind") or "tool"),
            "phase": str(step.get("phase") or "tools"),
            "name": str(step.get("name") or f"Tool step {index}"),
            "goal": str(step.get("goal") or ""),
            "depends_on": AgentNodes._normalize_depends_on(step.get("depends_on")),
            "tool_name": str(step.get("tool_name") or ""),
            "tool_input": AgentNodes._safe_dict(step.get("tool_input")),
            "status": "pending",
            "error_msg": "",
        }

    @staticmethod
    def _safe_dict(value: Any) -> dict[str, Any]:
        return safe_dict(value)

    @staticmethod
    def _normalize_depends_on(value: Any) -> list[str]:
        return normalize_depends_on(value)

    @classmethod
    def _resolve_context_references(
        cls,
        state: AgentState,
    ) -> list[dict[str, Any]]:
        task = str(state.get("task_desc") or "")
        if not cls._has_context_reference(task):
            return []
        observations = cls._recent_success_tool_observations(state)
        if not observations:
            return []

        preferred_tool = cls._preferred_reference_tool(task)
        candidates = list(reversed(observations))
        selected = None
        if preferred_tool:
            selected = next(
                (
                    item for item in candidates
                    if item.get("tool_name") == preferred_tool
                ),
                None,
            )
        if selected is None:
            selected = next(
                (
                    item for item in candidates
                    if item.get("tool_name") not in _SIDE_EFFECT_TOOLS
                ),
                None,
            )
        if selected is None:
            return []

        return [
            {
                "source": "recent_observation",
                "reason": "follow_up_reference",
                "tool_name": selected.get("tool_name", ""),
                "tool_input": cls._safe_dict(selected.get("tool_input")),
                "content": str(selected.get("content") or ""),
                "observation": dict(selected),
            }
        ]

    @staticmethod
    def _has_context_reference(task: str) -> bool:
        return any(marker in task for marker in _REFERENCE_MARKERS)

    @staticmethod
    def _preferred_reference_tool(task: str) -> str:
        if any(marker in task for marker in _WEATHER_MARKERS):
            return "weather"
        return ""

    @classmethod
    def _recent_success_tool_observations(
        cls,
        state: AgentState,
    ) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for source in (
            state.get("recent_observations", []),
            state.get("observations", []),
        ):
            if not isinstance(source, list):
                continue
            for item in source:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "tool_result":
                    continue
                if item.get("status") != "success":
                    continue
                observations.append(dict(item))
        return observations

    @classmethod
    def _sanitize_final_answer(cls, state: AgentState) -> str:
        answer = str(state.get("answer") or "").strip()
        consistent_answer = cls._answer_from_successful_side_effects(state, answer)
        if consistent_answer:
            return consistent_answer
        if not cls._looks_like_tool_call_text(answer):
            return answer
        summary = cls._build_observation_summary(state.get("observations", []))
        if summary:
            return (
                "执行过程中产生了内部工具调用指令，但该指令不能直接作为回复。"
                f"\n\n当前执行结果：\n{summary}"
            )
        return "执行过程中产生了内部工具调用指令，但没有得到可直接展示的结果。"

    @staticmethod
    def _looks_like_tool_call_text(answer: str) -> bool:
        text = answer.strip()
        if not text:
            return False
        lowered = text.lower()
        if lowered.startswith(("call\n", "call ", "tool_call", "function_call")):
            return True
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict):
            return False
        return (
            "arguments" in payload
            and ("name" in payload or "tool_name" in payload)
        )

    @classmethod
    def _answer_from_successful_side_effects(
        cls,
        state: AgentState,
        answer: str,
    ) -> str:
        if not answer:
            return ""
        if not any(marker in answer for marker in _EMAIL_CONTRADICTION_MARKERS):
            return ""
        email_observation = next(
            (
                item for item in reversed(cls._tool_observations(state))
                if item.get("tool_name") == "email"
                and item.get("status") == "success"
                and str(item.get("content") or "")
            ),
            None,
        )
        if not email_observation:
            return ""
        content = str(email_observation.get("content") or "").strip()
        weather_observation = next(
            (
                item for item in reversed(cls._tool_observations(state))
                if item.get("tool_name") == "weather"
                and item.get("status") == "success"
                and str(item.get("content") or "")
            ),
            None,
        )
        if weather_observation:
            return (
                f"{content}\n\n"
                f"\u90ae\u4ef6\u5185\u5bb9\u57fa\u4e8e\u4ee5\u4e0b\u5de5\u5177\u7ed3\u679c\uff1a\n"
                f"{str(weather_observation.get('content') or '').strip()}"
            )
        return content

    def _current_or_next_plan_step(self, state: AgentState) -> dict[str, Any] | None:
        current_step_id = str(state.get("current_step_id") or "")
        if current_step_id:
            for step in state.get("plan", []):
                if (
                    str(step.get("id") or "") == current_step_id
                    and step.get("status") in {"pending", "running"}
                ):
                    return dict(step)
        return self._next_executable_plan_step(state)

    @classmethod
    def _next_executable_plan_step(
        cls,
        state: AgentState,
    ) -> dict[str, Any] | None:
        plan = list(state.get("plan", []))
        for step in plan:
            if step.get("status") != "pending":
                continue
            if cls._plan_dependencies_satisfied(step, plan):
                return dict(step)
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
        deps = AgentNodes._normalize_depends_on(step.get("depends_on"))
        return all(dep in completed_ids for dep in deps)

    def _execute_plan_step(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        kind = self._step_kind(step)
        if kind == "tool":
            return self._execute_tool_step(step, state)
        if kind == "clarify":
            return self._execute_clarify_step(step, state)
        if kind == "compose":
            return self._execute_compose_step(step, state)
        return self._execute_respond_step(step, state)

    @staticmethod
    def _step_kind(step: dict[str, Any]) -> str:
        kind = str(step.get("kind") or "").strip()
        if kind:
            return kind
        if str(step.get("tool_name") or "").strip():
            return "tool"
        phase = str(step.get("phase") or "").strip()
        if phase == "action":
            return "respond"
        return phase or "respond"

    def _execute_tool_step(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        completed = self._tool_observations(state)
        reusable_context = self._tool_reuse_context(state)
        tool_name = str(step.get("tool_name") or "").strip()
        tool_input = self._safe_dict(step.get("tool_input"))
        start_time = time.perf_counter()
        if not tool_name:
            message = "tool step missing required tool_name"
            observation = self._build_tool_observation(
                step=step,
                tool_input=tool_input,
                content="",
                metadata={},
                status="failed",
                error_msg=message,
                start_time=start_time,
            )
            self._mark_tool_call(
                state,
                str(step.get("id") or ""),
                "failed",
                tool_input,
                message,
                observation.get("duration_ms"),
            )
            return observation
        tool_input = self._resolve_tool_input(
            tool_name,
            tool_input,
            completed + reusable_context,
        )
        self._mark_tool_call(state, str(step.get("id") or ""), "running", tool_input)

        reusable_observation = self._find_reusable_observation(
            tool_name,
            tool_input,
            state,
        )
        if reusable_observation is not None:
            observation = self._build_reused_tool_observation(
                step,
                tool_input,
                reusable_observation,
            )
            self._mark_tool_call(
                state,
                str(step.get("id") or ""),
                "completed",
                dict(observation.get("tool_input") or {}),
                "",
                observation.get("duration_ms"),
            )
            return observation

        tool_context = self._build_tool_context(state)
        try:
            tool = self._tool_registry.get(tool_name)
            result = self._run_tool_with_context(tool, tool_name, tool_input, tool_context)
            observation = self._build_tool_observation(
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
            self._mark_tool_call(
                state,
                str(step.get("id") or ""),
                "completed",
                dict(observation.get("tool_input") or {}),
                "",
                observation.get("duration_ms"),
            )
            return observation
        except (ToolException, ValueError) as exc:
            message = exc.message if isinstance(exc, ToolException) else str(exc)
            observation = self._build_tool_observation(
                step=step,
                tool_input=tool_input,
                content="",
                metadata={},
                status="failed",
                error_msg=message,
                start_time=start_time,
            )
            self._mark_tool_call(
                state,
                str(step.get("id") or ""),
                "failed",
                tool_input,
                message,
                observation.get("duration_ms"),
            )
            return observation
        except Exception as exc:
            message = str(exc)
            observation = self._build_tool_observation(
                step=step,
                tool_input=tool_input,
                content="",
                metadata={},
                status="failed",
                error_msg=message,
                start_time=start_time,
            )
            self._mark_tool_call(
                state,
                str(step.get("id") or ""),
                "failed",
                tool_input,
                message,
                observation.get("duration_ms"),
            )
            return observation

    def _execute_clarify_step(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        content = str(step.get("goal") or "请补充必要信息后我再继续处理。")
        state["answer"] = content
        return self._build_step_observation(
            step=step,
            observation_type="clarification_request",
            status="completed",
            content=content,
        )

    def _execute_compose_step(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        answer = self._generate_answer_from_observations(state)
        response_text = str(step.get("response_text") or "").strip()
        if response_text:
            answer = f"{answer}\n\n{response_text}" if answer else response_text
        state["answer"] = answer
        return self._build_step_observation(
            step=step,
            observation_type="answer_draft",
            status="completed",
            content=answer,
        )

    def _execute_respond_step(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        answer = str(step.get("response_text") or "").strip()
        if not answer:
            answer = self._generate_answer_from_observations(state)
        state["answer"] = answer
        return self._build_step_observation(
            step=step,
            observation_type="final_answer",
            status="completed",
            content=answer,
        )

    def _generate_answer_from_observations(self, state: AgentState) -> str:
        observations = state.get("observations", [])
        observation_summary = self._build_observation_summary(observations) or None
        if state.get("error_info") and not observation_summary:
            observation_summary = f"执行过程中遇到问题：{state['error_info']}"
        answer = self._model_client.generate(
            user_message=state["normalized_task"] or state["task_desc"],
            observation_summary=observation_summary,
            context=state["messages"],
            memories=state["relevant_memories"],
            plan=state.get("plan", []),
            observations=observations,
        )
        state["answer"] = answer
        return self._sanitize_final_answer(state)

    @staticmethod
    def _build_step_observation(
        step: dict[str, Any],
        observation_type: str,
        status: str,
        content: str,
        error_msg: str = "",
    ) -> dict[str, Any]:
        return {
            "type": observation_type,
            "step_id": step.get("id", ""),
            "kind": step.get("kind", ""),
            "phase": step.get("phase", ""),
            "name": step.get("name", ""),
            "goal": step.get("goal", ""),
            "content": content,
            "status": status,
            "error_msg": error_msg,
        }

    @staticmethod
    def _append_observation(state: AgentState, observation: dict[str, Any]) -> None:
        observations = list(state.get("observations") or [])
        observations.append(observation)
        state["observations"] = observations

    @staticmethod
    def _plan_status_from_observation(observation: dict[str, Any]) -> str:
        status = str(observation.get("status") or "")
        if status == "success":
            return "completed"
        if status in {"completed", "failed", "skipped"}:
            return status
        return "failed" if observation.get("error_msg") else "completed"

    @staticmethod
    def _mark_plan_step(
        state: AgentState,
        step_id: str,
        status: str,
        error_msg: str = "",
    ) -> None:
        updated: list[dict[str, Any]] = []
        for step in state.get("plan", []):
            item = dict(step)
            if str(item.get("id") or "") == step_id:
                item["status"] = status
                item["error_msg"] = error_msg
            updated.append(item)
        state["plan"] = updated

    @staticmethod
    def _mark_tool_call(
        state: AgentState,
        step_id: str,
        status: str,
        tool_input: dict[str, Any] | None = None,
        error_msg: str = "",
        duration_ms: Any = None,
    ) -> None:
        updated: list[dict[str, Any]] = []
        for step in state.get("tool_calls", []):
            item = dict(step)
            if str(item.get("step_id") or item.get("id")) == step_id:
                item["status"] = status
                item["error_msg"] = error_msg
                if tool_input is not None:
                    item["tool_input"] = dict(tool_input)
                if duration_ms is not None:
                    item["duration_ms"] = duration_ms
            updated.append(item)
        state["tool_calls"] = updated

    @classmethod
    def _request_replan_if_needed(cls, state: AgentState) -> None:
        if state.get("replan_requested"):
            return
        if int(state.get("replan_count") or 0) >= int(state.get("max_replans") or 0):
            return
        evaluation = cls._safe_dict(state.get("task_evaluation"))
        if str(evaluation.get("status") or "") in {
            "success",
            "partial",
            "blocked",
            "failed",
        }:
            return
        has_failed_tool = any(
            result.get("status") == "failed"
            for result in cls._tool_observations(state)
        )
        unmet_criteria = list(evaluation.get("unmet_criteria") or [])
        no_runnable_step = cls._next_executable_plan_step(state) is None
        if not has_failed_tool and not (unmet_criteria and no_runnable_step):
            return
        trigger = "execution_failure" if has_failed_tool else "criteria_unmet"
        state["replan_context"] = cls._build_replan_context(
            state,
            trigger=trigger,
        )
        state["replan_requested"] = True

    @staticmethod
    def _build_replan_context(
        state: AgentState,
        trigger: str = "execution_failure",
    ) -> dict[str, Any]:
        plan = AgentNodes._task_steps_only(
            [dict(step) for step in state.get("plan", [])]
        )
        observations = [dict(item) for item in state.get("observations", [])]
        failed_observations = [
            item
            for item in observations
            if item.get("status") == "failed"
        ]
        successful_observations = [
            item
            for item in observations
            if item.get("status") in {"success", "completed"}
        ]
        completed_steps = [
            step
            for step in plan
            if step.get("status") == "completed"
        ]
        remaining_steps = [
            step
            for step in plan
            if step.get("status") in {"pending", "running", "failed", "skipped"}
        ]
        return {
            "trigger": trigger,
            "attempt": int(state.get("replan_count") or 0) + 1,
            "original_goal": state.get("normalized_task") or state.get("task_desc") or "",
            "remaining_goal": AgentNodes._build_remaining_goal(
                state,
                failed_observations,
                successful_observations,
                remaining_steps,
            ),
            "previous_plan": plan,
            "completed_steps": completed_steps,
            "remaining_steps": remaining_steps,
            "failed_observations": failed_observations,
            "successful_observations": successful_observations,
            "task_evaluation": dict(state.get("task_evaluation") or {}),
            "error_info": state.get("error_info", ""),
        }

    @staticmethod
    def _build_remaining_goal(
        state: AgentState,
        failed_observations: list[dict[str, Any]],
        successful_observations: list[dict[str, Any]],
        remaining_steps: list[dict[str, Any]],
    ) -> str:
        original_goal = str(state.get("normalized_task") or state.get("task_desc") or "")
        if successful_observations:
            return (
                "Use completed observations to answer the original goal, and explain "
                f"any failed or skipped work. Original goal: {original_goal}"
            )
        if failed_observations:
            failed_names = [
                str(item.get("tool_name") or item.get("step_id") or "step")
                for item in failed_observations
            ]
            return (
                "Recover from failed execution without blindly repeating failed steps. "
                f"Failed steps: {', '.join(failed_names)}. Original goal: {original_goal}"
            )
        if remaining_steps:
            remaining_names = [
                str(item.get("name") or item.get("id") or "step")
                for item in remaining_steps
            ]
            return (
                "Finish the remaining planned work: "
                f"{', '.join(remaining_names)}. Original goal: {original_goal}"
            )
        return f"Finish the original goal: {original_goal}"

    @staticmethod
    def _tool_observations(state: AgentState) -> list[dict[str, Any]]:
        return [
            observation
            for observation in state.get("observations", [])
            if observation.get("type") == "tool_result"
        ]

    @staticmethod
    def _build_observation_summary(observations: list[dict[str, Any]]) -> str:
        relevant = [
            r for r in observations
            if r.get("type") == "tool_result"
            and r.get("status") in {"success", "failed"}
        ]
        return "\n\n".join(
            AgentNodes._format_observation_for_summary(r)
            for r in relevant
        )

    @staticmethod
    def _format_observation_for_summary(observation: dict[str, Any]) -> str:
        tool_name = str(observation.get("tool_name") or "unknown")
        status = str(observation.get("status") or "unknown")
        if status == "failed":
            error_msg = str(observation.get("error_msg") or "工具执行失败")
            return f"[{tool_name}:failed]\n{error_msg}"
        return f"[{tool_name}:success]\n{observation.get('content', '')}"

    @staticmethod
    def _build_tool_observation(
        step: dict[str, Any],
        tool_input: dict[str, Any],
        content: str,
        metadata: dict[str, Any],
        status: str,
        error_msg: str,
        start_time: float,
    ) -> dict[str, Any]:
        return {
            "type": "tool_result",
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

    @staticmethod
    def _dependency_failed(dep: str, tool_calls: list[dict[str, Any]]) -> bool:
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
    def _update_observation_status(state: AgentState) -> None:
        tool_observations = AgentNodes._tool_observations(state)
        failed = [
            r for r in tool_observations
            if r.get("status") == "failed"
        ]
        if failed:
            state["error_info"] = str(failed[-1].get("error_msg") or "")
        elif not state.get("error_info"):
            skipped = [
                s for s in state.get("tool_calls", [])
                if s.get("status") == "skipped"
            ]
            state["error_info"] = str(skipped[-1].get("error_msg") or "") if skipped else ""

    # ------------------------------------------------------------------
    # Tool result reuse
    # ------------------------------------------------------------------

    @classmethod
    def _tool_reuse_context(cls, state: AgentState) -> list[dict[str, Any]]:
        context: list[dict[str, Any]] = []
        for reference in state.get("resolved_references", []):
            if not isinstance(reference, dict):
                continue
            observation = reference.get("observation")
            if isinstance(observation, dict):
                context.append(dict(observation))
        context.extend(cls._recent_success_tool_observations(state))
        return cls._dedupe_observations(context)

    @classmethod
    def _find_reusable_observation(
        cls,
        tool_name: str,
        tool_input: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any] | None:
        if tool_name in _SIDE_EFFECT_TOOLS:
            return None

        current = cls._tool_observations(state)
        recent = cls._tool_reuse_context(state)
        for observation in reversed(current + recent):
            if not cls._observation_can_reuse(
                observation,
                tool_name,
                tool_input,
                state,
            ):
                continue
            return dict(observation)
        return None

    @classmethod
    def _observation_can_reuse(
        cls,
        observation: dict[str, Any],
        tool_name: str,
        tool_input: dict[str, Any],
        state: AgentState,
    ) -> bool:
        if observation.get("type") != "tool_result":
            return False
        if observation.get("status") != "success":
            return False
        if str(observation.get("tool_name") or "") != tool_name:
            return False

        previous_input = cls._safe_dict(observation.get("tool_input"))
        if cls._stable_json(previous_input) == cls._stable_json(tool_input):
            return True

        if tool_name == "weather" and cls._has_context_reference(
            str(state.get("task_desc") or "")
        ):
            return cls._weather_observation_matches(tool_input, observation)
        return False

    @staticmethod
    def _weather_observation_matches(
        tool_input: dict[str, Any],
        observation: dict[str, Any],
    ) -> bool:
        expected = str(
            tool_input.get("city")
            or tool_input.get("location")
            or tool_input.get("query")
            or ""
        ).strip()
        if not expected:
            return True
        previous_input = AgentNodes._safe_dict(observation.get("tool_input"))
        haystack = " ".join(
            [
                str(previous_input.get("city") or ""),
                str(previous_input.get("location") or ""),
                str(observation.get("content") or ""),
            ]
        )
        return expected in haystack

    @staticmethod
    def _stable_json(value: dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _dedupe_observations(
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for observation in observations:
            key = "|".join(
                [
                    str(observation.get("tool_name") or ""),
                    AgentNodes._stable_json(
                        AgentNodes._safe_dict(observation.get("tool_input"))
                    ),
                    str(observation.get("content") or "")[:120],
                ]
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(observation)
        return result

    @staticmethod
    def _build_reused_tool_observation(
        step: dict[str, Any],
        tool_input: dict[str, Any],
        source: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = dict(source.get("metadata") or {})
        metadata["reused"] = True
        metadata["reused_from_step_id"] = source.get("step_id", "")
        return {
            "type": "tool_result",
            "step_id": step.get("step_id") or step.get("id"),
            "phase": step.get("phase", "tools"),
            "name": step.get("name", ""),
            "goal": step.get("goal", ""),
            "depends_on": list(step.get("depends_on") or []),
            "tool_name": step.get("tool_name", ""),
            "tool_input": dict(source.get("tool_input") or tool_input),
            "content": str(source.get("content") or ""),
            "metadata": metadata,
            "status": "success",
            "error_msg": "",
            "duration_ms": 0,
            "reused": True,
        }

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
