"""Agent 图节点实现。"""

import json
import re
import time
from datetime import datetime
from datetime import timedelta
from typing import Any

from loguru import logger

from agent.state import AgentState
from config.settings import get_settings
from memory.store import agent_memory
from services.llm_service import get_model_client
from services.tool_service import get_tool_registry
from utils.common import normalize_text
from utils.common import text_hash
from utils.exception import ToolException
from utils.structured_log import log_agent_event
from utils.structured_log import preview_text


class AgentNodes:
    """任务理解、记忆、规划、工具、行动和反思归档节点。"""

    max_tool_steps = 3

    def __init__(self) -> None:
        self._model_client = get_model_client()
        self._tool_registry = get_tool_registry()
        self._langchain_tool_count = len(self._tool_registry.list_langchain_tools())
        log_agent_event(
            "langchain_tools_ready",
            count=self._langchain_tool_count,
        )

    def understand_node(self, state: AgentState) -> AgentState:
        """识别任务意图并抽取执行约束。"""
        start_time = time.perf_counter()
        analysis = self._model_client.analyze_task(
            message=state["task_desc"],
            context=state["messages"],
        )
        intent = dict(analysis.get("intent") or {})
        state["normalized_task"] = str(
            analysis.get("normalized_task") or state["task_desc"],
        )
        state["intent"] = analysis
        state["constraints"] = dict(analysis.get("constraints") or {})
        state["tool_name"] = str(intent.get("tool_name") or "")
        state["tool_input"] = dict(intent.get("tool_input") or {})
        log_agent_event(
            "task_understood",
            session_id=state["session_id"],
            intent_type=analysis.get("intent_type", "chat"),
            intent_category=analysis.get("intent_category", analysis.get("intent_type", "chat")),
            intent_subtype=analysis.get("intent_subtype", ""),
            confidence=analysis.get("confidence", 0),
            tool_name=state["tool_name"] or "direct_answer",
            constraints=state["constraints"],
            duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )
        return state

    def memory_node(self, state: AgentState) -> AgentState:
        """调取与当前任务相关的短期上下文和长期记忆。"""
        start_time = time.perf_counter()
        query = state["normalized_task"] or state["task_desc"]
        memories: list[dict[str, Any]] = []
        if query:
            settings = get_settings()
            for item in agent_memory.search_filtered(
                query=query,
                top_k=5,
                min_score=settings.agent_memory_similarity_threshold,
            ):
                score = float(item.get("score", 0))
                if score <= 0:
                    continue
                memories.append(
                    {
                        "score": round(score, 4),
                        "text": str(item.get("text") or ""),
                        "metadata": dict(item.get("metadata") or {}),
                    }
                )
        state["relevant_memories"] = memories
        log_agent_event(
            "memory_retrieved",
            session_id=state["session_id"],
            short_memory_count=len(state["messages"]),
            long_memory_count=len(memories),
            duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )
        return state

    def validate_node(self, state: AgentState) -> AgentState:
        """校验任务可行性并划定工具边界。"""
        start_time = time.perf_counter()
        available_tools = {spec["name"] for spec in self._tool_specs()}
        boundary = {
            "allowed_tools": sorted(available_tools),
            "max_tool_steps": self.max_tool_steps,
            "execution_mode": "function_call+browser+mcp_http",
            "rag_memory": "agent_memory_index",
        }
        state["boundary"] = boundary

        analysis = state["intent"]
        missing_capability = self._detect_missing_capability(state, self._tool_specs())
        if missing_capability:
            self._enter_capability_missing(state, missing_capability)
        elif analysis.get("needs_clarification"):
            self._enter_clarification(
                state,
                str(
                analysis.get("clarification_question")
                or "请补充更明确的任务目标或执行范围。",
                ),
                reason="task_analysis",
            )
        elif state["tool_name"] and state["tool_name"] not in available_tools:
            state["task_status"] = "blocked"
            state["error_info"] = f"工具未注册：{state['tool_name']}"
        else:
            state["task_status"] = "ready"

        log_agent_event(
            "task_validated",
            session_id=state["session_id"],
            task_status=state["task_status"],
            boundary=boundary,
            error=state["error_info"],
            duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )
        return state

    def planning_node(self, state: AgentState) -> AgentState:
        """拆分子目标、排序依赖并输出结构化执行计划。"""
        if state["task_status"] != "ready":
            log_agent_event(
                "planning_skipped",
                session_id=state["session_id"],
                task_status=state["task_status"],
            )
            return state

        start_time = time.perf_counter()

        if self._task_needs_clarification(state):
            self._enter_clarification(
                state,
                "当前任务缺少关键参数（如目的地、时间等），请补充后我再为你规划。",
                reason="vague_task",
            )
            return state

        state["plan"] = self._model_client.create_plan(
            analysis=state["intent"],
            memories=state["relevant_memories"],
            tool_specs=self._tool_specs(),
        )
        primary_tool = self._first_tool_step(state["plan"])
        if primary_tool:
            state["tool_name"] = str(primary_tool.get("tool_name") or "")
            state["tool_input"] = dict(primary_tool.get("tool_input") or {})
        if primary_tool and primary_tool.get("tool_name") == "search":
            self._enrich_search_query(state, primary_tool)
            state["tool_input"] = dict(primary_tool.get("tool_input") or {})
        clarification_step = self._first_clarification_step(state["plan"])
        if clarification_step:
            if not self._has_sufficient_context(state):
                self._enter_clarification(
                    state,
                    self._build_plan_clarification_question(state, clarification_step),
                    reason="planning_requires_clarification",
                )
                log_agent_event(
                    "clarification_required",
                    session_id=state["session_id"],
                    source="planning",
                    step_id=clarification_step.get("id", ""),
                    question=state["clarification_question"],
                )
            else:
                state["plan"] = [
                    s for s in state["plan"]
                    if s.get("id") != clarification_step.get("id")
                ]
        log_agent_event(
            "plan_created",
            session_id=state["session_id"],
            step_count=len(state["plan"]),
            tool_steps=[
                step.get("tool_name")
                for step in state["plan"]
                if step.get("tool_name")
            ],
            duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )
        return state

    def tool_node(self, state: AgentState) -> AgentState:
        """按计划执行函数调用工具并清洗融合返回数据。"""
        if state["task_status"] != "ready":
            return state

        tool_steps = [
            step for step in state["plan"] if str(step.get("tool_name") or "")
        ]
        if not tool_steps and state["tool_name"]:
            tool_steps = [
                {
                    "id": "tool_1",
                    "phase": "tools",
                    "name": f"调用 {state['tool_name']} 工具",
                    "goal": "获取回答信息问询所需的数据或资料。",
                    "tool_name": state["tool_name"],
                    "tool_input": state["tool_input"],
                    "depends_on": [],
                    "status": "pending",
                }
            ]
        if not tool_steps:
            log_agent_event(
                "tool_skipped",
                session_id=state["session_id"],
                reason="no_tool_required",
            )
            return state

        replan_attempted = False
        for step in tool_steps[: self.max_tool_steps]:
            if state["step_count"] >= self.max_tool_steps:
                state["error_info"] = "工具调用达到最大步数限制"
                break
            self._run_tool_step(state, step)

            if self._should_replan(state, step) and not replan_attempted:
                replan_attempted = True
                remaining = [
                    s for s in tool_steps if s.get("status") not in ("completed", "failed")
                ]
                new_plan_steps = self._attempt_replan(state, remaining)
                if new_plan_steps:
                    state["plan"].extend(new_plan_steps)
                    leftover_tools = [
                        s for s in new_plan_steps if s.get("tool_name")
                    ]
                    if leftover_tools and state["step_count"] < self.max_tool_steps:
                        tool_steps = leftover_tools
                        continue

        state["tool_result"] = self._fuse_tool_results(state["tool_results"])
        return state

    def action_node(self, state: AgentState) -> AgentState:
        """融合计划、记忆和工具结果，生成最终交付。"""
        if state["task_status"] == "needs_clarification":
            return state
        if state["task_status"] == "capability_missing":
            state["answer"] = self._build_capability_missing_answer(state)
            return state
        if state["task_status"] == "blocked":
            state["answer"] = f"任务无法执行：{state['error_info']}"
            return state
        if state["error_info"] and not state["tool_result"]:
            state["answer"] = f"执行过程中遇到问题：{state['error_info']}"
            state["task_status"] = "failed"
            return state
        if str(state.get("intent", {}).get("intent_subtype", "")) == "capability_question":
            state["answer"] = self._build_capability_overview_answer()
            state["task_status"] = "completed"
            log_agent_event(
                "capability_overview_answered",
                session_id=state["session_id"],
                answer_preview=preview_text(state["answer"]),
            )
            return state
        if str(state.get("intent", {}).get("intent_category", "")) == "interaction_chat":
            if self._is_suggestion_affirmative(state):
                state["answer"] = self._model_client.generate(
                    user_message=state["normalized_task"] or state["task_desc"],
                    context=state["messages"],
                    memories=state["relevant_memories"],
                )
                state["task_status"] = "completed"
            else:
                state["answer"] = self._build_interaction_chat_answer(state)
                state["task_status"] = "completed"
            log_agent_event(
                "interaction_chat_answered",
                session_id=state["session_id"],
                answer_preview=preview_text(state["answer"]),
            )
            return state

        start_time = time.perf_counter()
        log_agent_event(
            "action_started",
            session_id=state["session_id"],
            plan_count=len(state["plan"]),
            memory_count=len(state["relevant_memories"]),
            tool_result_count=len(state["tool_results"]),
        )
        state["answer"] = self._model_client.generate(
            user_message=state["normalized_task"] or state["task_desc"],
            tool_result=state["tool_result"] or None,
            context=state["messages"],
            memories=state["relevant_memories"],
            plan=state["plan"],
            tool_results=state["tool_results"],
        )
        state["task_status"] = "completed"
        self._add_proactive_suggestion(state)
        self._mark_plan_step(state["plan"], "action", "completed")
        log_agent_event(
            "action_finished",
            session_id=state["session_id"],
            status="success",
            answer_preview=preview_text(state["answer"]),
            duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )
        return state


    def _build_capability_overview_answer(self) -> str:
        """回答系统能力边界。"""
        return (
            "我可以处理三类请求：\n\n"
            "1. 信息问询：解释概念、查询资料、读取文档、检索知识库、获取时间或公开信息。\n"
            "2. 任务执行：生成报告、整理文档、编写或分析代码、做数据统计、拆解流程并调用可用工具。\n"
            "3. 交互闲聊：简单问候、确认上下文、日常对话。\n\n"
            "当前已接入内部工具：时间、搜索、浏览器、文件、知识库和代码执行。"
            "对于机票、酒店、高铁票这类实时库存/价格/预订任务，如果没有接入对应 MCP 或专用工具，"
            "我会明确提示能力缺失，不会用普通搜索冒充实时结果。"
        )

    def _is_suggestion_affirmative(self, state: AgentState) -> bool:
        """判断极短消息是否是对上一条建议的肯定回应。"""
        message = (state["normalized_task"] or state["task_desc"]).strip()
        if len(message) > 4 or message not in ("需要", "好的", "可以", "好", "是", "嗯", "要"):
            return False
        for msg in reversed(state.get("messages", [])[-6:]):
            if msg.get("role") == "assistant":
                content = str(msg.get("content", ""))
                if any(kw in content for kw in ("是否", "需要我", "建议", "要不要")):
                    return True
                break
        return False

    def _build_interaction_chat_answer(self, state: AgentState) -> str:
        """为无业务诉求的闲聊生成轻量回应。"""
        message = state["normalized_task"] or state["task_desc"]
        normalized = message.strip()
        if str(state.get("intent", {}).get("intent_subtype", "")) == "light_entertainment" or any(
            keyword in normalized for keyword in ("笑话", "段子", "逗我笑")
        ):
            return self._build_light_entertainment_answer(state)
        if any(keyword in normalized for keyword in ("谢谢", "感谢", "辛苦")):
            return "不客气。"
        if any(keyword in normalized for keyword in ("再见", "拜拜")):
            return "再见。"
        if any(keyword in normalized.lower() for keyword in ("hi", "hello", "你好", "您好", "在吗")):
            return (
                "你好，我是 AgentOffice，一个面向办公和知识处理的智能体。\n\n"
                "我可以帮你做三类事情：信息问询、任务执行和简单对话。"
                "比如解释概念、查询资料、读取文档、检索知识库、生成报告、整理内容、"
                "分析代码或调用已接入的工具。涉及机票、酒店、高铁票这类实时库存/预订任务时，"
                "如果没有接入对应专用工具或 MCP，我会明确说明能力边界。"
            )
        if state.get("messages"):
            response = self._model_client.generate(
                user_message=normalized,
                context=state["messages"],
                memories=state["relevant_memories"],
            )
            if response:
                return response
        return "请继续。"

    def _build_light_entertainment_answer(self, state: AgentState) -> str:
        """结合短期上下文返回轻量娱乐内容，并避免马上重复。"""
        history_text = "\n".join(
            str(message.get("content") or "") for message in state["messages"][-8:]
        )
        jokes = [
            (
                "讲一个办公场景的：\n\n"
                "同事问我：“为什么你的待办事项每天都做不完？”\n"
                "我说：“因为它们太懂团队协作了，每完成一个，就自动拉两个新任务进群。”"
            ),
            (
                "换一个：\n\n"
                "产品经理说：“这个需求很简单。”\n"
                "开发问：“简单到什么程度？”\n"
                "产品经理说：“简单到我已经在脑子里上线了。”"
            ),
            (
                "再来一个：\n\n"
                "测试同事说：“这个 bug 很稳定。”\n"
                "大家松了一口气。\n"
                "他接着说：“每次演示的时候都会出现。”"
            ),
            (
                "讲个轻松的：\n\n"
                "我问日程表：“今天能不能少安排一点？”\n"
                "日程表说：“可以，我已经把明天也安排满了。”"
            ),
        ]
        for joke in jokes:
            first_line = joke.splitlines()[0]
            if first_line not in history_text and joke not in history_text:
                return joke
        return jokes[0]

    def reflection_node(self, state: AgentState) -> AgentState:
        """复盘执行效果，记录缺陷和优化方向。"""
        if state["task_status"] in {
            "needs_clarification",
            "capability_missing",
            "blocked",
        }:
            log_agent_event(
                "reflection_skipped",
                session_id=state["session_id"],
                reason=state["task_status"],
            )
            return state
        if not state["answer"]:
            return state
        start_time = time.perf_counter()
        state["reflection"] = self._model_client.reflect(
            user_message=state["normalized_task"] or state["task_desc"],
            answer=state["answer"],
            plan=state["plan"],
            tool_results=state["tool_results"],
            error_info=state["error_info"],
        )
        self._mark_plan_step(state["plan"], "reflection", "completed")
        if self._should_rollback_from_reflection(state):
            self._enter_clarification(
                state,
                self._build_reflection_clarification_question(state),
                reason="reflection_rollback",
            )
            state["reflection"]["rollback_applied"] = True
            state["reflection_retry_count"] += 1
            log_agent_event(
                "reflection_rollback",
                session_id=state["session_id"],
                question=state["clarification_question"],
                reflection=state["reflection"],
            )
        log_agent_event(
            "reflection_finished",
            session_id=state["session_id"],
            reflection=state["reflection"],
            duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )
        return state

    def archive_node(self, state: AgentState) -> AgentState:
        """将高价值经验写入长期向量记忆。"""
        if state["task_status"] != "completed":
            log_agent_event(
                "memory_archive_skipped",
                session_id=state["session_id"],
                task_status=state["task_status"],
            )
            return state

        reflection = state.get("reflection") or {}
        reflection_score = float(reflection.get("score") or 0)
        has_tool_calls = bool(state["tool_results"])
        answer_length = len(state["answer"])
        if reflection_score < 0.6 and (not has_tool_calls or answer_length < 50):
            log_agent_event(
                "memory_archive_skipped",
                session_id=state["session_id"],
                reason=f"low_value: score={reflection_score}, tools={has_tool_calls}, len={answer_length}",
            )
            return state

        start_time = time.perf_counter()
        memory_items = self._model_client.extract_memory(
            user_message=state["normalized_task"] or state["task_desc"],
            answer=state["answer"],
            reflection=state["reflection"],
        )
        archived_ids: list[str] = []
        for index, memory_text in enumerate(memory_items, start=1):
            vector_id = (
                f"agent_memory_{state['session_id']}_"
                f"{text_hash(memory_text)[:16]}_{index}"
            )
            agent_memory.add_text(
                vector_id=vector_id,
                text=memory_text,
                metadata={
                    "memory_type": "agent_task_experience",
                    "session_id": state["session_id"],
                    "source": "reflection_archive",
                },
            )
            archived_ids.append(vector_id)
        state["archived_memory_ids"] = archived_ids
        log_agent_event(
            "memory_archived",
            session_id=state["session_id"],
            archive_count=len(archived_ids),
            duration_ms=round((time.perf_counter() - start_time) * 1000, 2),
        )
        return state

    def intent_node(self, state: AgentState) -> AgentState:
        """兼容旧调用名，等价于任务理解节点。"""
        return self.understand_node(state)

    def summary_node(self, state: AgentState) -> AgentState:
        """兼容旧调用名，等价于行动生成节点。"""
        return self.action_node(state)

    def _run_tool_step(self, state: AgentState, step: dict[str, Any]) -> None:
        """执行单个工具步骤并记录结构化调用结果。"""
        tool_name = str(step.get("tool_name") or "")
        tool_input = dict(step.get("tool_input") or {})
        tool_input = self._resolve_template_vars(
            tool_input, state["tool_results"],
        )
        state["tool_name"] = tool_name
        state["tool_input"] = tool_input
        start_time = time.perf_counter()
        log_agent_event(
            "tool_started",
            session_id=state["session_id"],
            step_id=step.get("id", ""),
            tool_name=tool_name,
            tool_input=tool_input,
            route="function_call",
            protocol="mcp_compatible",
        )
        try:
            tool = self._tool_registry.get(tool_name)
            result = self._run_tool_via_langchain(tool_name, tool_input)
            if result is None:
                result = tool.run(tool_input)
            content = normalize_text(result.content)
            call_result = {
                "step_id": step.get("id", ""),
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_result": content,
                "content": content,
                "metadata": result.metadata,
                "status": "success",
                "error_msg": "",
                "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }
            state["tool_results"].append(call_result)
            state["tool_calls"].append(call_result)
            step["status"] = "completed"
            log_agent_event(
                "tool_finished",
                session_id=state["session_id"],
                step_id=step.get("id", ""),
                tool_name=tool_name,
                status="success",
                result_preview=preview_text(content),
                metadata=result.metadata,
                duration_ms=call_result["duration_ms"],
            )
        except (ToolException, ValueError) as exc:
            message = exc.message if isinstance(exc, ToolException) else str(exc)
            logger.warning("tool failed: {}", message)
            state["error_info"] = message
            call_result = {
                "step_id": step.get("id", ""),
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_result": "",
                "content": "",
                "metadata": {},
                "status": "failed",
                "error_msg": message,
                "duration_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }
            state["tool_results"].append(call_result)
            state["tool_calls"].append(call_result)
            step["status"] = "failed"
            log_agent_event(
                "tool_finished",
                session_id=state["session_id"],
                step_id=step.get("id", ""),
                tool_name=tool_name,
                status="failed",
                error=message,
                duration_ms=call_result["duration_ms"],
            )
        finally:
            state["step_count"] += 1

    def _fuse_tool_results(self, tool_results: list[dict[str, Any]]) -> str:
        """清洗并融合多个工具的有效返回数据。"""
        fused_results: list[str] = []
        for index, result in enumerate(tool_results, start=1):
            if result.get("status") != "success":
                continue
            content = str(result.get("content") or "").strip()
            if not content:
                continue
            fused_results.append(
                f"[{index}] {result.get('tool_name', 'tool')}\n{content}",
            )
        return "\n\n".join(fused_results)

    def _should_replan(self, state: AgentState, step: dict[str, Any]) -> bool:
        """判断上一步工具结果是否表明需要重规划。"""
        if state["reflection_retry_count"] > 0:
            return False
        if step.get("status") != "completed":
            return False
        step_result = None
        for r in state["tool_results"]:
            if r.get("step_id") == step.get("id"):
                step_result = r
                break
        if not step_result:
            return False
        if step_result.get("status") != "success":
            return True
        content = str(step_result.get("content") or "").strip()
        tool_name = str(step.get("tool_name") or "")
        if not content:
            return True
        if tool_name == "search":
            raw_input = str(step.get("tool_input", {}))
            if "{{" in raw_input:
                return True
        return False

    def _attempt_replan(
        self,
        state: AgentState,
        remaining: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """尝试对剩余步骤进行重规划，失败时返回 None。"""
        if not remaining:
            return None
        from services.llm_service import get_model_client
        model_client = get_model_client()
        try:
            new_plan = model_client.create_plan(
                analysis=state["intent"],
                memories=state["relevant_memories"],
                tool_specs=self._tool_specs(),
                tool_results=state["tool_results"],
            )
            new_tool_steps = [
                s for s in new_plan
                if s.get("tool_name") and s.get("status") != "completed"
            ]
            if new_tool_steps:
                log_agent_event(
                    "replan_created",
                    session_id=state["session_id"],
                    new_step_count=len(new_tool_steps),
                )
                return new_tool_steps
        except Exception:
            log_agent_event(
                "replan_failed",
                session_id=state["session_id"],
            )
        return None

    def _resolve_template_vars(
        self,
        tool_input: dict[str, Any],
        tool_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """将 tool_input 中的 {{step_X_result}} 替换为实际工具输出。"""
        step_map: dict[str, str] = {}
        for r in tool_results:
            sid = str(r.get("step_id") or "")
            content = str(r.get("content") or "")[:200]
            if sid and content:
                step_map[sid] = content
        def _replace_var(m: re.Match) -> str:
            var = m.group(1).strip()
            return step_map.get(var, m.group(0))

        resolved: dict[str, Any] = {}
        for key, value in tool_input.items():
            if isinstance(value, str):
                resolved[key] = re.sub(r"\{\{(.+?)\}\}", _replace_var, value)
            else:
                resolved[key] = value
        return resolved

    def _run_tool_via_langchain(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> Any | None:
        """通过 LangChain StructuredTool 执行工具。"""
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

    def _tool_specs(self) -> list[dict[str, Any]]:
        """返回工具能力标签和入参规范。"""
        return [spec.__dict__ for spec in self._tool_registry.list_specs()]

    def _detect_missing_capability(
        self,
        state: AgentState,
        tool_specs: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """识别需要专用实时能力但当前工具集无法可靠完成的任务。"""
        message = f"{state['normalized_task']} {state['task_desc']}".strip()
        profiles = (
            {
                "resource_type": "train_ticket",
                "resource_label": "高铁/火车票",
                "resource_keywords": (
                    "高铁票",
                    "火车票",
                    "动车票",
                    "列车票",
                    "12306",
                    "铁路票",
                ),
                "query_keywords": (
                    "查",
                    "查询",
                    "看一下",
                    "看看",
                    "买",
                    "订",
                    "预订",
                    "余票",
                    "票价",
                    "车次",
                ),
                "specialized_keywords": (
                    "12306",
                    "train_ticket",
                    "railway",
                    "rail",
                    "ticket",
                    "火车票",
                    "高铁",
                    "铁路",
                    "余票",
                    "票务",
                ),
                "required_tool": "实时库存/价格/预订工具或对应 MCP（如 12306/铁路票务 MCP）",
                "reason": "当前仅有 search/browser 等通用工具，不能可靠查询 12306 实时车次、票价和余票。",
                "actions": ["query_inventory", "query_price"],
                "needs_route": True,
                "needs_time_preference": False,
            },
            {
                "resource_type": "flight",
                "resource_label": "航班/机票",
                "resource_keywords": (
                    "机票",
                    "航班",
                    "飞机",
                    "航空",
                    "航旅",
                    "舱位",
                ),
                "query_keywords": (
                    "查",
                    "查询",
                    "看一下",
                    "看看",
                    "买",
                    "订",
                    "预订",
                    "预定",
                    "票价",
                    "余票",
                    "上午",
                    "下午",
                    "晚上",
                ),
                "specialized_keywords": (
                    "flight",
                    "air",
                    "airline",
                    "aviation",
                    "trip",
                    "机票",
                    "航班",
                    "航空",
                    "航旅",
                    "飞机",
                    "舱位",
                ),
                "required_tool": "实时库存/价格/预订工具或对应 MCP（如航旅/机票 MCP）",
                "reason": "当前仅有 search/browser 等通用工具，不能可靠查询实时航班、票价、舱位余量，也不能代为预订机票。",
                "actions": ["query_inventory", "query_price", "book"],
                "needs_route": True,
                "needs_time_preference": True,
            },
            {
                "resource_type": "hotel",
                "resource_label": "酒店/住宿",
                "resource_keywords": (
                    "酒店",
                    "住宿",
                    "房间",
                    "房型",
                    "民宿",
                    "客房",
                ),
                "query_keywords": (
                    "查",
                    "查询",
                    "看一下",
                    "看看",
                    "订",
                    "预订",
                    "预定",
                    "入住",
                    "房态",
                    "房价",
                    "价格",
                    "可订",
                ),
                "specialized_keywords": (
                    "hotel",
                    "lodging",
                    "booking",
                    "travel",
                    "酒店",
                    "住宿",
                    "房态",
                    "房价",
                    "订房",
                    "预订",
                    "酒旅",
                ),
                "required_tool": "实时库存/价格/预订工具或对应 MCP（如酒店/酒旅 MCP）",
                "reason": "当前仅有 search/browser 等通用工具，不能可靠查询实时房态、房价、可订房型，也不能代为预订酒店。",
                "actions": ["query_inventory", "query_price", "book"],
                "needs_route": False,
                "needs_time_preference": False,
            },
        )

        for profile in profiles:
            if not self._matches_realtime_inventory_request(
                message,
                resource_keywords=profile["resource_keywords"],
                query_keywords=profile["query_keywords"],
            ):
                continue
            if self._has_specialized_tool(
                tool_specs,
                keywords=profile["specialized_keywords"],
            ):
                return None
            missing_capability = {
                "capability": "realtime_inventory_booking",
                "domain": "travel",
                "resource_type": profile["resource_type"],
                "resource_label": profile["resource_label"],
                "required_tool": profile["required_tool"],
                "reason": profile["reason"],
                "travel_date": self._extract_relative_date_text(message),
                "actions": profile["actions"],
            }
            if profile["needs_route"]:
                origin, destination = self._extract_route(message)
                missing_capability["origin"] = origin
                missing_capability["destination"] = destination
            else:
                missing_capability["destination"] = self._extract_destination_hint(message)
            if profile["needs_time_preference"]:
                missing_capability["time_preference"] = self._extract_time_preference(
                    message,
                )
            return missing_capability
        return None

    def _matches_realtime_inventory_request(
        self,
        message: str,
        resource_keywords: tuple[str, ...],
        query_keywords: tuple[str, ...],
    ) -> bool:
        """判断是否命中统一的实时库存/价格/预订类请求。"""
        return any(keyword in message for keyword in resource_keywords) and any(
            keyword in message for keyword in query_keywords
        )

    def _has_specialized_tool(
        self,
        tool_specs: list[dict[str, Any]],
        keywords: tuple[str, ...],
    ) -> bool:
        """判断工具集中是否存在专用领域工具。"""
        generic_tools = {"search", "browser", "time", "code", "file", "knowledge"}
        for spec in tool_specs:
            tool_name = str(spec.get("name") or "").lower()
            if tool_name in generic_tools:
                continue
            description = str(spec.get("description") or "").lower()
            searchable = f"{tool_name} {description}"
            if any(keyword.lower() in searchable for keyword in keywords):
                return True
        return False

    def _extract_route(self, message: str) -> tuple[str, str]:
        """从口语化出行请求中提取出发地和目的地。"""
        match = re.search(r"从(?P<origin>[^到\s，。；,;]{1,20})到(?P<dest>[^的\s，。；,;]{1,20})", message)
        if not match:
            return "", ""
        return match.group("origin"), match.group("dest")

    def _extract_destination_hint(self, message: str) -> str:
        """提取酒店等单目的地任务中的位置提示。"""
        match = re.search(r"(?:在|去|到)(?P<dest>[^住订查看\s，。；,;]{1,20})", message)
        if match:
            return match.group("dest")
        for city in ("北京", "上海", "广州", "深圳", "西安", "成都", "杭州", "南京"):
            if city in message:
                return city
        return ""

    def _extract_relative_date_text(self, message: str) -> str:
        """提取常见相对日期并给出可读日期。"""
        if "明天" in message:
            target_date = datetime.now().date() + timedelta(days=1)
            return f"明天（{target_date.isoformat()}）"
        if "今天" in message:
            return f"今天（{datetime.now().date().isoformat()}）"
        if "后天" in message:
            target_date = datetime.now().date() + timedelta(days=2)
            return f"后天（{target_date.isoformat()}）"
        return ""

    def _extract_time_preference(self, message: str) -> str:
        """提取出行时间偏好。"""
        for keyword in ("凌晨", "早上", "上午", "中午", "下午", "晚上", "夜间"):
            if keyword in message:
                return keyword
        return ""

    def _enter_capability_missing(
        self,
        state: AgentState,
        missing_capability: dict[str, Any],
    ) -> None:
        """进入能力缺失状态，避免被误判为用户信息缺失。"""
        state["task_status"] = "capability_missing"
        state["error_info"] = ""
        state["tool_name"] = ""
        state["tool_input"] = {}
        state["boundary"]["missing_capability"] = missing_capability
        log_agent_event(
            "capability_missing",
            session_id=state["session_id"],
            capability=missing_capability,
        )

    def _build_capability_missing_answer(self, state: AgentState) -> str:
        """生成面向用户的能力边界说明。"""
        missing_capability = dict(state["boundary"].get("missing_capability") or {})
        if missing_capability.get("capability") == "realtime_inventory_booking":
            route_parts = []
            if missing_capability.get("origin"):
                route_parts.append(f"出发地：{missing_capability['origin']}")
            if missing_capability.get("destination"):
                route_parts.append(f"目的地：{missing_capability['destination']}")
            if missing_capability.get("travel_date"):
                route_parts.append(f"日期：{missing_capability['travel_date']}")
            if missing_capability.get("time_preference"):
                route_parts.append(f"时间偏好：{missing_capability['time_preference']}")
            resource_type = str(missing_capability.get("resource_type") or "")
            resource_label = str(
                missing_capability.get("resource_label") or "实时库存资源",
            )
            route_text = "，".join(route_parts) or f"你的{resource_label}查询/预订需求"
            examples = {
                "train_ticket": "车次、出发到达时间、历时、席别余票和票价",
                "flight": "航班号、起降时间、机场、价格、舱位和预订步骤",
                "hotel": "可订酒店、房型、房态、价格和预订步骤",
            }
            fallback_channel = {
                "train_ticket": "中国铁路 12306 官网或 App",
                "flight": "航司官网、携程、飞猪、同程等正规渠道",
                "hotel": "酒店官网或正规酒旅平台",
            }
            details = examples.get(resource_type, "实时库存、价格、可用状态和预订步骤")
            channel = fallback_channel.get(resource_type, "对应官网或正规预订平台")
            alt_suggestions = {
                "train_ticket": (
                    "其他我可以帮你的：查询火车时刻表、计算出行时间、"
                    "整理行程清单、或查看目的地天气。"
                ),
                "flight": (
                    "其他我可以帮你的：查询航班时刻表、计算飞行时间、"
                    "整理出行清单、或查看目的地天气。"
                ),
                "hotel": (
                    "其他我可以帮你的：整理住宿需求清单、"
                    "查询目的地景点信息、或规划行程路线。"
                ),
            }
            alt = alt_suggestions.get(resource_type, "")
            return (
                f"我已识别到你要处理{route_text}。\n\n"
                "当前系统还没有接入实时库存/价格/预订工具或对应 MCP，"
                f"所以不能可靠返回{resource_label}的实时可用状态、价格或余量，也不能代为完成预订。"
                "继续用普通搜索只能得到入口、介绍或过期参考信息，不应当当作实时库存结果交付。\n\n"
                f"你可以先通过{channel}查询；如果后续接入对应 MCP/专用工具，"
                f"我可以按你的条件自动查询并整理{details}。\n\n"
                f"{alt}"
            )
        return (
            "当前任务需要专用实时外部能力，但系统尚未接入对应工具。"
            f"{missing_capability.get('reason') or ''}"
        )

    def _first_tool_step(self, plan: list[dict[str, Any]]) -> dict[str, Any] | None:
        """找到计划中的第一个工具步骤。"""
        for step in plan:
            if step.get("tool_name"):
                return step
        return None

    def _enrich_search_query(
        self,
        state: AgentState,
        step: dict[str, Any],
    ) -> None:
        """当搜索 query 缺少对话历史中已有关键上下文时自动补全。"""
        tool_input = dict(step.get("tool_input") or {})
        query = str(tool_input.get("query") or "").strip()
        if not query:
            return
        context_text = " ".join(
            str(m.get("content") or "")
            for m in (state.get("messages") or [])[-8:]
        )
        city_terms = (
            "北京", "上海", "广州", "深圳", "西安", "成都",
            "杭州", "南京", "武汉", "重庆", "苏州", "天津",
        )
        if any(term in query for term in city_terms):
            return
        context_term = next((t for t in city_terms if t in context_text), "")
        if context_term:
            tool_input["query"] = f"{context_term} {query}"
            step["tool_input"] = tool_input

    def _task_needs_clarification(self, state: AgentState) -> bool:
        """检查当前用户消息自身（不依赖历史上下文）是否缺少关键参数。"""
        task = (state.get("task_desc") or "").strip()
        intent = state.get("intent", {})
        intent_category = str(
            intent.get("intent_category") or intent.get("intent_type") or ""
        )
        if intent_category != "task_execution":
            return False
        location_keywords = (
            "北京", "西安", "上海", "广州", "深圳", "成都", "杭州", "南京",
            "苏州", "天津", "重庆", "武汉",
            "国博", "国家博物馆", "故宫", "博物院",
        )
        if any(kw in task for kw in location_keywords):
            return False
        vague_planning = ("规划", "路线", "行程", "安排路线", "计划路线")
        if any(kw in task for kw in vague_planning) and len(task) < 20:
            return True
        return False

    def _has_sufficient_context(self, state: AgentState) -> bool:
        """判断对话历史是否已为当前任务提供足够上下文跳过澄清。"""
        context_text = " ".join(
            str(m.get("content") or "")
            for m in (state.get("messages") or [])[-8:]
        )
        if not context_text.strip():
            return False
        task_text = state.get("task_desc") or state.get("normalized_task") or ""
        combined = f"{context_text} {task_text}"
        specific_keywords = (
            "北京", "西安", "上海", "广州", "深圳", "成都", "杭州", "南京",
            "苏州", "天津", "重庆", "武汉",
            "国博", "国家博物馆",
            "故宫", "博物院",
            "旅游", "旅行", "景点", "门票",
        )
        return any(kw in combined for kw in specific_keywords)

    def _first_clarification_step(
        self,
        plan: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """找到计划中未完成的澄清步骤。"""
        for step in plan:
            text = " ".join(
                str(step.get(key) or "")
                for key in ("id", "phase", "name", "goal", "status")
            )
            if step.get("status") == "completed":
                continue
            if any(keyword in text for keyword in ("澄清", "确认", "补充", "clarify")):
                return step
        return None

    def _enter_clarification(
        self,
        state: AgentState,
        question: str,
        reason: str,
    ) -> None:
        """进入澄清状态并设置面向用户的问题。"""
        state["task_status"] = "needs_clarification"
        state["clarification_question"] = question
        state["answer"] = question
        state["error_info"] = ""
        log_agent_event(
            "clarification_entered",
            session_id=state["session_id"],
            reason=reason,
            question=question,
        )

    def _build_plan_clarification_question(
        self,
        state: AgentState,
        clarification_step: dict[str, Any],
    ) -> str:
        """根据规划中的澄清步骤生成用户问题。"""
        planned_goal = str(clarification_step.get("goal") or "").strip()
        if planned_goal:
            return f"继续执行前需要先确认：{planned_goal}"
        analysis_question = str(state["intent"].get("clarification_question") or "")
        return analysis_question or "继续执行前需要补充任务目标、范围或输出格式。"

    def _should_rollback_from_reflection(self, state: AgentState) -> bool:
        """判断反思结果是否需要回退到澄清。"""
        if state["reflection_retry_count"] > 0:
            return False
        reflection = state["reflection"]
        if not reflection:
            return False
        issues = "；".join(str(issue) for issue in reflection.get("issues", []))
        retry_reason = str(reflection.get("retry_reason") or "")
        text = f"{issues}；{retry_reason}"
        hard_keywords = (
            "澄清",
            "未明确",
            "假设",
            "缺少",
            "不匹配",
            "无关",
            "知识库",
            "未命中",
            "检索失效",
            "pending",
            "日期",
            "参数",
        )
        has_hard_issue = any(keyword in text for keyword in hard_keywords)
        if reflection.get("requires_retry") and has_hard_issue:
            return True
        score = float(reflection.get("score") or 1)
        status = str(reflection.get("status") or "")
        return score < 0.75 and status in {"partial_success", "failed"} and has_hard_issue

    def _build_reflection_clarification_question(self, state: AgentState) -> str:
        """将反思发现的问题转化为下一轮澄清问题。"""
        reflection = state["reflection"]
        issues = [str(issue) for issue in reflection.get("issues", [])]
        issue_text = "；".join(issues)
        if any(keyword in issue_text for keyword in ("假设", "澄清", "未明确", "缺少")):
            question = "我不应继续基于假设交付。请补充任务主题、适用场景、关键对象和期望输出格式。"
        elif any(keyword in issue_text for keyword in ("日期", "时间", "offset_days")):
            question = "请确认本次任务的目标日期或时间范围，我会据此重新规划。"
        elif any(
            keyword in issue_text
            for keyword in (
                "工具调用失败",
                "联网搜索请求失败",
                "无法连接",
                "外部依赖",
            )
        ):
            question = (
                "当前失败来自外部工具或网络依赖，不是用户任务描述缺失。"
                "请检查搜索服务、网络连接或对应 API 配置后重试；"
                "在工具不可用时，我只能基于已有上下文输出非实时的框架性分析。"
            )
        elif any(keyword in issue_text for keyword in ("知识库", "RAG", "无关", "未命中", "检索失败", "模板")):
            question = (
                "当前知识库没有命中合适资料，且已过滤无关文档。"
                "请确认是否跳过知识库并仅基于公开资料生成框架报告，"
                "或先上传企业供应链、碳足迹、供应商、LCA/ESG 等相关文档后再分析。"
            )
        elif any(
            keyword in issue_text
            for keyword in (
                "企业内部供应链数据",
                "实际碳足迹",
                "碳足迹基线",
                "初始碳排",
                "供应链数据",
            )
        ):
            question = (
                "要完成精确合规差距分析，需要企业现有电池供应链和碳足迹基线数据。"
                "请提供电芯/正负极/隔膜/电解液/Pack 供应商、产地、能耗、运输、LCA 或 ESG 数据，"
                "或确认允许我仅基于行业公开信息输出假设版路径报告。"
            )
        else:
            question = str(state["intent"].get("clarification_question") or "")
            question = question or "反思发现当前结果不够可靠，请补充关键约束后我再重新执行。"
        if issues:
            return f"{question}\n\n本次回退原因：{issues[0]}"
        return question

    def _mark_plan_step(
        self,
        plan: list[dict[str, Any]],
        step_id: str,
        status: str,
    ) -> None:
        """按步骤 id 更新计划状态。"""
        for step in plan:
            if step.get("id") == step_id:
                step["status"] = status
                return

    def _add_proactive_suggestion(self, state: AgentState) -> None:
        """在任务完成后追加一条主动建议。"""
        if state["task_status"] != "completed":
            return
        intent_category = str(state.get("intent", {}).get("intent_category", ""))
        if intent_category in ("interaction_chat",):
            return
        has_real_work = bool(state["tool_results"]) or len(state["answer"]) > 60
        if not has_real_work:
            return
        answer_tail = state["answer"][-100:] if len(state["answer"]) > 100 else state["answer"]
        if any(marker in answer_tail for marker in ("建议", "需要我", "要不要", "我可以帮你")):
            return
        task_desc = state["normalized_task"] or state["task_desc"]
        suggestions = []
        if any(kw in task_desc for kw in ("旅游", "旅行", "景点", "门票")):
            suggestions.append("如果需要，我可以帮你把以上信息整理成详细的日程表文档。")
        elif any(kw in task_desc for kw in ("查", "搜索", "找一下")) and not any(
            kw in task_desc for kw in ("生成", "写", "做", "创建", "整理")
        ):
            suggestions.append("如果需要进一步分析或整理这些信息，请告诉我。")
        if suggestions:
            state["answer"] += "\n\n" + suggestions[0]
