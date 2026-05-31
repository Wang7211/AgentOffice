"""Agent 图节点实现。"""

import hashlib
import json
import time
from typing import Any

from loguru import logger

from agent.state import AgentState
from config.settings import get_settings
from memory.memory_context import MemoryContext
from memory.store import agent_memory
from services.llm_service import get_model_client
from services.tool_service import get_tool_registry
from utils.common import normalize_text
from utils.exception import ToolException


class AgentNodes:
    """理解、记忆、规划、工具、行动节点。"""

    def __init__(self) -> None:
        self._model_client = get_model_client()
        self._tool_registry = get_tool_registry()

    # ------------------------------------------------------------------
    # 前置记忆节点
    # ------------------------------------------------------------------

    def mem_pre_node(self, state: AgentState) -> AgentState:
        """前置记忆加载：检索与当前任务相关的长期记忆，注入 state 供意图理解和规划使用。"""
        query = state["task_desc"]
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
        state["memory_context"] = MemoryContext.build(
            messages=state.get("messages", []),
            relevant_memories=memories,
        )
        logger.debug("[记忆] mem_pre 加载 {} 条长期记忆", len(memories))
        return state

    # ------------------------------------------------------------------
    # 后置记忆节点
    # ------------------------------------------------------------------

    def mem_post_node(self, state: AgentState) -> AgentState:
        """后置记忆更新：保存本轮意图、参数和执行结果到长期记忆，为下一轮对话提供上下文。"""
        normalized_task = state.get("normalized_task") or state["task_desc"]
        intent = state.get("intent", {})
        tool_name = str(intent.get("tool_name") or "")
        tool_input = dict(intent.get("tool_input") or {})
        tool_result = state.get("tool_result", "")
        error_info = state.get("error_info", "")
        answer = state.get("answer", "")
        session_id = state.get("session_id", "")

        # 构造记忆文本
        memory_lines = [f"用户问题: {normalized_task}", f"意图: {tool_name or '无需工具'}"]
        if tool_input:
            memory_lines.append(f"参数: {json.dumps(tool_input, ensure_ascii=False)}")
        if tool_result:
            memory_lines.append(f"工具结果: {tool_result[:200]}")
        if error_info:
            memory_lines.append(f"错误: {error_info[:200]}")
        if answer:
            memory_lines.append(f"回答: {answer[:200]}")
        memory_text = "\n".join(memory_lines)

        memory_id = hashlib.md5(
            f"{session_id}{normalized_task}{time.time()}".encode()
        ).hexdigest()[:16]

        try:
            agent_memory.add_text(
                vector_id=f"mem_{memory_id}",
                text=memory_text,
                metadata={
                    "session_id": session_id,
                    "tool_name": tool_name,
                    "normalized_task": normalized_task,
                    "timestamp": time.time(),
                    "type": "agent_execution",
                },
            )
            logger.debug("[记忆] mem_post 已保存 (tool={})", tool_name or "none")
        except Exception as exc:
            logger.warning("[记忆] mem_post 保存失败: {}", exc)

        return state

    # ------------------------------------------------------------------
    # 意图理解节点
    # ------------------------------------------------------------------

    def understand_node(self, state: AgentState) -> AgentState:
        """识别任务意图，理解任务，提取规范化描述。"""
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
        """任务编排：基于意图生成工具执行计划（支持多步骤序列）。"""
        analysis = state["intent"]
        tool_name = str(analysis.get("tool_name") or "")
        tool_input = dict(analysis.get("tool_input") or {})

        if not tool_name:
            state["need_tool"] = False
            state["tool_calls"] = []
            return state

        # 生成执行计划（LLM 可产出多步骤，本地模型兜底单步）
        tool_specs = self._tool_registry.list_specs()
        specs_dict = [{"name": s.name, "description": s.description} for s in tool_specs]

        plan = self._model_client.create_plan(
            analysis=analysis,
            memories=state["relevant_memories"],
            tool_specs=specs_dict,
            memory_ctx=state.get("memory_context"),
        )

        tool_steps = [
            s for s in plan
            if s.get("tool_name") and s["tool_name"].strip()
        ]

        if tool_steps:
            state["need_tool"] = True
            state["tool_name"] = tool_steps[0]["tool_name"]
            state["tool_input"] = tool_steps[0].get("tool_input", {})
            state["tool_calls"] = [
                {
                    "tool_name": s["tool_name"],
                    "tool_input": s.get("tool_input", {}),
                    "status": "pending",
                }
                for s in tool_steps
            ]
        else:
            # 兜底：直接使用分析结果作为单步计划
            state["need_tool"] = True
            state["tool_calls"] = [
                {"tool_name": tool_name, "tool_input": tool_input, "status": "pending"},
            ]

        return state

    def tool_node(self, state: AgentState) -> AgentState:
        """纯工具执行：循环执行工具列表，无硬编码分支。"""
        if not state["need_tool"] or not state["tool_calls"]:
            return state

        completed: list[dict[str, Any]] = []
        for step in state["tool_calls"]:
            tool_name = step["tool_name"]
            tool_input = step.get("tool_input", {})
            tool_input = self._resolve_tool_input(tool_name, tool_input, completed)

            try:
                tool = self._tool_registry.get(tool_name)
                result = self._run_tool_via_langchain(tool_name, tool_input)
                if result is None:
                    result = tool.run(tool_input)
                content = normalize_text(result.content)
                entry = {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "content": content,
                    "metadata": dict(result.metadata or {}),
                    "status": "success",
                    "error_msg": "",
                }
                completed.append(entry)
                step["status"] = "completed"
                logger.info("[工具] {} 执行成功", tool_name)
            except (ToolException, ValueError) as exc:
                message = exc.message if isinstance(exc, ToolException) else str(exc)
                logger.warning("[工具] {} 执行失败: {}", tool_name, message)
                entry = {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "content": "",
                    "status": "failed",
                    "error_msg": message,
                }
                completed.append(entry)
                step["status"] = "failed"

        state["tool_results"] = completed
        state["step_count"] += 1

        # 汇总结果：合并所有成功工具的输出，避免多工具时后一个覆盖前一个
        successful = [r for r in completed if r["status"] == "success"]
        failed = [r for r in completed if r["status"] == "failed"]
        if successful:
            merged = "\n\n".join(
                f"[{r['tool_name']}]\n{r['content']}" for r in successful
            )
            state["tool_result"] = merged
        else:
            state["tool_result"] = ""
        state["error_info"] = failed[-1]["error_msg"] if failed else ""

        return state

    def action_node(self, state: AgentState) -> AgentState:
        """融合记忆和工具结果，生成最终回答。"""
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

    @staticmethod
    def _resolve_tool_input(
        tool_name: str,
        tool_input: dict[str, Any],
        completed: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """在执行前解析工具入参：将前序（同轮次）工具结果注入后续步骤。

        当前支持：
          - email.body 为简短标题时，自动填充已完成 weather 工具的数据。
          - email.to 为空时，从 subject/body 中提取邮箱地址。
        """
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
        """同轮次 weather→email 正文注入。"""
        body = tool_input.get("body", "")
        if not body:
            return tool_input
        if any(kw in body for kw in ("°C", "温度", "湿度", "风力")):
            return tool_input
        if len(body) >= 50:
            return tool_input
        weather_result = next(
            (r for r in completed
             if r.get("tool_name") == "weather" and r.get("status") == "success"),
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
        """email.to 为空时，从 subject/body 中提取邮箱地址。"""
        import re

        to_val = tool_input.get("to", "")
        if to_val and str(to_val).strip():
            return tool_input
        resolved = dict(tool_input)
        for field in ("subject", "body"):
            val = str(resolved.get(field, ""))
            match = re.search(
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", val,
            )
            if match:
                resolved["to"] = match.group(0)
                cleaned = val.replace(match.group(0), "").strip()
                resolved[field] = cleaned
                if not resolved.get("subject"):
                    resolved["subject"] = cleaned
                break
        return resolved
