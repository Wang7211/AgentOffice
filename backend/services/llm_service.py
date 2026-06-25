"""Model client abstraction for understanding, planning, and response generation."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import httpx
from loguru import logger

from config.settings import get_settings
from memory.store import redis_kv
from schemas.agent_contract import filter_task_steps
from schemas.agent_contract import normalize_depends_on
from schemas.agent_contract import safe_dict

_LLM_CACHE_TTL = 300

SemanticUnderstanding = dict[str, Any]
TaskAnalysis = dict[str, Any]
PlanStep = dict[str, Any]


class PlanningError(RuntimeError):
    """Raised when no LLM can produce a valid executable plan."""


class LocalModelClient:
    """Deterministic local model for non-planning fallback behavior."""

    def generate(
        self,
        user_message: str,
        observation_summary: str | None = None,
        context: list[dict[str, str]] | None = None,
        memories: list[dict[str, Any]] | None = None,
        plan: list[PlanStep] | None = None,
        observations: list[dict[str, Any]] | None = None,
    ) -> str:
        _ = memories
        _ = plan
        if observations and not observation_summary:
            observation_summary = self._summarize_observations(observations)
        if observation_summary:
            return self._answer_with_observations(user_message, observation_summary)
        return self._answer_directly(user_message, context or [])

    def evaluate_completion_quality(
        self,
        task_contract: dict[str, Any],
        answer: str,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Provide a deterministic fallback for semantic completion checks."""
        _ = task_contract
        _ = observations
        satisfied = bool(str(answer or "").strip())
        return {
            "satisfied": satisfied,
            "reason": (
                "nonempty_answer_fallback"
                if satisfied
                else "empty_answer_fallback"
            ),
        }

    def analyze_task(
        self,
        message: str,
        context: list[dict[str, str]] | None = None,
        memories: list[dict[str, Any]] | None = None,
        memory_ctx: Any = None,
    ) -> TaskAnalysis:
        _ = context
        normalized_message = self._normalize_message(message)
        base = self._build_base_understanding(normalized_message, memories, memory_ctx)
        layer1_facts = self._layer1_semantic_facts(normalized_message)
        if layer1_facts:
            base["semantic_facts"] = layer1_facts
            base["understanding_source"] = "layer1_keyword"
        else:
            interpreted = self.interpret_semantics(
                normalized_message,
                memories=memories,
                memory_ctx=memory_ctx,
            )
            if not isinstance(interpreted, dict):
                interpreted = {}
            base.update(interpreted)
            base["understanding_source"] = str(
                interpreted.get("understanding_source") or "layer2_llm"
            )
        self._finalize_understanding(base)
        return base

    def _layer1_semantic_facts(
        self,
        message: str,
    ) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        if self._is_capability_question(message):
            facts.append(
                self._fact(
                    fact_type="user_request",
                    predicate="ask",
                    obj="agent_capability",
                    source="user",
                )
            )
        if self._is_interaction_chat(message):
            facts.append(
                self._fact(
                    fact_type="user_request",
                    predicate="chat",
                    obj="conversation",
                    source="user",
                )
            )
        if self._is_weather_query(message):
            facts.append(
                self._fact(
                    fact_type="user_request",
                    predicate="query",
                    obj="weather",
                    qualifiers={"location": self._extract_city(message)},
                    source="user",
                )
            )
        if self._is_time_query(message):
            facts.append(
                self._fact(
                    fact_type="user_request",
                    predicate="query",
                    obj="time",
                    source="user",
                )
            )
        if self._is_math_query(message):
            facts.append(
                self._fact(
                    fact_type="user_request",
                    predicate="calculate",
                    obj="expression",
                    qualifiers={"expression": message},
                    source="user",
                )
            )
        if self._is_email_query(message):
            facts.append(
                self._fact(
                    fact_type="delivery_request",
                    predicate="send",
                    obj="message",
                    qualifiers={
                        "channel": "email",
                        "recipient": self._extract_email(message),
                        "subject": message,
                        "body": message,
                    },
                    source="user",
                )
            )
        if self._is_file_query(message):
            file_paths = self._extract_file_paths(message)
            if file_paths:
                facts.append(
                    self._fact(
                        fact_type="user_request",
                        predicate="read",
                        obj="file",
                        qualifiers={"file_path": file_paths[0]},
                        source="user",
                    )
                )
        if self._is_search_query(message):
            facts.append(
                self._fact(
                    fact_type="user_request",
                    predicate="search",
                    obj="public_information",
                    qualifiers={"query": message},
                    source="user",
                )
            )
        return facts

    def interpret_semantics(
        self,
        message: str,
        memories: list[dict[str, Any]] | None = None,
        memory_ctx: Any = None,
    ) -> SemanticUnderstanding:
        _ = memories
        _ = memory_ctx
        return {
            "semantic_facts": [
                self._fact(
                    fact_type="user_request",
                    predicate="ask",
                    obj="general_answer",
                    qualifiers={"text": message},
                    source="user",
                )
            ],
            "provider": "local",
        }

    def create_plan(
        self,
        analysis: TaskAnalysis,
        memories: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        observations: list[dict[str, Any]] | None = None,
        memory_ctx: Any = None,
        replan_context: dict[str, Any] | None = None,
        capability_context: dict[str, Any] | None = None,
    ) -> list[PlanStep]:
        _ = analysis
        _ = memories
        _ = tool_specs
        _ = observations
        _ = memory_ctx
        _ = replan_context
        _ = capability_context
        raise PlanningError("LLM planner is not configured")

    @staticmethod
    def _task_steps_only(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return filter_task_steps(plan)

    def _build_base_understanding(
        self,
        normalized_message: str,
        memories: list[dict[str, Any]] | None,
        memory_ctx: Any,
    ) -> SemanticUnderstanding:
        constraints = self._extract_constraints(normalized_message)
        return {
            "source": {
                "modality": "text",
                "raw_text": normalized_message,
                "normalized_text": normalized_message,
            },
            "normalized_task": normalized_message,
            "semantic_facts": [],
            "entities": constraints.get("entities", {}),
            "constraints": constraints,
            "context_links": {
                "relevant_memory_count": len(memories or []),
                "has_memory_context": memory_ctx is not None,
            },
            "ambiguities": [],
            "missing_info": [],
            "provider": "local",
        }

    @staticmethod
    def _fact(
        fact_type: str,
        predicate: str,
        obj: str,
        qualifiers: dict[str, Any] | None = None,
        source: str = "user",
    ) -> dict[str, Any]:
        return {
            "type": fact_type,
            "predicate": predicate,
            "object": obj,
            "qualifiers": qualifiers or {},
            "source": source,
        }

    def _finalize_understanding(self, understanding: SemanticUnderstanding) -> None:
        semantic_facts = [
            self._normalize_fact(fact)
            for fact in self._safe_list(understanding.get("semantic_facts"))
            if isinstance(fact, dict)
        ]
        if not semantic_facts:
            semantic_facts = [
                self._fact(
                    fact_type="user_request",
                    predicate="ask",
                    obj="general_answer",
                    qualifiers={"text": str(understanding.get("normalized_task") or "")},
                    source="user",
                )
            ]
            understanding["semantic_facts"] = semantic_facts
        else:
            understanding["semantic_facts"] = semantic_facts
        entities = self._safe_dict(understanding.get("entities"))
        entities.update(self._entities_from_facts(semantic_facts))
        understanding["entities"] = entities
        understanding["constraints"] = self._safe_dict(understanding.get("constraints"))
        understanding["ambiguities"] = self._safe_list(understanding.get("ambiguities"))
        understanding["missing_info"] = self._safe_list(understanding.get("missing_info"))

    @classmethod
    def _entities_from_facts(cls, semantic_facts: list[dict[str, Any]]) -> dict[str, Any]:
        entities: dict[str, Any] = {}
        locations: list[str] = []
        emails: list[str] = []
        file_paths: list[str] = []
        expressions: list[str] = []
        for fact in semantic_facts:
            qualifiers = cls._safe_dict(fact.get("qualifiers"))
            location = str(qualifiers.get("location") or "")
            recipient = str(qualifiers.get("recipient") or "")
            file_path = str(qualifiers.get("file_path") or "")
            expression = str(qualifiers.get("expression") or "")
            if location and location not in locations:
                locations.append(location)
            if recipient and recipient not in emails:
                emails.append(recipient)
            if file_path and file_path not in file_paths:
                file_paths.append(file_path)
            if expression and expression not in expressions:
                expressions.append(expression)
        if locations:
            entities["locations"] = locations
        if emails:
            entities["emails"] = emails
        if file_paths:
            entities["file_paths"] = file_paths
        if expressions:
            entities["expressions"] = expressions
        return entities

    @staticmethod
    def _allowed_tools_from_context(
        capability_context: dict[str, Any] | None,
        tool_specs: list[dict[str, Any]],
    ) -> set[str]:
        if capability_context is not None:
            return {
                str(tool_name)
                for tool_name in capability_context.get("allowed_tools", [])
                if str(tool_name)
            }
        return {
            str(spec.get("name"))
            for spec in tool_specs
            if str(spec.get("name") or "")
        }

    @classmethod
    def _normalize_fact(cls, fact: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": str(fact.get("type") or "user_request"),
            "predicate": str(fact.get("predicate") or "ask"),
            "object": str(fact.get("object") or "general_answer"),
            "qualifiers": cls._safe_dict(fact.get("qualifiers")),
            "source": str(fact.get("source") or "user"),
        }

    @staticmethod
    def _safe_dict(value: Any) -> dict[str, Any]:
        return safe_dict(value)

    @staticmethod
    def _safe_list(value: Any) -> list[Any]:
        return list(value) if isinstance(value, list) else []

    @staticmethod
    def _normalize_depends_on(value: Any) -> list[str]:
        return normalize_depends_on(value)

    @staticmethod
    def _normalize_message(message: str) -> str:
        return str(message or "").strip()

    def _extract_constraints(self, message: str) -> dict[str, Any]:
        file_paths = self._extract_file_paths(message)
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", message)
        return {
            "dates": dates,
            "entities": {
                "file_paths": file_paths,
            },
        }

    @staticmethod
    def _extract_file_paths(message: str) -> list[str]:
        pattern = r"[A-Za-z]:\\[^\s]+"
        return re.findall(pattern, message)

    @staticmethod
    def _extract_email(message: str) -> str:
        match = re.search(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            message,
        )
        return match.group(0) if match else ""

    @staticmethod
    def _normalize_keyword_text(message: str) -> str:
        return (
            message.replace("，", "")
            .replace("。", "")
            .replace("？", "")
            .replace("?", "")
            .strip()
        )

    def _is_capability_question(self, message: str) -> bool:
        text = self._normalize_keyword_text(message)
        return any(
            keyword in text
            for keyword in ("能做什么", "会做什么", "你能", "功能", "能力", "鍋氫粈涔")
        )

    def _is_interaction_chat(self, message: str) -> bool:
        text = self._normalize_keyword_text(message)
        return any(
            keyword in text
            for keyword in (
                "你好",
                "您好",
                "再见",
                "拜拜",
                "谢谢",
                "随便聊",
                "聊天",
                "浣犲ソ",
                "鍐嶈",
                "闅忎究鑱",
            )
        )

    @staticmethod
    def _is_weather_query(message: str) -> bool:
        return any(keyword in message for keyword in ("天气", "气温", "下雨", "澶╂皵"))

    @staticmethod
    def _is_time_query(message: str) -> bool:
        return any(
            keyword in message
            for keyword in ("几点", "时间", "日期", "今天", "鐜板湪", "鍑犵偣")
        )

    @staticmethod
    def _is_math_query(message: str) -> bool:
        return any(keyword in message for keyword in ("计算", "算一下", "公式", "平方", "开方", "璁＄畻"))

    @staticmethod
    def _is_email_query(message: str) -> bool:
        return any(keyword in message for keyword in ("发邮件", "发送邮件", "写邮件", "鍙戦偖"))

    @staticmethod
    def _is_file_query(message: str) -> bool:
        return any(keyword in message for keyword in ("读取文件", "打开文件", "查看文件", "璇诲彇"))

    def _is_compound_task(self, message: str) -> bool:
        groups = [
            self._is_weather_query(message),
            self._is_time_query(message),
            self._is_email_query(message),
            self._is_file_query(message) and bool(self._extract_file_paths(message)),
        ]
        return sum(1 for item in groups if item) >= 2

    @staticmethod
    def _is_search_query(message: str) -> bool:
        return any(keyword in message for keyword in ("搜索", "查找", "检索", "鎼滅储"))

    @staticmethod
    def _extract_city(message: str) -> str:
        known_cities = (
            "北京",
            "上海",
            "广州",
            "深圳",
            "杭州",
            "成都",
            "重庆",
            "鍖椾含",
        )
        for city in known_cities:
            if city in message:
                return city
        return message

    @staticmethod
    def _summarize_observations(observations: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for idx, observation in enumerate(observations, start=1):
            if observation.get("type") != "tool_result":
                continue
            if observation.get("status") != "success":
                continue
            content = str(observation.get("content") or "").strip()
            if content:
                tool_name = observation.get("tool_name", "unknown")
                lines.append(f"[observation {idx}:{tool_name}]\n{content}")
        return "\n\n".join(lines)

    @staticmethod
    def _answer_with_observations(
        user_message: str,
        observation_summary: str,
    ) -> str:
        _ = user_message
        return (
            "已完成任务执行，观察结果如下：\n\n"
            f"{observation_summary}\n\n"
            "如需继续处理，我可以基于该结果进行总结、改写或下一步分析。"
        )

    @staticmethod
    def _answer_directly(
        user_message: str,
        context: list[dict[str, str]],
    ) -> str:
        if context:
            return f"我已结合上下文理解你的问题：{user_message}"
        return f"当前运行在本地规则模型模式，已接收问题：{user_message}"


class OpenAICompatibleModelClient(LocalModelClient):
    """OpenAI-compatible model client with LLM-only planning."""

    provider_name = "openai-compatible"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model_name: str = "",
        fallback_client: LocalModelClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._fallback_client = fallback_client or LocalModelClient()

    def interpret_semantics(
        self,
        message: str,
        memories: list[dict[str, Any]] | None = None,
        memory_ctx: Any = None,
    ) -> SemanticUnderstanding:
        if not self._api_key:
            return self._fallback_client.interpret_semantics(
                message,
                memories=memories,
                memory_ctx=memory_ctx,
            )
        cache_key = self._build_understanding_cache_key(message, memories, memory_ctx)
        cached = redis_kv.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        try:
            messages = self._build_understanding_messages(message, memories, memory_ctx)
            raw_result = self._request_chat_completion(messages, temperature=0.0)
            result = self._parse_understanding_result(raw_result, message)
            redis_kv.set(cache_key, result, ttl=_LLM_CACHE_TTL)
            return result
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, httpx.HTTPError) as exc:
            logger.warning("{} semantic understanding failed: {}", self.provider_name, exc)
            return self._fallback_client.interpret_semantics(
                message,
                memories=memories,
                memory_ctx=memory_ctx,
            )

    def generate(
        self,
        user_message: str,
        observation_summary: str | None = None,
        context: list[dict[str, str]] | None = None,
        memories: list[dict[str, Any]] | None = None,
        plan: list[PlanStep] | None = None,
        observations: list[dict[str, Any]] | None = None,
    ) -> str:
        if not self._api_key:
            return self._fallback_client.generate(
                user_message=user_message,
                observation_summary=observation_summary,
                context=context,
                memories=memories,
                plan=plan,
                observations=observations,
            )
        if observations and not observation_summary:
            observation_summary = self._summarize_observations(observations)
        cache_key = self._build_generate_cache_key(user_message, observation_summary, context)
        cached = redis_kv.get(cache_key)
        if cached is not None:
            return str(cached)
        try:
            messages = self._build_messages(
                user_message=user_message,
                observation_summary=observation_summary,
                context=context or [],
                memories=memories or [],
                plan=plan or [],
            )
            result = self._request_chat_completion(messages, temperature=0.2)
            redis_kv.set(cache_key, result, ttl=_LLM_CACHE_TTL)
            return result
        except httpx.HTTPError as exc:
            logger.warning("{} generate failed: {}", self.provider_name, exc)
            return self._fallback_client.generate(
                user_message=user_message,
                observation_summary=observation_summary,
                context=context,
                memories=memories,
                plan=plan,
                observations=observations,
            )

    def evaluate_completion_quality(
        self,
        task_contract: dict[str, Any],
        answer: str,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Use the remote model only for semantic-quality acceptance criteria."""
        if not self._api_key:
            return self._fallback_client.evaluate_completion_quality(
                task_contract=task_contract,
                answer=answer,
                observations=observations,
            )
        payload = {
            "task_contract": task_contract,
            "answer": answer,
            "observations": observations,
        }
        try:
            raw_result = self._request_chat_completion(
                [
                    {
                        "role": "system",
                        "content": (
                            "Evaluate only whether the answer satisfies the semantic-quality "
                            "criteria in the task contract. Do not evaluate tool execution "
                            "or invent missing evidence. Return raw JSON with boolean "
                            "satisfied and string reason."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                temperature=0.0,
            )
            result = self._loads_json_object(raw_result)
            if not isinstance(result, dict) or not isinstance(result.get("satisfied"), bool):
                raise ValueError("completion quality result must contain boolean satisfied")
            return {
                "satisfied": bool(result["satisfied"]),
                "reason": str(result.get("reason") or "semantic_quality_checked"),
            }
        except (ValueError, TypeError, json.JSONDecodeError, httpx.HTTPError) as exc:
            logger.warning("{} completion quality check failed: {}", self.provider_name, exc)
            return self._fallback_client.evaluate_completion_quality(
                task_contract=task_contract,
                answer=answer,
                observations=observations,
            )

    def create_plan(
        self,
        analysis: TaskAnalysis,
        memories: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        observations: list[dict[str, Any]] | None = None,
        memory_ctx: Any = None,
        replan_context: dict[str, Any] | None = None,
        capability_context: dict[str, Any] | None = None,
    ) -> list[PlanStep]:
        if not self._api_key:
            raise PlanningError("LLM planner is not configured")
        payload = {
            "analysis": analysis,
            "memories": memories[:5],
            "memory_context": (
                memory_ctx.to_plan_context()
                if hasattr(memory_ctx, "to_plan_context")
                else ""
            ),
            "recent_observations": list(
                getattr(memory_ctx, "recent_observations", []) or []
            ),
            "tools": tool_specs,
            "capability_context": capability_context or {},
            "observations": observations or [],
            "replan_context": replan_context or {},
        }
        previous_output = ""
        last_error = "planning failed"
        for attempt in range(2):
            request_payload = dict(payload)
            if attempt == 1:
                request_payload["repair_context"] = {
                    "validation_error": last_error,
                    "previous_output": previous_output,
                    "instruction": "Return a corrected complete plan.",
                }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are the planning node for a task-planning agent. "
                        "Return JSON with a steps array. Each step includes "
                        "id, kind, phase, name, goal, depends_on, status, and "
                        "optional tool_name/tool_input. Return only executable task "
                        "steps. Never add understand, plan, or replan workflow steps; "
                        "those phases are controlled by the agent graph. "
                        "Tool steps may only use "
                        "tool_name values listed in capability_context.allowed_tools. "
                        "For unavailable_capabilities, forbidden_actions, or "
                        "missing_requirements, plan a respond or clarify step instead "
                        "of inventing a tool. When replan_context is "
                        "present, return a revised plan that accounts for failed "
                        "observations, completed steps, and the remaining goal. "
                        "If a step has kind=\"tool\", it must include a non-empty "
                        "tool_name from allowed_tools. If no tool is needed, use "
                        "kind=\"respond\" or kind=\"compose\", never kind=\"tool\". "
                        "recent_observations and memory_context are verified context "
                        "from previous turns. Reuse successful recent_observations for "
                        "follow-up references such as this/previous/that result; do not "
                        "repeat an equivalent read-only tool call unless the user asks "
                        "for a refresh or new parameters. Never treat side-effect tools "
                        "such as email as already executed unless there is a successful "
                        "current observation for that exact action. "
                        "Preserve completed steps as completed and do not blindly "
                        "repeat failed tool steps unless the revised goal requires it. "
                        "Return raw JSON only, without markdown fences, prose, or reasoning text."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(request_payload, ensure_ascii=False),
                },
            ]
            try:
                previous_output = self._request_chat_completion(
                    messages,
                    temperature=0.1,
                )
                plan = self._normalize_plan_result(
                    previous_output,
                    analysis,
                    tool_specs,
                    capability_context=capability_context,
                )
                if not plan:
                    raise ValueError("planner returned no executable task steps")
                return plan
            except (
                ValueError,
                KeyError,
                TypeError,
                json.JSONDecodeError,
                httpx.HTTPError,
            ) as exc:
                last_error = str(exc)
                logger.warning(
                    "{} planning attempt {} failed: {}",
                    self.provider_name,
                    attempt + 1,
                    exc,
                )

        raise PlanningError(f"invalid plan after repair: {last_error}")

    @staticmethod
    def _build_understanding_cache_key(
        message: str,
        memories: list[dict[str, Any]] | None = None,
        memory_ctx: Any = None,
    ) -> str:
        parts = [message]
        if memories:
            mem_preview = "|".join(str(m.get("text", ""))[:60] for m in memories[:3])
            parts.append(hashlib.md5(mem_preview.encode()).hexdigest()[:12])
        if memory_ctx is not None:
            parts.append(getattr(memory_ctx, "last_user_msg", ""))
        return "llm:understanding:" + hashlib.md5("||".join(parts).encode()).hexdigest()

    @staticmethod
    def _build_generate_cache_key(
        user_message: str,
        observation_summary: str | None,
        context: list[dict[str, str]] | None,
    ) -> str:
        parts = [user_message]
        if observation_summary:
            parts.append(observation_summary[:120])
        if context:
            ctx_snap = "|".join(m.get("content", "")[:60] for m in context[-3:])
            parts.append(hashlib.md5(ctx_snap.encode()).hexdigest()[:12])
        return "llm:generate:" + hashlib.md5("||".join(parts).encode()).hexdigest()

    def _build_understanding_messages(
        self,
        message: str,
        memories: list[dict[str, Any]] | None = None,
        memory_ctx: Any = None,
    ) -> list[dict[str, str]]:
        payload = {
            "message": message,
            "memories": memories or [],
            "memory_context": getattr(memory_ctx, "to_llm_context", lambda: "")(),
        }
        return [
            {
                "role": "system",
                "content": (
                    "Translate the user request into semantic facts only. "
                    "Describe what is true or requested, not how to execute it. "
                    "Return JSON with semantic_facts, entities, constraints, "
                    "ambiguities, and missing_info. Do not return tool names or tool inputs. "
                    "Return raw JSON only, without markdown fences, prose, or reasoning text."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _build_messages(
        self,
        user_message: str,
        observation_summary: str | None,
        context: list[dict[str, str]],
        memories: list[dict[str, Any]],
        plan: list[PlanStep],
    ) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer accurately using the plan and observations when present. "
                    "Treat successful tool observations as authoritative facts. Do not "
                    "claim an action is impossible or unperformed when a successful "
                    "observation says it was completed."
                ),
            }
        ]
        messages.extend(context[-10:])
        if memories:
            messages.append(
                {
                    "role": "system",
                    "content": json.dumps(memories[:5], ensure_ascii=False),
                }
            )
        if plan:
            messages.append(
                {
                    "role": "system",
                    "content": "Plan:\n" + json.dumps(plan, ensure_ascii=False),
                }
            )
        if observation_summary:
            messages.append(
                {
                    "role": "user",
                    "content": f"Observations:\n{observation_summary}",
                }
            )
        messages.append({"role": "user", "content": user_message})
        return messages

    def _request_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        if not self._api_key or not self._base_url or not self._model_name:
            raise httpx.HTTPError("model client is not configured")
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model_name,
                "messages": messages,
                "temperature": temperature,
            },
            timeout=get_settings().request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])

    def _parse_understanding_result(
        self,
        raw_result: str,
        message: str,
    ) -> SemanticUnderstanding:
        payload = self._loads_json_object(raw_result)
        if not isinstance(payload, dict):
            raise ValueError("understanding JSON must be an object")
        semantic_facts = self._safe_list(payload.get("semantic_facts"))
        if not isinstance(semantic_facts, list):
            raise ValueError("semantic_facts must be a list")
        normalized_facts: list[dict[str, Any]] = []
        for raw_fact in semantic_facts:
            if not isinstance(raw_fact, dict):
                continue
            normalized_facts.append(self._normalize_fact(raw_fact))
        return {
            "normalized_task": message,
            "semantic_facts": normalized_facts,
            "entities": self._safe_dict(payload.get("entities")),
            "constraints": self._safe_dict(payload.get("constraints")),
            "ambiguities": self._safe_list(payload.get("ambiguities")),
            "missing_info": self._safe_list(payload.get("missing_info")),
            "understanding_source": "layer2_llm",
            "provider": self.provider_name,
        }

    def _normalize_plan_result(
        self,
        raw_result: str,
        analysis: TaskAnalysis,
        tool_specs: list[dict[str, Any]],
        capability_context: dict[str, Any] | None = None,
    ) -> list[PlanStep]:
        payload = self._loads_json_object(raw_result)
        raw_steps = payload.get("steps", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_steps, list):
            raise ValueError("plan steps must be a list")
        available_tools = self._allowed_tools_from_context(
            capability_context,
            tool_specs,
        )
        steps: list[PlanStep] = []
        for index, raw_step in enumerate(raw_steps, start=1):
            if not isinstance(raw_step, dict):
                continue
            step = {
                "id": str(raw_step.get("id") or f"step_{index}"),
                "kind": str(raw_step.get("kind") or ""),
                "phase": str(raw_step.get("phase") or ""),
                "name": str(raw_step.get("name") or f"Step {index}"),
                "goal": str(raw_step.get("goal") or ""),
                "depends_on": self._normalize_depends_on(raw_step.get("depends_on")),
                "status": str(raw_step.get("status") or "pending"),
            }
            tool_name = str(raw_step.get("tool_name") or "").strip().lower()
            if step["kind"] == "tool" and not tool_name:
                raise ValueError("tool step missing required tool_name")
            if tool_name:
                if tool_name not in available_tools:
                    raise ValueError(f"plan uses unregistered tool: {tool_name}")
                step["kind"] = "tool"
                step["tool_name"] = tool_name
                step["tool_input"] = self._normalize_tool_input(
                    tool_name,
                    self._safe_dict(raw_step.get("tool_input")),
                    str(analysis.get("normalized_task") or ""),
                )
            steps.append(step)
        return self._task_steps_only(steps)

    @staticmethod
    def _normalize_depends_on(value: Any) -> list[str]:
        return normalize_depends_on(value)

    @staticmethod
    def _loads_json_object(raw_result: str) -> Any:
        """Parse model JSON while tolerating markdown/prose wrappers."""
        text = (raw_result or "").strip()
        if not text:
            raise ValueError("model returned empty JSON content")
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start_candidates = [
                index for index in (text.find("{"), text.find("["))
                if index >= 0
            ]
            if not start_candidates:
                raise
            start = min(start_candidates)
            end = max(text.rfind("}"), text.rfind("]"))
            if end <= start:
                raise
            return json.loads(text[start : end + 1])

    def _normalize_tool_input(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        if tool_name == "weather":
            return {"city": str(tool_input.get("city") or self._extract_city(message)).strip()}
        if tool_name == "email":
            return {
                "to": str(tool_input.get("to", "")).strip(),
                "subject": str(tool_input.get("subject") or message).strip(),
                "body": str(tool_input.get("body") or message).strip(),
                "cc": str(tool_input.get("cc", "")).strip(),
            }
        if tool_name == "code":
            return {"expression": str(tool_input.get("expression") or message).strip()}
        if tool_name == "file":
            file_paths = self._extract_file_paths(message)
            file_path = str(tool_input.get("file_path") or (file_paths[0] if file_paths else ""))
            return {"file_path": file_path.strip()}
        if tool_name == "knowledge":
            return {"query": str(tool_input.get("query") or message).strip()}
        if tool_name == "browser":
            return {
                "url": str(tool_input.get("url") or "").strip(),
                "action": str(tool_input.get("action") or "read"),
            }
        if tool_name == "time":
            result: dict[str, Any] = {}
            if "offset_days" in tool_input:
                result["offset_days"] = int(tool_input["offset_days"])
            return result
        return dict(tool_input)


class OpenAIModelClient(OpenAICompatibleModelClient):
    provider_name = "openai"


class DeepSeekModelClient(OpenAICompatibleModelClient):
    provider_name = "deepseek"


class QwenModelClient(OpenAICompatibleModelClient):
    provider_name = "qwen"


def _build_fallback_client(settings: Any) -> LocalModelClient | None:
    if getattr(settings, "deepseek_api_key", "") and getattr(settings, "deepseek_model", ""):
        return DeepSeekModelClient(
            api_key=settings.deepseek_api_key,
            base_url=getattr(settings, "deepseek_base_url", ""),
            model_name=settings.deepseek_model,
        )
    return None


def get_model_client() -> LocalModelClient:
    settings = get_settings()
    provider = str(getattr(settings, "model_provider", "") or "").lower()

    if provider == "qwen":
        if getattr(settings, "qwen_api_key", ""):
            return QwenModelClient(
                api_key=settings.qwen_api_key,
                base_url=getattr(settings, "qwen_base_url", ""),
                model_name=getattr(settings, "qwen_model", ""),
            )
        return LocalModelClient()

    if provider == "deepseek":
        if getattr(settings, "deepseek_api_key", ""):
            return DeepSeekModelClient(
                api_key=settings.deepseek_api_key,
                base_url=getattr(settings, "deepseek_base_url", ""),
                model_name=getattr(settings, "deepseek_model", ""),
            )
        return LocalModelClient()

    if provider == "openai":
        if getattr(settings, "openai_api_key", ""):
            return OpenAIModelClient(
                api_key=settings.openai_api_key,
                base_url=getattr(settings, "openai_base_url", ""),
                model_name=getattr(settings, "openai_model", ""),
                fallback_client=_build_fallback_client(settings),
            )
        return LocalModelClient()

    if getattr(settings, "openai_api_key", ""):
        return OpenAIModelClient(
            api_key=settings.openai_api_key,
            base_url=getattr(settings, "openai_base_url", ""),
            model_name=getattr(settings, "openai_model", ""),
            fallback_client=_build_fallback_client(settings),
        )
    if getattr(settings, "deepseek_api_key", ""):
        return DeepSeekModelClient(
            api_key=settings.deepseek_api_key,
            base_url=getattr(settings, "deepseek_base_url", ""),
            model_name=getattr(settings, "deepseek_model", ""),
        )
    return LocalModelClient()
