"""LocalModelClient semantic understanding and planning tests."""

from typing import Any

import pytest

import services.llm_service as llm_service
from services.llm_service import LocalModelClient
from services.llm_service import QwenModelClient


class TestLocalModelClient:
    def setup_method(self) -> None:
        self._client = LocalModelClient()

    def test_local_model_has_no_deterministic_planner(self) -> None:
        planning_error = getattr(llm_service, "PlanningError")

        with pytest.raises(planning_error):
            self._client.create_plan(
                analysis={"normalized_task": "answer"},
                memories=[],
                tool_specs=[],
            )

    def test_remote_planner_repairs_invalid_plan_once(self) -> None:
        client = QwenModelClient(
            api_key="test-key",
            base_url="https://example.test",
            model_name="test-model",
        )
        responses = [
            '{"steps":[{"id":"bad","kind":"tool","tool_name":"missing"}]}',
            '{"steps":[{"id":"respond","kind":"respond","depends_on":[],"status":"pending"}]}',
        ]
        calls = []

        def fake_completion(messages, temperature):
            calls.append(messages)
            return responses.pop(0)

        client._request_chat_completion = fake_completion  # type: ignore[method-assign]
        plan = client.create_plan(
            analysis={"normalized_task": "answer"},
            memories=[],
            tool_specs=[],
            capability_context={"allowed_tools": []},
        )

        assert [step["id"] for step in plan] == ["respond"]
        assert len(calls) == 2
        assert "validation_error" in calls[1][-1]["content"]

    def test_remote_planner_blocks_after_failed_repair(self) -> None:
        planning_error = getattr(llm_service, "PlanningError")
        client = QwenModelClient(
            api_key="test-key",
            base_url="https://example.test",
            model_name="test-model",
        )
        calls = []

        def fake_completion(messages, temperature):
            calls.append(messages)
            return '{"steps":[{"id":"bad","kind":"tool","tool_name":"missing"}]}'

        client._request_chat_completion = fake_completion  # type: ignore[method-assign]

        with pytest.raises(planning_error):
            client.create_plan(
                analysis={"normalized_task": "answer"},
                memories=[],
                tool_specs=[],
                capability_context={"allowed_tools": []},
            )

        assert len(calls) == 2

    def test_remote_planner_never_delegates_to_fallback_planner(self) -> None:
        planning_error = getattr(llm_service, "PlanningError")

        class FallbackPlanner(LocalModelClient):
            def create_plan(self, **kwargs):  # type: ignore[no-untyped-def]
                _ = kwargs
                raise AssertionError("fallback planner must not be called")

        client = QwenModelClient(
            api_key="test-key",
            base_url="https://example.test",
            model_name="test-model",
            fallback_client=FallbackPlanner(),
        )
        calls = []

        def fake_completion(messages, temperature):
            calls.append(messages)
            return '{"steps":[{"id":"bad","kind":"tool","tool_name":"missing"}]}'

        client._request_chat_completion = fake_completion  # type: ignore[method-assign]

        with pytest.raises(planning_error):
            client.create_plan(
                analysis={"normalized_task": "answer"},
                memories=[],
                tool_specs=[],
                capability_context={"allowed_tools": []},
            )

        assert len(calls) == 2

    def test_understand_weather_as_semantic_fact_not_tool_selection(self) -> None:
        result = self._client.analyze_task("北京天气怎么样", [])

        assert "tool_name" not in result
        assert "tool_input" not in result
        assert result["normalized_task"] == "北京天气怎么样"
        assert result["understanding_source"] == "layer1_keyword"
        assert {
            "type": "user_request",
            "predicate": "query",
            "object": "weather",
            "qualifiers": {"location": "北京"},
            "source": "user",
        } in result["semantic_facts"]

    def test_understand_chat_as_semantic_fact(self) -> None:
        result = self._client.analyze_task("我今天有点累，想和你随便聊一聊", [])

        assert "tool_name" not in result
        assert result["semantic_facts"][0]["predicate"] == "chat"
        assert result["semantic_facts"][0]["object"] == "conversation"

    def test_understand_calculation_as_fact_not_code_tool(self) -> None:
        result = self._client.analyze_task("计算 25 * 4 + 10", [])

        assert "tool_name" not in result
        assert result["semantic_facts"][0]["predicate"] == "calculate"
        assert result["semantic_facts"][0]["object"] == "expression"

    def test_understand_uses_llm_after_keyword_miss(self) -> None:
        class LLMOnlyClient(LocalModelClient):
            def __init__(self) -> None:
                self.semantic_calls = 0

            def _layer1_semantic_facts(self, message: str) -> list[dict[str, Any]]:
                _ = message
                return []

            def interpret_semantics(
                self,
                message: str,
                memories: list[dict[str, Any]] | None = None,
                memory_ctx: Any = None,
            ) -> dict[str, Any]:
                _ = memories
                _ = memory_ctx
                self.semantic_calls += 1
                return {
                    "semantic_facts": [
                        {
                            "type": "user_request",
                            "predicate": "search",
                            "object": "knowledge",
                            "qualifiers": {"query": message},
                            "source": "user",
                        }
                    ],
                    "provider": "test",
                }

        client = LLMOnlyClient()
        result = client.analyze_task("请处理这个请求", [])

        assert client.semantic_calls == 1
        assert result["semantic_facts"][0]["object"] == "knowledge"
        assert result["understanding_source"] == "layer2_llm"

    def test_understand_normalizes_invalid_llm_shapes(self) -> None:
        class OddLLMClient(LocalModelClient):
            def _layer1_semantic_facts(self, message: str) -> list[dict[str, Any]]:
                _ = message
                return []

            def interpret_semantics(
                self,
                message: str,
                memories: list[dict[str, Any]] | None = None,
                memory_ctx: Any = None,
            ) -> dict[str, Any]:
                _ = message
                _ = memories
                _ = memory_ctx
                return {
                    "semantic_facts": [
                        {
                            "predicate": "query",
                            "object": "weather",
                            "qualifiers": ["not", "a", "dict"],
                        }
                    ],
                    "entities": ["not", "a", "dict"],
                    "constraints": "not-a-dict",
                    "ambiguities": "not-a-list",
                }

        result = OddLLMClient().analyze_task("查天气", [])

        assert result["entities"] == {}
        assert result["constraints"] == {}
        assert result["ambiguities"] == []
        assert result["semantic_facts"][0]["qualifiers"] == {}

    def test_layer1_does_not_treat_bare_email_as_action(self) -> None:
        result = self._client.analyze_task("user@example.com", [])

        assert result["understanding_source"] == "layer2_llm"
        assert result["semantic_facts"][0]["object"] == "general_answer"

    def test_layer1_does_not_treat_bare_file_path_as_read_action(self) -> None:
        result = self._client.analyze_task("C:\\docs\\test.pdf", [])

        assert result["understanding_source"] == "layer2_llm"
        assert result["semantic_facts"][0]["object"] == "general_answer"

    def test_extract_file_path_windows(self) -> None:
        constraints = self._client._extract_constraints("读取 C:\\docs\\test.pdf")
        entities = constraints["entities"]["file_paths"]
        assert any("C:\\docs\\test.pdf" in e for e in entities)

    def test_extract_dates(self) -> None:
        constraints = self._client._extract_constraints("报告 2024-01-15 内容")
        assert len(constraints["dates"]) >= 1
        assert "2024-01-15" in constraints["dates"][0]

    def test_generate_directly(self) -> None:
        answer = self._client.generate(
            "你好吗",
            context=[{"role": "user", "content": "你好吗"}],
        )
        assert "我已结合上下文理解你的问题" in answer

    def test_generate_with_observation_summary(self) -> None:
        answer = self._client.generate(
            "查询结果如何",
            observation_summary="温度 25C，湿度 60%",
            context=[],
        )
        assert "观察结果如下" in answer
        assert "温度 25C" in answer

    def test_local_completion_quality_requires_nonempty_answer(self) -> None:
        contract = {
            "objective": "summarize report",
            "criteria": [
                {
                    "id": "fact_1",
                    "type": "semantic_quality",
                    "description": "The answer summarizes the report.",
                }
            ],
        }

        empty = self._client.evaluate_completion_quality(
            task_contract=contract,
            answer="",
            observations=[],
        )
        nonempty = self._client.evaluate_completion_quality(
            task_contract=contract,
            answer="Summary",
            observations=[],
        )

        assert empty["satisfied"] is False
        assert nonempty["satisfied"] is True

    def test_remote_completion_quality_parses_structured_result(self) -> None:
        client = QwenModelClient(
            api_key="test-key",
            base_url="https://example.test",
            model_name="test-model",
        )
        calls = []

        def fake_completion(messages, temperature):
            calls.append({"messages": messages, "temperature": temperature})
            return '{"satisfied": true, "reason": "covers required points"}'

        client._request_chat_completion = fake_completion  # type: ignore[method-assign]
        result = client.evaluate_completion_quality(
            task_contract={
                "objective": "summarize report",
                "criteria": [{"type": "semantic_quality"}],
            },
            answer="Summary",
            observations=[],
        )

        assert result == {
            "satisfied": True,
            "reason": "covers required points",
        }
        assert calls[0]["temperature"] == 0.0
        assert "semantic-quality" in calls[0]["messages"][0]["content"]

    def test_normalize_plan_result_defends_step_shapes(self) -> None:
        raw = """
        ```json
        {"steps": [{"id": "x", "kind": "tool", "tool_name": "weather",
        "tool_input": ["bad"], "depends_on": "plan", "status": "pending"}]}
        ```
        """
        plan = QwenModelClient()._normalize_plan_result(
            raw,
            {"normalized_task": "北京天气"},
            [{"name": "weather"}],
        )

        assert plan[0]["depends_on"] == []
        assert plan[0]["tool_input"] == {"city": "北京"}

    def test_normalize_plan_result_rejects_tool_outside_capability_boundary(self) -> None:
        raw = '{"steps": [{"id": "x", "kind": "tool", "tool_name": "weather"}]}'

        try:
            QwenModelClient()._normalize_plan_result(
                raw,
                {"normalized_task": "Beijing weather"},
                [{"name": "weather"}],
                capability_context={"allowed_tools": []},
            )
        except ValueError as exc:
            assert "unregistered tool" in str(exc)
        else:
            raise AssertionError("expected unavailable tool to be rejected")

    def test_normalize_plan_result_rejects_tool_step_without_tool_name(self) -> None:
        raw = '{"steps": [{"id": "x", "kind": "tool", "goal": "explain context"}]}'

        with pytest.raises(ValueError, match="tool step missing required tool_name"):
            QwenModelClient()._normalize_plan_result(
                raw,
                {"normalized_task": "explain context"},
                [{"name": "knowledge"}],
                capability_context={"allowed_tools": ["knowledge"]},
            )

    def test_remote_planner_prompt_requires_tool_name_for_tool_steps(self) -> None:
        client = QwenModelClient(
            api_key="test-key",
            base_url="https://example.test",
            model_name="test-model",
        )
        calls = []

        def fake_completion(messages, temperature):
            calls.append(messages)
            return '{"steps":[{"id":"respond","kind":"respond","depends_on":[],"status":"pending"}]}'

        client._request_chat_completion = fake_completion  # type: ignore[method-assign]

        client.create_plan(
            analysis={"normalized_task": "explain context"},
            memories=[],
            tool_specs=[{"name": "knowledge"}],
            capability_context={"allowed_tools": ["knowledge"]},
        )

        system_prompt = calls[0][0]["content"]
        assert 'kind="tool"' in system_prompt
        assert "must include a non-empty tool_name" in system_prompt
        assert 'use kind="respond" or kind="compose"' in system_prompt

    def test_is_search_query(self) -> None:
        assert self._client._is_search_query("搜索人工智能") is True
        assert self._client._is_search_query("你好") is False

    def test_is_capability_question(self) -> None:
        assert self._client._is_capability_question("你能做什么") is True
        assert self._client._is_capability_question("今天天气") is False

    def test_is_interaction_chat(self) -> None:
        assert self._client._is_interaction_chat("你好") is True
        assert self._client._is_interaction_chat("再见") is True
        assert self._client._is_interaction_chat("帮我查资料") is False
