"""Agent node helper tests."""

import services.llm_service as llm_service
from agent.nodes import AgentNodes
from tools.base import ToolResult
from utils.exception import ToolException


class _FakeKnowledgeTool:
    required_permissions = frozenset({"knowledge:read"})
    context_schema = {"user_id": "upload_user_id"}

    def run_with_context(self, tool_input, context):
        scoped_input = dict(tool_input)
        scoped_input["upload_user_id"] = context.user_id
        return ToolResult(
            content=str(scoped_input["upload_user_id"]),
            metadata={"scoped_input": scoped_input},
        )


class _FakeRegistry:
    def get(self, tool_name):
        assert tool_name == "knowledge"
        return _FakeKnowledgeTool()

    def get_langchain_tool(self, tool_name):
        return None

    def list_specs(self):
        return []


class _FakeAgentMemory:
    def __init__(self, search_results=None) -> None:
        self.search_kwargs = None
        self.add_text_kwargs = None
        self.add_text_calls = []
        self.search_results = search_results

    def search_filtered(self, **kwargs):
        self.search_kwargs = kwargs
        if self.search_results is not None:
            return self.search_results
        return [
            {
                "score": 0.9,
                "text": "user scoped memory",
                "metadata": {
                    "user_id": kwargs["metadata_filter"]["user_id"],
                    "memory_kind": "long_term",
                    "status": "active",
                },
            }
        ]

    def add_text(self, **kwargs):
        self.add_text_kwargs = kwargs
        self.add_text_calls.append(kwargs)


def _run_memory_write_immediately(records, state):
    AgentNodes._save_memory_records(records, state)  # type: ignore[arg-type]


class _StaticTool:
    def __init__(self, content: str) -> None:
        self.content = content

    def run(self, tool_input):
        return ToolResult(content=self.content, metadata={"input": tool_input})


class _FailingTool:
    def run(self, tool_input):
        raise ToolException("tool failed")


class _MultiToolRegistry:
    def __init__(self, tools, specs=None) -> None:
        self._tools = tools
        self._specs = specs or []

    def get(self, tool_name):
        return self._tools[tool_name]

    def get_langchain_tool(self, tool_name):
        return None

    def list_specs(self):
        return self._specs


class _ToolSpec:
    def __init__(
        self,
        name: str,
        description: str = "",
        input_schema=None,
        required_permissions=(),
        context_schema=None,
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema or {}
        self.required_permissions = required_permissions
        self.context_schema = context_schema or {}


class _PlanOnlyModel:
    def __init__(self, plan) -> None:
        self.plan = plan
        self.create_plan_calls = 0
        self.create_plan_kwargs = None

    def create_plan(self, **kwargs):
        self.create_plan_calls += 1
        self.create_plan_kwargs = kwargs
        return self.plan


class _CapturingGenerateModel:
    def __init__(self, answer: str = "answer") -> None:
        self.generate_kwargs = None
        self.answer = answer

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return self.answer


class _SemanticEvaluationModel:
    def __init__(self, satisfied: bool = True) -> None:
        self.satisfied = satisfied
        self.calls = 0

    def evaluate_completion_quality(self, **kwargs):
        self.calls += 1
        return {
            "satisfied": self.satisfied,
            "reason": "semantic quality checked",
        }


class TestAgentNodesHelpers:
    def setup_method(self) -> None:
        self._nodes = AgentNodes()

    def test_build_task_contract_uses_semantic_facts_as_acceptance_criteria(self) -> None:
        contract = self._nodes._build_task_contract(
            {
                "normalized_task": "check Beijing weather",
                "constraints": {"language": "Chinese"},
                "semantic_facts": [
                    {
                        "type": "user_request",
                        "predicate": "query",
                        "object": "weather",
                        "qualifiers": {"location": "Beijing"},
                    }
                ],
            }
        )

        assert contract["objective"] == "check Beijing weather"
        assert contract["constraints"] == {"language": "Chinese"}
        assert contract["criteria"][0]["type"] == "observation"
        assert contract["criteria"][0]["tool_name"] == "weather"
        assert [item["type"] for item in contract["criteria"]] == ["observation"]

    def test_text_writing_is_semantic_quality_not_external_capability(self) -> None:
        fact = {
            "type": "user_request",
            "predicate": "write",
            "object": "report",
            "qualifiers": {},
        }
        contract = self._nodes._build_task_contract(
            {
                "normalized_task": "write a report",
                "semantic_facts": [fact],
            }
        )
        capability_context = self._nodes._build_capability_context(
            understanding={"semantic_facts": [fact]},
            tool_specs=[],
            tool_context=self._nodes._build_tool_context(
                {"user_id": 1, "session_id": "s1"}  # type: ignore[arg-type]
            ),
        )

        assert contract["criteria"][0]["type"] == "semantic_quality"
        assert capability_context["unavailable_capabilities"] == []

    def test_planning_node_creates_plan_without_tool_steps(self) -> None:
        plan = [
            {
                "id": "respond",
                "kind": "respond",
                "phase": "action",
                "goal": "answer the user",
                "depends_on": [],
                "status": "pending",
            },
        ]
        model = _PlanOnlyModel(plan)
        self._nodes._model_client = model
        self._nodes._tool_registry = _MultiToolRegistry({})
        state = {
            "understanding": {
                "normalized_task": "write a plan",
                "semantic_facts": [
                    {"type": "user_request", "predicate": "ask", "object": "general_answer"}
                ],
            },
            "relevant_memories": [],
            "memory_context": None,
            "observations": [],
            "current_step_id": "old",
        }

        result = self._nodes.planning_node(state)  # type: ignore[arg-type]

        assert model.create_plan_calls == 1
        assert model.create_plan_kwargs["capability_context"]["allowed_tools"] == []
        assert result["plan"] == plan
        assert result["current_step_id"] == ""
        assert result["tool_calls"] == []
        assert result["capability_context"]["unavailable_capabilities"] == []

    def test_planning_node_removes_workflow_pseudo_steps_from_model_plan(self) -> None:
        model = _PlanOnlyModel(
            [
                {
                    "id": "understand",
                    "kind": "understand",
                    "status": "completed",
                    "depends_on": [],
                },
                {
                    "id": "plan",
                    "kind": "plan",
                    "status": "completed",
                    "depends_on": ["understand"],
                },
                {
                    "id": "respond",
                    "kind": "respond",
                    "status": "pending",
                    "depends_on": ["plan"],
                },
            ]
        )
        self._nodes._model_client = model
        self._nodes._tool_registry = _MultiToolRegistry({})
        state = {
            "understanding": {
                "normalized_task": "answer",
                "semantic_facts": [
                    {"type": "user_request", "predicate": "ask", "object": "general_answer"}
                ],
            },
            "relevant_memories": [],
            "memory_context": None,
            "observations": [],
            "current_step_id": "",
        }

        result = self._nodes.planning_node(state)  # type: ignore[arg-type]

        assert [step["id"] for step in result["plan"]] == ["respond"]
        assert result["plan"][0]["depends_on"] == []

    def test_planning_node_marks_task_blocked_when_llm_planning_fails(self) -> None:
        planning_error = getattr(llm_service, "PlanningError")

        class FailingPlanner:
            def create_plan(self, **kwargs):
                raise planning_error("invalid plan after repair")

        self._nodes._model_client = FailingPlanner()
        self._nodes._tool_registry = _MultiToolRegistry({})
        state = {
            "understanding": {
                "normalized_task": "answer",
                "semantic_facts": [
                    {"type": "user_request", "predicate": "ask", "object": "general_answer"}
                ],
            },
            "relevant_memories": [],
            "memory_context": None,
            "observations": [],
            "current_step_id": "",
            "tool_calls": [],
            "error_info": "",
        }

        result = self._nodes.planning_node(state)  # type: ignore[arg-type]

        assert result["plan"] == []
        assert result["tool_calls"] == []
        assert result["task_evaluation"]["status"] == "blocked"
        assert result["task_evaluation"]["reason"] == "planning_failed"
        assert "invalid plan after repair" in result["error_info"]

    def test_planning_node_consumes_replan_request(self) -> None:
        plan = [{"id": "action", "kind": "compose", "phase": "action", "status": "pending"}]
        model = _PlanOnlyModel(plan)
        self._nodes._model_client = model
        self._nodes._tool_registry = _MultiToolRegistry({})
        state = {
            "understanding": {
                "normalized_task": "Beijing weather",
                "semantic_facts": [
                    {
                        "type": "user_request",
                        "predicate": "query",
                        "object": "weather",
                        "qualifiers": {"location": "Beijing"},
                    }
                ],
            },
            "relevant_memories": [],
            "memory_context": None,
            "observations": [
                {"type": "tool_result", "tool_name": "weather", "status": "failed"}
            ],
            "replan_requested": True,
            "replan_count": 0,
        }

        result = self._nodes.planning_node(state)  # type: ignore[arg-type]

        assert result["replan_requested"] is False
        assert result["replan_count"] == 1
        assert model.create_plan_kwargs["observations"] == state["observations"]
        assert model.create_plan_kwargs["replan_context"]["attempt"] == 1
        assert model.create_plan_kwargs["replan_context"]["failed_observations"]
        assert model.create_plan_kwargs["replan_context"]["original_goal"] == ""
        assert model.create_plan_kwargs["capability_context"]["unavailable_capabilities"][0][
            "desired_tool"
        ] == "weather"
        assert result["tool_calls"] == []

    def test_planning_node_builds_allowed_tool_boundary(self) -> None:
        plan = [
            {
                "id": "tool_1",
                "kind": "tool",
                "phase": "tools",
                "tool_name": "weather",
                "tool_input": {"city": "Beijing"},
                "depends_on": [],
                "status": "pending",
            }
        ]
        model = _PlanOnlyModel(plan)
        self._nodes._model_client = model
        self._nodes._tool_registry = _MultiToolRegistry(
            {},
            specs=[
                _ToolSpec(
                    name="weather",
                    description="weather",
                    input_schema={"city": "city"},
                    required_permissions=("network:read",),
                )
            ],
        )
        state = {
            "understanding": {
                "normalized_task": "Beijing weather",
                "semantic_facts": [
                    {
                        "type": "user_request",
                        "predicate": "query",
                        "object": "weather",
                        "qualifiers": {"location": "Beijing"},
                    }
                ],
            },
            "relevant_memories": [],
            "memory_context": None,
            "observations": [],
            "current_step_id": "",
            "user_id": 1,
            "session_id": "s1",
            "task_desc": "Beijing weather",
            "normalized_task": "Beijing weather",
        }

        result = self._nodes.planning_node(state)  # type: ignore[arg-type]

        capability_context = result["capability_context"]
        assert capability_context["allowed_tools"] == ["weather"]
        assert capability_context["unavailable_capabilities"] == []
        assert model.create_plan_kwargs["capability_context"] == capability_context

    def test_planning_node_records_unavailable_tool_boundary(self) -> None:
        plan = [{"id": "respond", "kind": "respond", "status": "pending"}]
        model = _PlanOnlyModel(plan)
        self._nodes._model_client = model
        self._nodes._tool_registry = _MultiToolRegistry({})
        state = {
            "understanding": {
                "normalized_task": "Beijing weather",
                "semantic_facts": [
                    {
                        "type": "user_request",
                        "predicate": "query",
                        "object": "weather",
                        "qualifiers": {"location": "Beijing"},
                    }
                ],
            },
            "relevant_memories": [],
            "memory_context": None,
            "observations": [],
            "current_step_id": "",
            "user_id": 1,
            "session_id": "s1",
            "task_desc": "Beijing weather",
            "normalized_task": "Beijing weather",
        }

        result = self._nodes.planning_node(state)  # type: ignore[arg-type]

        gaps = result["capability_context"]["unavailable_capabilities"]
        assert gaps[0]["desired_tool"] == "weather"
        assert gaps[0]["reason"] == "tool_not_registered"

    def test_execute_node_runs_respond_step(self) -> None:
        model = _CapturingGenerateModel()
        self._nodes._model_client = model
        state = {
            "normalized_task": "write a plan",
            "task_desc": "write a plan",
            "observations": [],
            "messages": [],
            "relevant_memories": [],
            "plan": [
                {
                    "id": "respond",
                    "kind": "respond",
                    "phase": "action",
                    "goal": "answer directly",
                    "depends_on": [],
                    "status": "pending",
                },
            ],
            "tool_calls": [],
            "current_step_id": "",
            "step_count": 0,
            "max_steps": 6,
            "error_info": "",
        }

        result = self._nodes.execute_node(state)  # type: ignore[arg-type]

        assert result["answer"] == "answer"
        assert result["plan"][-1]["status"] == "completed"
        assert result["current_step_id"] == "respond"
        assert result["observations"][-1]["type"] == "final_answer"
        assert model.generate_kwargs["plan"][-1]["status"] == "running"

    def test_finalize_sanitizes_tool_call_text_answer(self) -> None:
        model = _CapturingGenerateModel(
            'call\n{"name":"browser","arguments":{"url":"https://example.com"}}'
        )
        self._nodes._model_client = model
        state = {
            "normalized_task": "book ticket",
            "task_desc": "book ticket",
            "observations": [
                {
                    "type": "tool_result",
                    "tool_name": "browser",
                    "status": "failed",
                    "error_msg": "missing url",
                }
            ],
            "messages": [],
            "relevant_memories": [],
            "plan": [],
            "error_info": "",
            "answer": "",
        }

        result = self._nodes.finalize_node(state)  # type: ignore[arg-type]

        assert "call" not in result["answer"]
        assert "missing url" in result["answer"]
        assert "browser:failed" in result["answer"]

        observed = self._nodes.observe_node(result)  # type: ignore[arg-type]
        assert observed["current_step_id"] == ""

    def test_finalize_node_generates_only_when_no_answer_exists(self) -> None:
        model = _CapturingGenerateModel()
        self._nodes._model_client = model
        state = {
            "normalized_task": "summarize result",
            "task_desc": "summarize result",
            "observations": [
                {
                    "type": "tool_result",
                    "tool_name": "weather",
                    "status": "success",
                    "content": "Beijing 20C",
                }
            ],
            "messages": [],
            "relevant_memories": [],
            "plan": [],
            "error_info": "",
            "answer": "",
        }

        result = self._nodes.finalize_node(state)  # type: ignore[arg-type]

        assert result["answer"] == "answer"
        assert "Beijing 20C" in model.generate_kwargs["observation_summary"]

    def test_finalize_node_preserves_existing_answer(self) -> None:
        model = _CapturingGenerateModel()
        self._nodes._model_client = model
        state = {"answer": "already done"}

        result = self._nodes.finalize_node(state)  # type: ignore[arg-type]

        assert result["answer"] == "already done"
        assert model.generate_kwargs is None

    def test_finalize_node_explains_blocked_planning(self) -> None:
        model = _CapturingGenerateModel()
        self._nodes._model_client = model
        state = {
            "answer": "",
            "task_evaluation": {
                "status": "blocked",
                "reason": "planning_failed",
            },
            "error_info": "planning_failed: invalid plan after repair",
            "observations": [],
        }

        result = self._nodes.finalize_node(state)  # type: ignore[arg-type]

        assert "reliable execution plan" in result["answer"]
        assert "invalid plan after repair" in result["answer"]
        assert model.generate_kwargs is None

    def test_run_tool_via_langchain_returns_none_for_unknown_tool(self) -> None:
        result = self._nodes._run_tool_via_langchain(
            "nonexistent_tool",
            {"query": "test"},
        )
        assert result is None

    def test_build_tool_context_grants_authenticated_permissions(self) -> None:
        context = self._nodes._build_tool_context(
            {"user_id": 42, "session_id": "s1"}  # type: ignore[arg-type]
        )

        assert context.user_id == 42
        assert context.session_id == "s1"
        assert "knowledge:read" in context.permissions

    def test_execute_node_preserves_tool_context_scope(self) -> None:
        self._nodes._tool_registry = _FakeRegistry()

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("knowledge must preserve internal user scope")

        self._nodes._run_tool_via_langchain = _fail_if_called  # type: ignore[method-assign]
        tool_step = self._nodes._build_tool_step(
            {
                "id": "knowledge_step",
                "kind": "tool",
                "tool_name": "knowledge",
                "tool_input": {"query": "policy"},
                "depends_on": [],
            },
            1,
        )
        state = {
            "plan": [dict(tool_step)],
            "tool_calls": [tool_step],
            "observations": [],
            "current_step_id": "",
            "step_count": 0,
            "max_steps": 6,
            "error_info": "",
            "user_id": 42,
            "session_id": "s1",
            "task_desc": "search policy",
            "normalized_task": "search policy",
        }

        self._nodes.execute_node(state)  # type: ignore[arg-type]
        result = self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert "42" in result["observations"][0]["content"]
        result_metadata = result["observations"][0]["metadata"]
        assert result_metadata["scoped_input"]["upload_user_id"] == 42
        assert result_metadata["execution_context"]["user_id"] == 42
        assert "upload_user_id" not in result["observations"][0]["tool_input"]

    def test_execute_node_reports_invalid_tool_step_without_raw_key_error(self) -> None:
        state = {
            "plan": [
                {
                    "id": "bad_tool_step",
                    "kind": "tool",
                    "status": "pending",
                    "depends_on": [],
                }
            ],
            "tool_calls": [],
            "observations": [],
            "current_step_id": "",
            "step_count": 0,
            "max_steps": 6,
            "error_info": "",
            "user_id": 1,
        }

        self._nodes.execute_node(state)  # type: ignore[arg-type]

        assert state["error_info"] == "tool step missing required tool_name"
        assert state["observations"][0]["status"] == "failed"
        assert state["observations"][0]["error_msg"] == (
            "tool step missing required tool_name"
        )

    def test_execute_node_runs_one_plan_step_at_a_time(self) -> None:
        self._nodes._tool_registry = _MultiToolRegistry({
            "weather": _StaticTool("Beijing 20C"),
            "email": _StaticTool("email sent"),
        })
        weather_step = self._nodes._build_tool_step(
            {
                "id": "weather_step",
                "kind": "tool",
                "tool_name": "weather",
                "tool_input": {"city": "Beijing"},
                "depends_on": [],
            },
            1,
        )
        email_step = self._nodes._build_tool_step(
            {
                "id": "email_step",
                "kind": "tool",
                "tool_name": "email",
                "tool_input": {
                    "to": "ops@example.com",
                    "subject": "weather",
                    "body": "send weather",
                },
                "depends_on": ["weather_step"],
            },
            2,
        )
        state = {
            "plan": [dict(weather_step), dict(email_step)],
            "tool_calls": [weather_step, email_step],
            "observations": [],
            "current_step_id": "",
            "step_count": 0,
            "max_steps": 6,
            "error_info": "",
            "user_id": 1,
        }

        self._nodes.execute_node(state)  # type: ignore[arg-type]
        self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert state["plan"][0]["status"] == "completed"
        assert state["plan"][1]["status"] == "pending"
        assert len(state["observations"]) == 1

        self._nodes.execute_node(state)  # type: ignore[arg-type]
        self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert state["plan"][1]["status"] == "completed"
        assert len(state["observations"]) == 2
        assert "Beijing 20C" in state["observations"][1]["tool_input"]["body"]

    def test_observe_skips_steps_with_failed_dependency(self) -> None:
        self._nodes._tool_registry = _MultiToolRegistry({
            "weather": _FailingTool(),
            "email": _StaticTool("email sent"),
        })
        weather_step = self._nodes._build_tool_step(
            {
                "id": "weather_step",
                "kind": "tool",
                "tool_name": "weather",
                "tool_input": {"city": "Beijing"},
                "depends_on": [],
            },
            1,
        )
        email_step = self._nodes._build_tool_step(
            {
                "id": "email_step",
                "kind": "tool",
                "tool_name": "email",
                "tool_input": {"to": "ops@example.com", "subject": "weather"},
                "depends_on": ["weather_step"],
            },
            2,
        )
        state = {
            "plan": [dict(weather_step), dict(email_step)],
            "tool_calls": [weather_step, email_step],
            "observations": [],
            "current_step_id": "",
            "step_count": 0,
            "max_steps": 6,
            "error_info": "",
            "user_id": 1,
        }

        self._nodes.execute_node(state)  # type: ignore[arg-type]
        self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert state["tool_calls"][0]["status"] == "failed"
        assert state["tool_calls"][1]["status"] == "skipped"
        assert state["tool_calls"][1]["error_msg"] == "dependency_failed:weather_step"
        assert state["error_info"] == "tool failed"

    def test_observe_requests_replan_after_tool_failure(self) -> None:
        state = {
            "tool_calls": [],
            "observations": [
                {
                    "type": "tool_result",
                    "tool_name": "weather",
                    "status": "failed",
                    "error_msg": "provider unavailable",
                }
            ],
            "error_info": "",
            "replan_requested": False,
            "replan_count": 0,
            "max_replans": 1,
            "plan": [],
            "current_step_id": "weather",
        }

        self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert state["replan_requested"] is True
        assert state["error_info"] == "provider unavailable"
        assert state["current_step_id"] == ""
        assert state["replan_context"]["failed_observations"][0]["tool_name"] == "weather"
        assert "provider unavailable" == state["replan_context"]["error_info"]

    def test_observe_rule_evaluation_does_not_call_semantic_model(self) -> None:
        model = _SemanticEvaluationModel()
        self._nodes._model_client = model
        state = {
            "task_contract": {
                "objective": "check weather",
                "criteria": [
                    {
                        "id": "fact_1",
                        "type": "observation",
                        "tool_name": "weather",
                        "required": True,
                    },
                ],
            },
            "tool_calls": [],
            "observations": [
                {
                    "type": "tool_result",
                    "tool_name": "weather",
                    "status": "success",
                    "content": "Beijing 20C",
                }
            ],
            "plan": [
                {"id": "weather", "kind": "tool", "status": "completed"},
                {"id": "respond", "kind": "compose", "status": "pending"},
            ],
            "answer": "",
            "error_info": "",
            "replan_requested": False,
            "replan_count": 0,
            "max_replans": 1,
            "current_step_id": "weather",
        }

        self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert state["task_evaluation"]["status"] == "success"
        assert state["task_evaluation"]["satisfied_criteria"] == ["fact_1"]
        assert state["task_evaluation"]["unmet_criteria"] == []
        assert model.calls == 0

    def test_observe_does_not_replan_after_business_criteria_satisfied(self) -> None:
        state = {
            "task_contract": {
                "objective": "check weather",
                "criteria": [
                    {
                        "id": "fact_1",
                        "type": "observation",
                        "tool_name": "weather",
                        "required": True,
                    },
                ],
            },
            "tool_calls": [],
            "observations": [
                {
                    "type": "tool_result",
                    "tool_name": "weather",
                    "status": "success",
                    "content": "Beijing weather result",
                }
            ],
            "plan": [
                {"id": "weather", "kind": "tool", "status": "completed"},
            ],
            "answer": "",
            "error_info": "",
            "replan_requested": False,
            "replan_count": 0,
            "max_replans": 1,
            "current_step_id": "weather",
        }

        self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert state["task_evaluation"]["status"] == "success"
        assert state["task_evaluation"]["unmet_criteria"] == []
        assert state["replan_requested"] is False
        assert state.get("replan_context", {}) == {}

    def test_observe_rejects_success_observation_for_wrong_qualifier(self) -> None:
        state = {
            "task_contract": {
                "objective": "check Shanghai weather",
                "criteria": [
                    {
                        "id": "weather",
                        "type": "observation",
                        "tool_name": "weather",
                        "qualifiers": {"location": "Shanghai"},
                        "required": True,
                    }
                ],
            },
            "tool_calls": [],
            "observations": [
                {
                    "type": "tool_result",
                    "tool_name": "weather",
                    "status": "success",
                    "content": "Beijing 20C",
                }
            ],
            "plan": [],
            "answer": "",
            "error_info": "",
            "replan_requested": False,
            "replan_count": 1,
            "max_replans": 1,
            "current_step_id": "weather",
        }

        self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert state["task_evaluation"]["status"] == "failed"
        assert state["task_evaluation"]["unmet_criteria"] == ["weather"]

    def test_observe_uses_semantic_model_only_for_quality_criterion(self) -> None:
        model = _SemanticEvaluationModel(satisfied=True)
        self._nodes._model_client = model
        state = {
            "task_contract": {
                "objective": "summarize the report",
                "criteria": [
                    {
                        "id": "fact_1",
                        "type": "semantic_quality",
                        "description": "The answer summarizes the report.",
                        "required": True,
                    },
                ],
            },
            "tool_calls": [],
            "observations": [
                {
                    "type": "final_answer",
                    "step_id": "respond",
                    "status": "completed",
                    "content": "Summary",
                }
            ],
            "plan": [
                {"id": "respond", "kind": "respond", "status": "completed"},
            ],
            "answer": "Summary",
            "error_info": "",
            "replan_requested": False,
            "replan_count": 0,
            "max_replans": 1,
            "current_step_id": "respond",
        }

        self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert state["task_evaluation"]["status"] == "success"
        assert state["task_evaluation"]["unmet_criteria"] == []
        assert model.calls == 1

    def test_observe_replans_when_criteria_unmet_without_pending_steps(self) -> None:
        state = {
            "task_contract": {
                "objective": "check weather",
                "criteria": [
                    {
                        "id": "fact_1",
                        "type": "observation",
                        "tool_name": "weather",
                        "required": True,
                    }
                ],
            },
            "tool_calls": [],
            "observations": [],
            "plan": [{"id": "weather", "kind": "tool", "status": "completed"}],
            "answer": "",
            "error_info": "",
            "replan_requested": False,
            "replan_count": 0,
            "max_replans": 1,
            "current_step_id": "weather",
            "task_desc": "check weather",
            "normalized_task": "check weather",
        }

        self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert state["task_evaluation"]["status"] == "in_progress"
        assert state["task_evaluation"]["unmet_criteria"] == ["fact_1"]
        assert state["replan_requested"] is True
        assert state["replan_context"]["trigger"] == "criteria_unmet"

    def test_observe_marks_partial_when_replans_exhausted(self) -> None:
        state = {
            "task_contract": {
                "objective": "check weather and send email",
                "criteria": [
                    {
                        "id": "weather",
                        "type": "observation",
                        "tool_name": "weather",
                        "required": True,
                    },
                    {
                        "id": "email",
                        "type": "observation",
                        "tool_name": "email",
                        "required": True,
                    },
                ],
            },
            "tool_calls": [],
            "observations": [
                {
                    "type": "tool_result",
                    "tool_name": "weather",
                    "status": "success",
                    "content": "Beijing 20C",
                }
            ],
            "plan": [],
            "answer": "",
            "error_info": "smtp unavailable",
            "replan_requested": False,
            "replan_count": 1,
            "max_replans": 1,
            "current_step_id": "email",
        }

        self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert state["task_evaluation"]["status"] == "partial"
        assert state["task_evaluation"]["satisfied_criteria"] == ["weather"]
        assert state["task_evaluation"]["unmet_criteria"] == ["email"]
        assert state["replan_requested"] is False

    def test_replan_context_includes_completed_and_remaining_steps(self) -> None:
        state = {
            "task_desc": "send weather report",
            "normalized_task": "send weather report",
            "tool_calls": [],
            "observations": [
                {
                    "type": "tool_result",
                    "step_id": "weather_step",
                    "tool_name": "weather",
                    "status": "success",
                    "content": "Beijing 20C",
                },
                {
                    "type": "tool_result",
                    "step_id": "email_step",
                    "tool_name": "email",
                    "status": "failed",
                    "error_msg": "smtp unavailable",
                },
            ],
            "error_info": "",
            "replan_requested": False,
            "replan_count": 0,
            "max_replans": 1,
            "plan": [
                {"id": "understand", "kind": "understand", "status": "completed"},
                {"id": "plan", "kind": "plan", "status": "completed"},
                {"id": "weather_step", "kind": "tool", "status": "completed"},
                {"id": "email_step", "kind": "tool", "status": "failed"},
                {
                    "id": "respond",
                    "kind": "compose",
                    "phase": "action",
                    "status": "pending",
                },
            ],
            "current_step_id": "email_step",
        }

        self._nodes.observe_node(state)  # type: ignore[arg-type]

        context = state["replan_context"]
        assert state["replan_requested"] is True
        assert context["original_goal"] == "send weather report"
        assert [step["id"] for step in context["completed_steps"]] == ["weather_step"]
        assert [step["id"] for step in context["remaining_steps"]] == [
            "email_step",
            "respond",
        ]
        assert context["failed_observations"][0]["tool_name"] == "email"
        assert context["successful_observations"][0]["content"] == "Beijing 20C"

    def test_mem_pre_filters_long_term_memory_by_user(self, monkeypatch) -> None:
        fake_memory = _FakeAgentMemory()
        monkeypatch.setattr("agent.nodes.agent_memory", fake_memory)

        state = {
            "task_desc": "search policy",
            "messages": [],
            "user_id": 42,
        }

        result = self._nodes.mem_pre_node(state)  # type: ignore[arg-type]

        assert fake_memory.search_kwargs["metadata_filter"] == {
            "user_id": 42,
            "memory_kind": "long_term",
        }
        assert result["relevant_memories"][0]["metadata"]["user_id"] == 42
        assert result["relevant_memories"][0]["metadata"]["memory_kind"] == "long_term"

    def test_mem_pre_loads_long_term_memory_only(self, monkeypatch) -> None:
        fake_memory = _FakeAgentMemory(
            search_results=[
                {
                    "score": 0.92,
                    "text": "preferred city is Beijing",
                    "metadata": {
                        "user_id": 42,
                        "memory_kind": "long_term",
                        "status": "active",
                    },
                },
            ]
        )
        monkeypatch.setattr("agent.nodes.agent_memory", fake_memory)

        state = {
            "task_desc": "weather",
            "messages": [],
            "user_id": 42,
        }

        result = self._nodes.mem_pre_node(state)  # type: ignore[arg-type]

        assert len(result["relevant_memories"]) == 1
        assert result["relevant_memories"][0]["metadata"]["memory_kind"] == "long_term"

    def test_mem_post_does_not_archive_tool_execution_episode(self, monkeypatch) -> None:
        fake_memory = _FakeAgentMemory()
        monkeypatch.setattr("agent.nodes.agent_memory", fake_memory)
        monkeypatch.setattr(
            "agent.nodes.AgentNodes._schedule_memory_write",
            staticmethod(_run_memory_write_immediately),
        )

        state = {
            "task_desc": "search policy",
            "normalized_task": "search policy",
            "understanding": {
                "normalized_task": "search policy",
                "semantic_facts": [
                    {"type": "user_request", "predicate": "search", "object": "policy"}
                ],
            },
            "observations": [
                {
                    "type": "tool_result",
                    "tool_name": "knowledge",
                    "status": "success",
                    "content": "result",
                    "error_msg": "",
                }
            ],
            "tool_calls": [],
            "plan": [],
            "error_info": "",
            "answer": "answer",
            "session_id": "session-1",
            "user_id": 42,
        }

        self._nodes.mem_post_node(state)  # type: ignore[arg-type]

        assert fake_memory.add_text_calls == []

    def test_mem_post_does_not_archive_ordinary_direct_chat(self, monkeypatch) -> None:
        fake_memory = _FakeAgentMemory()
        monkeypatch.setattr("agent.nodes.agent_memory", fake_memory)

        state = {
            "task_desc": "hello",
            "normalized_task": "hello",
            "understanding": {
                "normalized_task": "hello",
                "semantic_facts": [
                    {"type": "user_request", "predicate": "chat", "object": "conversation"}
                ],
            },
            "observations": [],
            "tool_calls": [],
            "error_info": "",
            "answer": "hello",
            "session_id": "session-1",
            "user_id": 42,
        }

        self._nodes.mem_post_node(state)  # type: ignore[arg-type]

        assert fake_memory.add_text_calls == []

    def test_mem_post_archives_explicit_semantic_memory(self, monkeypatch) -> None:
        fake_memory = _FakeAgentMemory()
        monkeypatch.setattr("agent.nodes.agent_memory", fake_memory)
        monkeypatch.setattr(
            "agent.nodes.AgentNodes._schedule_memory_write",
            staticmethod(_run_memory_write_immediately),
        )

        state = {
            "task_desc": "please remember my default city is Beijing",
            "normalized_task": "please remember my default city is Beijing",
            "understanding": {
                "normalized_task": "please remember my default city is Beijing",
                "semantic_facts": [
                    {
                        "type": "user_request",
                        "predicate": "remember",
                        "object": "preference",
                    }
                ],
            },
            "observations": [],
            "tool_calls": [],
            "error_info": "",
            "answer": "ok",
            "session_id": "session-1",
            "user_id": 42,
        }

        self._nodes.mem_post_node(state)  # type: ignore[arg-type]

        assert len(fake_memory.add_text_calls) == 1
        call = fake_memory.add_text_calls[0]
        assert call["metadata"]["memory_kind"] == "long_term"
        assert call["metadata"]["status"] == "active"
        assert call["metadata"]["reusable"] is True
        assert "Durable fact" in call["text"]

    def test_mem_post_does_not_archive_failed_tool_episode(self, monkeypatch) -> None:
        fake_memory = _FakeAgentMemory()
        monkeypatch.setattr("agent.nodes.agent_memory", fake_memory)
        monkeypatch.setattr(
            "agent.nodes.AgentNodes._schedule_memory_write",
            staticmethod(_run_memory_write_immediately),
        )

        state = {
            "task_desc": "check weather",
            "normalized_task": "check weather",
            "understanding": {
                "normalized_task": "check weather",
                "semantic_facts": [
                    {
                        "type": "user_request",
                        "predicate": "query",
                        "object": "weather",
                        "qualifiers": {"location": "Beijing"},
                    }
                ],
            },
            "observations": [
                {
                    "type": "tool_result",
                    "tool_name": "weather",
                    "status": "failed",
                    "content": "",
                    "error_msg": "provider unavailable",
                }
            ],
            "tool_calls": [],
            "plan": [],
            "error_info": "provider unavailable",
            "answer": "weather lookup failed",
            "session_id": "session-1",
            "user_id": 42,
        }

        self._nodes.mem_post_node(state)  # type: ignore[arg-type]

        assert fake_memory.add_text_calls == []
