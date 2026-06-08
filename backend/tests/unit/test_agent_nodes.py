"""Agent 图节点 helper 方法测试。"""

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
                    "memory_kind": "semantic",
                    "status": "active",
                },
            }
        ]

    def add_text(self, **kwargs):
        self.add_text_kwargs = kwargs
        self.add_text_calls.append(kwargs)


class _StaticTool:
    def __init__(self, content: str) -> None:
        self.content = content

    def run(self, tool_input):
        return ToolResult(content=self.content, metadata={"input": tool_input})


class _FailingTool:
    def run(self, tool_input):
        raise ToolException("tool failed")


class _MultiToolRegistry:
    def __init__(self, tools) -> None:
        self._tools = tools

    def get(self, tool_name):
        return self._tools[tool_name]

    def get_langchain_tool(self, tool_name):
        return None


class TestAgentNodesHelpers:
    def setup_method(self) -> None:
        self._nodes = AgentNodes()

    # ------------------------------------------------------------------
    # 工具执行
    # ------------------------------------------------------------------

    def test_run_tool_via_langchain_returns_none_for_unknown_tool(
        self,
    ) -> None:
        """不存在的工具应返回 None。"""
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

    def test_knowledge_tool_bypasses_langchain_adapter(self) -> None:
        self._nodes._tool_registry = _FakeRegistry()

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("knowledge must preserve internal user scope")

        self._nodes._run_tool_via_langchain = _fail_if_called  # type: ignore[method-assign]
        state = {
            "need_tool": True,
            "tool_calls": [
                self._nodes._build_tool_step(
                    {
                        "id": "knowledge_step",
                        "tool_name": "knowledge",
                        "tool_input": {"query": "policy"},
                        "depends_on": ["plan"],
                    },
                    1,
                ),
            ],
            "tool_results": [],
            "tool_result": "",
            "step_count": 0,
            "error_info": "",
            "user_id": 42,
        }

        self._nodes.tool_node(state)  # type: ignore[arg-type]
        result = self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert "42" in result["tool_result"]
        result_metadata = result["tool_results"][0]["metadata"]
        assert result_metadata["scoped_input"]["upload_user_id"] == 42
        assert result_metadata["execution_context"]["user_id"] == 42
        assert "upload_user_id" not in result["tool_results"][0]["tool_input"]

    def test_tool_node_executes_one_runnable_step_at_a_time(self) -> None:
        self._nodes._tool_registry = _MultiToolRegistry({
            "weather": _StaticTool("北京 20°C"),
            "email": _StaticTool("email sent"),
        })
        weather_step = self._nodes._build_tool_step(
            {
                "id": "weather_step",
                "tool_name": "weather",
                "tool_input": {"city": "北京"},
                "depends_on": ["plan"],
            },
            1,
        )
        email_step = self._nodes._build_tool_step(
            {
                "id": "email_step",
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
            "need_tool": True,
            "tool_calls": [weather_step, email_step],
            "tool_results": [],
            "tool_result": "",
            "step_count": 0,
            "max_steps": 6,
            "error_info": "",
            "user_id": 1,
            "plan": [dict(weather_step), dict(email_step)],
        }

        self._nodes.tool_node(state)  # type: ignore[arg-type]
        self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert state["tool_calls"][0]["status"] == "completed"
        assert state["tool_calls"][1]["status"] == "pending"
        assert len(state["tool_results"]) == 1

        self._nodes.tool_node(state)  # type: ignore[arg-type]
        self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert state["tool_calls"][1]["status"] == "completed"
        assert len(state["tool_results"]) == 2
        assert "北京 20°C" in state["tool_results"][1]["tool_input"]["body"]

    def test_observe_skips_steps_with_failed_dependency(self) -> None:
        self._nodes._tool_registry = _MultiToolRegistry({
            "weather": _FailingTool(),
            "email": _StaticTool("email sent"),
        })
        weather_step = self._nodes._build_tool_step(
            {
                "id": "weather_step",
                "tool_name": "weather",
                "tool_input": {"city": "北京"},
                "depends_on": ["plan"],
            },
            1,
        )
        email_step = self._nodes._build_tool_step(
            {
                "id": "email_step",
                "tool_name": "email",
                "tool_input": {"to": "ops@example.com", "subject": "weather"},
                "depends_on": ["weather_step"],
            },
            2,
        )
        state = {
            "need_tool": True,
            "tool_calls": [weather_step, email_step],
            "tool_results": [],
            "tool_result": "",
            "step_count": 0,
            "max_steps": 6,
            "error_info": "",
            "user_id": 1,
            "plan": [dict(weather_step), dict(email_step)],
        }

        self._nodes.tool_node(state)  # type: ignore[arg-type]
        self._nodes.observe_node(state)  # type: ignore[arg-type]

        assert state["tool_calls"][0]["status"] == "failed"
        assert state["tool_calls"][1]["status"] == "skipped"
        assert state["tool_calls"][1]["error_msg"] == "dependency_failed:weather_step"
        assert state["error_info"] == "tool failed"

    # ------------------------------------------------------------------
    # Agent memory user isolation
    # ------------------------------------------------------------------

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
            "memory_kind": ["semantic", "episodic"],
        }
        assert result["relevant_memories"][0]["metadata"]["user_id"] == 42
        assert result["relevant_memories"][0]["metadata"]["memory_kind"] == "semantic"

    def test_mem_pre_ignores_failed_episodic_memory(self, monkeypatch) -> None:
        fake_memory = _FakeAgentMemory(
            search_results=[
                {
                    "score": 0.92,
                    "text": "failed weather call",
                    "metadata": {
                        "user_id": 42,
                        "memory_kind": "episodic",
                        "status": "failed",
                    },
                },
                {
                    "score": 0.91,
                    "text": "preferred city is Beijing",
                    "metadata": {
                        "user_id": 42,
                        "memory_kind": "semantic",
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
        assert result["relevant_memories"][0]["metadata"]["memory_kind"] == "semantic"

    def test_mem_post_archives_user_scope_and_status(self, monkeypatch) -> None:
        fake_memory = _FakeAgentMemory()
        monkeypatch.setattr("agent.nodes.agent_memory", fake_memory)

        state = {
            "task_desc": "search policy",
            "normalized_task": "search policy",
            "intent": {"tool_name": "knowledge", "tool_input": {"query": "policy"}},
            "tool_result": "result",
            "error_info": "",
            "answer": "answer",
            "session_id": "session-1",
            "user_id": 42,
        }

        self._nodes.mem_post_node(state)  # type: ignore[arg-type]

        metadata = fake_memory.add_text_kwargs["metadata"]
        assert metadata["user_id"] == 42
        assert metadata["session_id"] == "session-1"
        assert metadata["type"] == "agent_memory"
        assert metadata["memory_kind"] == "episodic"
        assert metadata["status"] == "success"
        assert metadata["reusable"] is True

    def test_mem_post_does_not_archive_ordinary_direct_chat(self, monkeypatch) -> None:
        fake_memory = _FakeAgentMemory()
        monkeypatch.setattr("agent.nodes.agent_memory", fake_memory)

        state = {
            "task_desc": "hello",
            "normalized_task": "hello",
            "intent": {"tool_name": "", "tool_input": {}},
            "tool_result": "",
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

        state = {
            "task_desc": "please remember my default city is Beijing",
            "normalized_task": "please remember my default city is Beijing",
            "intent": {"tool_name": "", "tool_input": {}},
            "tool_result": "",
            "error_info": "",
            "answer": "ok",
            "session_id": "session-1",
            "user_id": 42,
        }

        self._nodes.mem_post_node(state)  # type: ignore[arg-type]

        assert len(fake_memory.add_text_calls) == 1
        call = fake_memory.add_text_calls[0]
        assert call["metadata"]["memory_kind"] == "semantic"
        assert call["metadata"]["status"] == "active"
        assert call["metadata"]["reusable"] is True
        assert "Durable user note" in call["text"]

    def test_mem_post_marks_failed_episode_not_reusable(self, monkeypatch) -> None:
        fake_memory = _FakeAgentMemory()
        monkeypatch.setattr("agent.nodes.agent_memory", fake_memory)

        state = {
            "task_desc": "check weather",
            "normalized_task": "check weather",
            "intent": {"tool_name": "weather", "tool_input": {"city": "Beijing"}},
            "tool_result": "",
            "tool_results": [
                {
                    "tool_name": "weather",
                    "status": "failed",
                    "content": "",
                    "error_msg": "provider unavailable",
                }
            ],
            "error_info": "provider unavailable",
            "answer": "weather lookup failed",
            "session_id": "session-1",
            "user_id": 42,
        }

        self._nodes.mem_post_node(state)  # type: ignore[arg-type]

        metadata = fake_memory.add_text_kwargs["metadata"]
        assert metadata["memory_kind"] == "episodic"
        assert metadata["status"] == "failed"
        assert metadata["reusable"] is False
        assert "Error: provider unavailable" in fake_memory.add_text_kwargs["text"]
