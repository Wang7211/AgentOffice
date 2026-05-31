"""Agent 图节点 helper 方法测试。"""

from agent.nodes import AgentNodes


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
