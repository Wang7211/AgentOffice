"""LocalModelClient 本地规则模型测试。"""

from typing import Any

from services.llm_service import LocalModelClient


class TestLocalModelClient:
    def setup_method(self) -> None:
        self._client = LocalModelClient()

    # ------------------------------------------------------------------
    # 意图分类
    # ------------------------------------------------------------------

    def test_analyze_capability_question(self) -> None:
        """用户询问能力边界，不应触发工具。"""
        result = self._client.analyze_task("你能做什么", [])
        assert result["tool_name"] == ""

    def test_analyze_weather_intent(self) -> None:
        result = self._client.analyze_task("北京天气怎么样", [])
        assert result["tool_name"] == "weather"

    def test_analyze_time_intent(self) -> None:
        result = self._client.analyze_task("现在几点了", [])
        assert result["tool_name"] == "time"

    def test_analyze_calculation_intent(self) -> None:
        result = self._client.analyze_task("计算 25 * 4 + 10", [])
        assert result["tool_name"] == "code"

    def test_analyze_chat_intent(self) -> None:
        """简单问候不应触发工具。"""
        result = self._client.analyze_task("你好", [])
        assert result["tool_name"] == ""

    def test_analyze_math_expression_detection(self) -> None:
        """带运算符和数字的消息应识别为计算。"""
        result = self._client.analyze_task("3 + 5", [])
        assert result["tool_name"] == "code"

    def test_analyze_light_entertainment(self) -> None:
        result = self._client.analyze_task("讲个笑话", [])
        # 娱乐请求不应触发工具
        assert result["tool_name"] == ""

    def test_farewell_chat(self) -> None:
        result = self._client.analyze_task("再见", [])
        assert result["tool_name"] == ""

    # ------------------------------------------------------------------
    # 约束抽取
    # ------------------------------------------------------------------

    def test_extract_file_path_windows(self) -> None:
        constraints = self._client._extract_constraints("读取 C:\\docs\\test.pdf")
        entities = constraints["entities"]["file_paths"]
        assert any("C:\\docs\\test.pdf" in e for e in entities)

    def test_extract_dates(self) -> None:
        constraints = self._client._extract_constraints("报告 2024-01-15 内容")
        assert len(constraints["dates"]) >= 1
        assert "2024-01-15" in constraints["dates"][0]

    # ------------------------------------------------------------------
    # 生成 / 计划 / 反思 / 记忆（本地模型确定性行为）
    # ------------------------------------------------------------------

    def test_generate_directly(self) -> None:
        answer = self._client.generate("你好吗", context=[{"role": "user", "content": "你好吗"}])
        assert "当前运行在本地规则模型模式" in answer or "我已结合之前对话理解你的问题" in answer

    def test_generate_with_tool_result(self) -> None:
        answer = self._client.generate(
            "查询结果如何",
            tool_result="温度 25°C，湿度 60%",
            context=[],
        )
        assert "已完成工具调用" in answer
        assert "温度 25°C" in answer

    def test_create_plan_search(self) -> None:
        analysis = {
            "tool_name": "search",
            "tool_input": {"query": "北京天气"},
            "normalized_task": "北京天气",
        }
        plan = self._client.create_plan(
            analysis=analysis,
            memories=[],
            tool_specs=[{"name": "search", "description": "搜索"}],
        )
        step_names = [s["name"] for s in plan]
        assert any("search" in n for n in step_names)

    def test_create_plan_no_tool(self) -> None:
        analysis = {
            "tool_name": "",
            "tool_input": {},
            "normalized_task": "你好",
        }
        plan = self._client.create_plan(
            analysis=analysis,
            memories=[],
            tool_specs=[],
        )
        assert len(plan) == 3  # understand + plan + action

    # ------------------------------------------------------------------
    # 工具和业务规则检测
    # ------------------------------------------------------------------

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
