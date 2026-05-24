"""LocalModelClient 本地规则模型测试。"""

from typing import Any

import pytest

from services.llm_service import LocalModelClient


class TestLocalModelClient:
    def setup_method(self) -> None:
        self._client = LocalModelClient()

    # ------------------------------------------------------------------
    # 意图分类
    # ------------------------------------------------------------------

    def test_analyze_capability_question(self) -> None:
        """用户询问能力边界。"""
        result = self._client.analyze_task("你能做什么", [])
        assert result["intent_category"] == "information_inquiry"
        assert result["intent_subtype"] == "capability_question"

    def test_analyze_search_intent(self) -> None:
        result = self._client.analyze_task("搜索今天的新闻", [])
        assert result["intent"]["tool_name"] == "search"

    def test_analyze_time_intent(self) -> None:
        result = self._client.analyze_task("现在几点了", [])
        assert result["intent"]["tool_name"] == "time"

    def test_analyze_calculation_intent(self) -> None:
        result = self._client.analyze_task("计算 25 * 4 + 10", [])
        assert result["intent"]["tool_name"] == "code"

    def test_analyze_chat_intent(self) -> None:
        result = self._client.analyze_task("你好", [])
        assert result["intent_category"] == "interaction_chat"

    def test_analyze_math_expression_detection(self) -> None:
        """带运算符和数字的消息应识别为计算。"""
        result = self._client.analyze_task("3 + 5", [])
        assert result["intent"]["tool_name"] == "code"

    def test_analyze_light_entertainment(self) -> None:
        result = self._client.analyze_task("讲个笑话", [])
        assert result["intent_category"] == "interaction_chat"
        assert result["intent_subtype"] == "light_entertainment"

    def test_farewell_chat(self) -> None:
        result = self._client.analyze_task("再见", [])
        assert result["intent_category"] == "interaction_chat"
        assert result["intent_subtype"] == "farewell"

    # ------------------------------------------------------------------
    # 意图识别细化
    # ------------------------------------------------------------------

    def test_identify_intent_search(self) -> None:
        intent = self._client._identify_intent_by_rules("搜索北京天气")
        assert intent["tool_name"] == "search"

    def test_identify_intent_time(self) -> None:
        intent = self._client._identify_intent_by_rules("今天日期")
        assert intent["tool_name"] == "time"

    def test_identify_intent_no_match(self) -> None:
        intent = self._client._identify_intent_by_rules("你好")
        assert intent["tool_name"] == ""

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
        assert "当前运行在本地规则模型模式" in answer or "我已结合当前会话上下文" in answer

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
            "intent": {"tool_name": "search", "tool_input": {"query": "北京天气"}},
            "normalized_task": "北京天气",
            "intent_category": "information_inquiry",
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
            "intent": {"tool_name": "", "tool_input": {}},
            "normalized_task": "你好",
        }
        plan = self._client.create_plan(
            analysis=analysis,
            memories=[],
            tool_specs=[],
        )
        assert len(plan) == 4  # understand + plan + action + reflection

    def test_reflect_success(self) -> None:
        reflection = self._client.reflect(
            user_message="查询天气",
            answer="今天 25°C",
            plan=[],
            tool_results=[],
        )
        assert reflection["score"] >= 0
        assert reflection["status"] in ("success", "partial_success", "failed")

    def test_reflect_with_errors(self) -> None:
        reflection = self._client.reflect(
            user_message="查询天气",
            answer="",
            plan=[],
            tool_results=[],
            error_info="工具调用失败",
        )
        assert reflection["score"] < 0.6
        assert "工具调用失败" in reflection["retry_reason"]

    def test_extract_memory_archive_worthy(self) -> None:
        memories = self._client.extract_memory(
            user_message="帮我查资料",
            answer="这是查到的资料内容，有很长的篇幅" * 20,
            reflection={"archive_worthy": True, "score": 0.86},
        )
        assert len(memories) == 1
        assert "任务经验" in memories[0]

    def test_extract_memory_not_worthy(self) -> None:
        memories = self._client.extract_memory(
            user_message="hi",
            answer="hello",
            reflection={"archive_worthy": False, "score": 0.2},
        )
        assert memories == []

    # ------------------------------------------------------------------
    # 工具和业务规则检测
    # ------------------------------------------------------------------

    def test_realtime_inventory_booking_request(self) -> None:
        assert self._client._is_realtime_inventory_booking_request("查一下北京到上海的高铁票") is True
        assert self._client._is_realtime_inventory_booking_request("今天天气怎么样") is False

    def test_is_search_query(self) -> None:
        assert self._client._is_search_query("搜索人工智能") is True
        assert self._client._is_search_query("你好") is False

    def test_is_light_entertainment_chat(self) -> None:
        assert self._client._is_light_entertainment_chat("讲个笑话") is True
        assert self._client._is_light_entertainment_chat("你好") is False

    def test_is_capability_question(self) -> None:
        assert self._client._is_capability_question("你能做什么") is True
        assert self._client._is_capability_question("今天天气") is False

    def test_is_pure_information_question(self) -> None:
        assert self._client._is_pure_information_question("什么是人工智能") is True
        assert self._client._is_pure_information_question("生成一份报告") is False

    def test_classify_intent_category_chat(self) -> None:
        category = self._client._classify_intent_category("你好", {})
        assert category == "interaction_chat"

    def test_classify_intent_category_task(self) -> None:
        category = self._client._classify_intent_category("生成一份报告", {"entities": {}})
        assert category == "task_execution"

    def test_classify_intent_category_info(self) -> None:
        category = self._client._classify_intent_category("什么是 Python", {"entities": {}})
        assert category == "information_inquiry"

    def test_is_interaction_chat(self) -> None:
        assert self._client._is_interaction_chat("你好") is True
        assert self._client._is_interaction_chat("再见") is True
        assert self._client._is_interaction_chat("帮我查资料") is False

    def test_contextual_followup(self) -> None:
        # 需要至少 2 条历史消息（最后一条需为 assistant 且含问号）
        assert (
            self._client._is_clarification_followup(
                "北京",
                [
                    {"role": "user", "content": "查天气"},
                    {"role": "assistant", "content": "请补充目的地?"},
                ],
            )
            is True
        )
        assert (
            self._client._is_clarification_followup("很长的消息不应该是追问", [])
            is False
        )

    def test_merge_with_previous_task(self) -> None:
        merged = self._client._merge_with_previous_task(
            "北京",
            [{"role": "user", "content": "查询明天的天气"}, {"role": "assistant", "content": "请问哪个城市?"}],
        )
        assert "查询明天的天气" in merged
        assert "北京" in merged

    def test_infer_chat_subtype(self) -> None:
        assert self._client._infer_chat_subtype("谢谢") == "social_ack"
        assert self._client._infer_chat_subtype("再见") == "farewell"
        assert self._client._infer_chat_subtype("你好") == "greeting"
