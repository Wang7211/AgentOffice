"""Agent 图节点 helper 方法测试（不依赖 LLM 调用）。"""

import pytest

from agent.nodes import AgentNodes
from agent.state import AgentState


def _make_state(overrides: dict | None = None) -> AgentState:
    """构建最小 AgentState 用于测试。"""
    defaults: AgentState = {
        "messages": [],
        "task_desc": "",
        "normalized_task": "",
        "intent": {},
        "constraints": {},
        "task_status": "received",
        "boundary": {},
        "clarification_question": "",
        "relevant_memories": [],
        "plan": [],
        "tool_name": "",
        "tool_input": {},
        "tool_result": "",
        "tool_calls": [],
        "tool_results": [],
        "step_count": 0,
        "error_info": "",
        "answer": "",
        "reflection": {},
        "reflection_retry_count": 0,
        "archived_memory_ids": [],
        "session_id": "test-session",
    }
    if overrides:
        defaults.update(overrides)
    return defaults


class TestAgentNodesHelpers:
    def setup_method(self) -> None:
        self._nodes = AgentNodes()

    # ------------------------------------------------------------------
    # 能力边界检测
    # ------------------------------------------------------------------

    def test_detect_missing_capability_train_ticket(self) -> None:
        """检测到高铁票查询但无专用工具。"""
        state = _make_state({"task_desc": "查一下明天北京到上海的高铁票"})
        result = self._nodes._detect_missing_capability(state, [{"name": "search"}])
        assert result is not None
        assert result["resource_type"] == "train_ticket"
        assert result["capability"] == "realtime_inventory_booking"

    def test_detect_no_capability_issue_for_simple_query(self) -> None:
        """简单查询不应触发能力缺失。"""
        state = _make_state({"task_desc": "今天天气怎么样"})
        result = self._nodes._detect_missing_capability(state, [{"name": "search"}])
        assert result is None

    def test_detect_with_specialized_tool_available(self) -> None:
        """已注册专用工具时不应返回能力缺失。"""
        state = _make_state({"task_desc": "查高铁票"})
        result = self._nodes._detect_missing_capability(
            state,
            [{"name": "search"}, {"name": "mcp_12306_ticket", "description": "12306 票务工具"}],
        )
        assert result is None

    # ------------------------------------------------------------------
    # 路由 / 表达式匹配
    # ------------------------------------------------------------------

    def test_matches_realtime_inventory_request(self) -> None:
        assert (
            self._nodes._matches_realtime_inventory_request(
                "查高铁票", ("高铁票", "火车票"), ("查", "查询")
            )
            is True
        )

    def test_matches_realtime_inventory_no_match(self) -> None:
        assert (
            self._nodes._matches_realtime_inventory_request(
                "今天天气", ("高铁票", "火车票"), ("查", "查询")
            )
            is False
        )

    # ------------------------------------------------------------------
    # 工具集合检查
    # ------------------------------------------------------------------

    def test_has_specialized_tool(self) -> None:
        assert (
            self._nodes._has_specialized_tool(
                [{"name": "mcp_train_ticket", "description": "铁路票务 MCP"}],
                ("train_ticket", "ticket", "铁路"),
            )
            is True
        )

    def test_has_specialized_tool_generic_not_counted(self) -> None:
        assert (
            self._nodes._has_specialized_tool(
                [{"name": "search"}, {"name": "browser"}],
                ("train_ticket",),
            )
            is False
        )

    # ------------------------------------------------------------------
    # 实体抽取
    # ------------------------------------------------------------------

    def test_extract_route(self) -> None:
        origin, dest = self._nodes._extract_route("从北京到上海的高铁")
        assert origin == "北京"
        assert dest == "上海"

    def test_extract_route_no_match(self) -> None:
        origin, dest = self._nodes._extract_route("你好")
        assert origin == ""
        assert dest == ""

    def test_extract_destination_hint(self) -> None:
        dest = self._nodes._extract_destination_hint("在北京订酒店")
        assert dest == "北京"

    def test_extract_destination_hint_city_fallback(self) -> None:
        dest = self._nodes._extract_destination_hint("有什么好酒店")
        for city in ("北京", "上海", "广州"):
            if city in dest:
                break
        else:
            assert dest == ""

    def test_extract_relative_date_text_today(self) -> None:
        result = self._nodes._extract_relative_date_text("今天去")
        assert "今天" in result

    def test_extract_relative_date_text_tomorrow(self) -> None:
        result = self._nodes._extract_relative_date_text("明天去")
        assert "明天" in result

    def test_extract_time_preference(self) -> None:
        assert self._nodes._extract_time_preference("下午的航班") == "下午"
        assert self._nodes._extract_time_preference("没有时间词") == ""

    # ------------------------------------------------------------------
    # 搜索查询增强
    # ------------------------------------------------------------------

    def test_enrich_search_query_adds_context(self) -> None:
        state = _make_state({
            "task_desc": "查门票",
            "messages": [{"role": "user", "content": "北京故宫有什么好玩的"}],
        })
        step = {"tool_name": "search", "tool_input": {"query": "门票价格"}}
        self._nodes._enrich_search_query(state, step)
        assert "北京" in step["tool_input"]["query"]

    def test_enrich_search_query_already_has_city(self) -> None:
        state = _make_state({
            "task_desc": "上海迪士尼门票",
            "messages": [],
        })
        step = {"tool_name": "search", "tool_input": {"query": "上海迪士尼门票"}}
        self._nodes._enrich_search_query(state, step)
        assert step["tool_input"]["query"] == "上海迪士尼门票"

    # ------------------------------------------------------------------
    # 任务是否需要澄清
    # ------------------------------------------------------------------

    def test_task_needs_clarification_vague_travel(self) -> None:
        """模糊的行程规划应需要澄清。"""
        state = _make_state({
            "task_desc": "帮我规划路线",
            "intent": {"intent_category": "task_execution"},
        })
        assert self._nodes._task_needs_clarification(state) is True

    def test_task_needs_clarification_specific_location(self) -> None:
        """包含地点的不应需要澄清。"""
        state = _make_state({
            "task_desc": "帮我规划北京旅游路线",
            "intent": {"intent_category": "task_execution"},
        })
        assert self._nodes._task_needs_clarification(state) is False

    def test_task_needs_clarification_chat(self) -> None:
        """闲聊不应需要澄清。"""
        state = _make_state({
            "task_desc": "你好",
            "intent": {"intent_category": "interaction_chat"},
        })
        assert self._nodes._task_needs_clarification(state) is False

    # ------------------------------------------------------------------
    # 上下文是否充足
    # ------------------------------------------------------------------

    def test_has_sufficient_context_with_location(self) -> None:
        state = _make_state({
            "task_desc": "规划路线",
            "messages": [{"role": "user", "content": "我想去北京玩"}],
        })
        assert self._nodes._has_sufficient_context(state) is True

    def test_has_sufficient_context_empty(self) -> None:
        state = _make_state({"task_desc": "规划路线", "messages": []})
        assert self._nodes._has_sufficient_context(state) is False

    # ------------------------------------------------------------------
    # 能力缺失回答构建
    # ------------------------------------------------------------------

    def test_build_capability_missing_answer(self) -> None:
        state = _make_state({
            "task_desc": "查高铁票",
            "boundary": {
                "missing_capability": {
                    "capability": "realtime_inventory_booking",
                    "resource_type": "train_ticket",
                    "resource_label": "高铁/火车票",
                    "reason": "无专用工具",
                    "origin": "北京",
                    "destination": "上海",
                    "travel_date": "明天",
                },
            },
        })
        answer = self._nodes._build_capability_missing_answer(state)
        assert "高铁" in answer or "火车" in answer
        assert "北京" in answer
        assert "上海" in answer

    # ------------------------------------------------------------------
    # 工具调用辅助方法
    # ------------------------------------------------------------------

    def test_resolve_template_vars(self) -> None:
        tool_input = {"query": "上一步结果: {{step_1}}"}
        tool_results = [{"step_id": "step_1", "content": "42"}]
        resolved = self._nodes._resolve_template_vars(tool_input, tool_results)
        assert "{{step_1}}" not in resolved["query"]
        assert "42" in resolved["query"]

    def test_fuse_tool_results(self) -> None:
        tool_results = [
            {"status": "success", "tool_name": "search", "content": "结果A"},
            {"status": "failed", "tool_name": "browser", "content": ""},
            {"status": "success", "tool_name": "time", "content": "结果C"},
        ]
        fused = self._nodes._fuse_tool_results(tool_results)
        assert "结果A" in fused
        assert "结果C" in fused
        assert "结果B" not in fused

    def test_first_tool_step(self) -> None:
        plan = [
            {"id": "understand", "tool_name": ""},
            {"id": "tool_1", "tool_name": "search"},
        ]
        step = self._nodes._first_tool_step(plan)
        assert step["tool_name"] == "search"

    def test_first_tool_step_none(self) -> None:
        plan = [{"id": "understand", "tool_name": ""}]
        assert self._nodes._first_tool_step(plan) is None

    # ------------------------------------------------------------------
    # 计划工具步骤
    # ------------------------------------------------------------------

    def test_first_clarification_step(self) -> None:
        plan = [
            {"id": "step_1", "name": "确认目的地", "status": "pending"},
        ]
        step = self._nodes._first_clarification_step(plan)
        assert step is not None

    def test_first_clarification_step_completed_skipped(self) -> None:
        plan = [
            {"id": "step_1", "name": "确认目的地", "status": "completed"},
        ]
        assert self._nodes._first_clarification_step(plan) is None

    # ------------------------------------------------------------------
    # 建议生成
    # ------------------------------------------------------------------

    def test_add_proactive_suggestion_travel(self) -> None:
        state = _make_state({
            "task_status": "completed",
            "task_desc": "北京旅游景点",
            "intent": {"intent_category": "information_inquiry"},
            "answer": "故宫、天安门都是著名景点。" * 20,
            "tool_results": [{"status": "success"}],
        })
        self._nodes._add_proactive_suggestion(state)
        assert "日程表" in state["answer"]

    def test_add_proactive_suggestion_no_marker(self) -> None:
        """已有建议标记时不重复添加。"""
        state = _make_state({
            "task_status": "completed",
            "task_desc": "北京旅游景点",
            "intent": {"intent_category": "information_inquiry"},
            "answer": "故宫很好。是否需要我帮你整理行程？",
            "tool_results": [{"status": "success"}],
        })
        self._nodes._add_proactive_suggestion(state)
        # 不应重复 append

    # ------------------------------------------------------------------
    # 动作 / 交互聊天辅助方法
    # ------------------------------------------------------------------

    def test_is_suggestion_affirmative(self) -> None:
        state = _make_state({
            "normalized_task": "需要",
            "messages": [{"role": "assistant", "content": "是否需要我帮你查一下"}],
        })
        assert self._nodes._is_suggestion_affirmative(state) is True

    def test_is_suggestion_affirmative_too_long(self) -> None:
        state = _make_state({
            "normalized_task": "我需要很多帮助请帮我查询",
            "messages": [],
        })
        assert self._nodes._is_suggestion_affirmative(state) is False

    # ------------------------------------------------------------------
    # 娱乐回答
    # ------------------------------------------------------------------

    def test_build_light_entertainment_answer_returns_joke(self) -> None:
        state = _make_state({
            "task_desc": "讲个笑话",
            "messages": [],
        })
        answer = self._nodes._build_light_entertainment_answer(state)
        assert len(answer) > 0

    # ------------------------------------------------------------------
    # 标记计划步骤
    # ------------------------------------------------------------------

    def test_mark_plan_step(self) -> None:
        plan = [{"id": "step_1", "status": "pending"}]
        self._nodes._mark_plan_step(plan, "step_1", "completed")
        assert plan[0]["status"] == "completed"

    def test_mark_plan_step_not_found(self) -> None:
        plan = [{"id": "step_1", "status": "pending"}]
        self._nodes._mark_plan_step(plan, "nonexistent", "completed")
        assert plan[0]["status"] == "pending"

    # ------------------------------------------------------------------
    # 提示词（_build 类方法）
    # ------------------------------------------------------------------

    def test_build_capability_overview_answer(self) -> None:
        answer = self._nodes._build_capability_overview_answer()
        assert "信息问询" in answer or "信息" in answer
        assert "任务执行" in answer

    def test_build_interaction_chat_answer_greeting(self) -> None:
        state = _make_state({"task_desc": "你好", "intent": {"intent_subtype": ""}})
        answer = self._nodes._build_interaction_chat_answer(state)
        assert "你好" in answer

    def test_build_interaction_chat_answer_thanks(self) -> None:
        state = _make_state({"task_desc": "谢谢", "intent": {"intent_subtype": ""}})
        answer = self._nodes._build_interaction_chat_answer(state)
        assert answer == "不客气。"

    def test_build_interaction_chat_answer_farewell(self) -> None:
        state = _make_state({"task_desc": "再见", "intent": {"intent_subtype": ""}})
        answer = self._nodes._build_interaction_chat_answer(state)
        assert answer == "再见。"
