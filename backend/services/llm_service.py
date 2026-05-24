"""模型抽象层。"""

import json
import re
from typing import Any

import httpx
from loguru import logger

from config.settings import get_settings


IntentResult = dict[str, Any]
TaskAnalysis = dict[str, Any]
PlanStep = dict[str, Any]
ReflectionResult = dict[str, Any]


class LocalModelClient:
    """面向私有化部署的本地确定性模型兜底实现。"""

    def generate(
        self,
        user_message: str,
        tool_result: str | None = None,
        context: list[dict[str, str]] | None = None,
        memories: list[dict[str, Any]] | None = None,
        plan: list[PlanStep] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> str:
        """生成助手回答。

        参数:
            user_message: 最新用户消息。
            tool_result: 可选的工具执行结果。
            context: 最近聊天上下文。
            memories: 已检索的长期记忆片段。
            plan: 已定型的执行计划。
            tool_results: 多工具执行结果列表。
        返回:
            助手回复文本。
        异常:
            无。
        """
        if tool_results and not tool_result:
            merged_lines: list[str] = []
            for idx, r in enumerate(tool_results, start=1):
                if r.get("status") != "success":
                    continue
                content = str(r.get("content") or "").strip()
                if content:
                    merged_lines.append(f"[工具{idx}:{r.get('tool_name', 'unknown')}]\n{content}")
            tool_result = "\n\n".join(merged_lines)
        if tool_result:
            return self._answer_with_tool(user_message, tool_result)
        return self._answer_directly(user_message, context or [])

    def analyze_task(
        self,
        message: str,
        context: list[dict[str, str]] | None = None,
    ) -> TaskAnalysis:
        """将自然语言任务规整为标准化意图数据。"""
        context = context or []
        if self._is_clarification_followup(message, context):
            message = self._merge_with_previous_task(message, context)
        normalized_message = self._normalize_message(message)
        constraints = self._extract_constraints(normalized_message)
        context_text = self._context_text(context)
        rule_category = self._classify_intent_category(
            normalized_message,
            constraints,
            context_text,
        )
        fast_intent = self._fast_intent_for_obvious_cases(
            normalized_message,
            rule_category,
            context_text,
        )
        intent = fast_intent or self.identify_intent(normalized_message)
        if intent.get("tool_name") == "knowledge" and self._is_public_info_query(
            normalized_message,
        ):
            intent["tool_name"] = "search"
            intent["tool_input"] = {"query": normalized_message}
        intent_category = self._normalize_intent_category(
            intent.get("intent_category"),
        ) or rule_category
        if rule_category == "interaction_chat":
            intent_category = rule_category
            intent["tool_name"] = ""
            intent["tool_input"] = {}
        intent_subtype = str(
            intent.get("intent_subtype")
            or self._infer_intent_subtype(
                intent.get("tool_name", ""),
                normalized_message,
                context_text,
            ),
        )
        needs_clarification = self._needs_clarification(
            normalized_message,
            context,
            constraints,
            intent_category,
        )
        return {
            "normalized_task": normalized_message,
            "intent_type": intent_category,
            "intent_category": intent_category,
            "intent_subtype": intent_subtype,
            "intent": intent,
            "constraints": constraints,
            "entities": constraints.get("entities", {}),
            "needs_clarification": needs_clarification,
            "clarification_question": self._build_clarification_question(
                normalized_message,
                constraints,
            ),
            "confidence": self._intent_confidence(intent_category, intent),
            "provider": intent.get("provider", "local"),
        }

    def _fast_intent_for_obvious_cases(
        self,
        message: str,
        rule_category: str,
        context_text: str = "",
    ) -> IntentResult | None:
        """明显无需模型判断的请求直接本地路由，避免外部模型延迟。"""
        if self._is_capability_question(message):
            return {
                "intent_category": "information_inquiry",
                "tool_name": "",
                "tool_input": {},
                "intent_source": "rule_fast_path",
                "provider": "local",
            }
        if self._is_light_entertainment_chat(
            message,
        ) or self._is_contextual_light_entertainment_followup(message, context_text):
            return {
                "intent_category": "interaction_chat",
                "intent_subtype": "light_entertainment",
                "tool_name": "",
                "tool_input": {},
                "intent_source": "rule_fast_path",
                "provider": "local",
            }
        if rule_category == "interaction_chat":
            return {
                "intent_category": "interaction_chat",
                "intent_subtype": self._infer_chat_subtype(message, context_text),
                "tool_name": "",
                "tool_input": {},
                "intent_source": "rule_fast_path",
                "provider": "local",
            }
        return None

    def create_plan(
        self,
        analysis: TaskAnalysis,
        memories: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        tool_results: list[dict[str, Any]] | None = None,
    ) -> list[PlanStep]:
        """根据意图、记忆和工具能力生成可调度计划。"""
        if tool_results is not None:
            pass  # 签名兼容，LLM 子类中使用
        intent = dict(analysis.get("intent") or {})
        tool_name = str(intent.get("tool_name") or "")
        tool_input = dict(intent.get("tool_input") or {})
        available_tools = {str(spec.get("name")) for spec in tool_specs}
        if tool_name and tool_name not in available_tools:
            tool_name = ""
            tool_input = {}

        plan: list[PlanStep] = [
            {
                "id": "understand",
                "phase": "memory",
                "name": "任务理解与边界校验",
                "goal": "确认用户诉求、关键约束和可执行范围。",
                "depends_on": [],
                "status": "completed",
            },
            {
                "id": "plan",
                "phase": "planning",
                "name": "子目标拆解与顺序排布",
                "goal": "将任务整理为可顺序调度的执行步骤。",
                "depends_on": ["understand"],
                "status": "completed",
            },
        ]
        if tool_name:
            plan.append(
                {
                    "id": "tool_1",
                    "phase": "tools",
                    "name": f"调用 {tool_name} 工具",
                    "goal": "获取完成任务所需的外部数据或计算结果。",
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "depends_on": ["plan"],
                    "status": "pending",
                }
            )
        plan.extend(
            [
                {
                    "id": "action",
                    "phase": "action",
                    "name": "信息融合与结果生成",
                    "goal": "结合计划、记忆和工具结果输出最终答复。",
                    "depends_on": ["tool_1"] if tool_name else ["plan"],
                    "status": "pending",
                },
                {
                    "id": "reflection",
                    "phase": "reflection",
                    "name": "反思复盘与记忆归档",
                    "goal": "评估执行效果并抽取可复用经验。",
                    "depends_on": ["action"],
                    "status": "pending",
                },
            ]
        )
        return plan

    def reflect(
        self,
        user_message: str,
        answer: str,
        plan: list[PlanStep],
        tool_results: list[dict[str, Any]],
        error_info: str = "",
    ) -> ReflectionResult:
        """对本轮任务执行效果进行本地复盘。"""
        success = bool(answer) and not error_info
        failed_tools = [
            result
            for result in tool_results
            if result.get("status") and result.get("status") != "success"
        ]
        issues: list[str] = []
        if error_info:
            issues.append(error_info)
        if failed_tools:
            issues.append("存在工具调用失败，需要检查参数或外部依赖。")
        no_match_knowledge = [
            result
            for result in tool_results
            if result.get("tool_name") == "knowledge"
            and not dict(result.get("metadata") or {}).get("matches")
        ]
        if no_match_knowledge:
            issues.append("知识库未命中相关内容，需要换更具体的问题、上传相关文档或跳过知识库。")
        if not tool_results and self._identify_intent_by_rules(user_message).get(
            "tool_name"
        ):
            issues.append("任务可能需要外部能力，但本轮未获得工具结果。")
        is_reliable = success and not issues
        status = "success" if is_reliable else "partial_success" if success else "failed"
        return {
            "score": 0.86 if is_reliable else 0.45,
            "status": status,
            "issues": issues,
            "improvements": ["继续补齐可用工具和长期记忆。"] if issues else [],
            "requires_retry": bool(issues),
            "retry_reason": "；".join(issues),
            "archive_worthy": is_reliable and (bool(tool_results) or len(answer) >= 80),
        }

    def extract_memory(
        self,
        user_message: str,
        answer: str,
        reflection: ReflectionResult,
    ) -> list[str]:
        """从本轮对话中抽取可写入长期记忆的片段。"""
        if not reflection.get("archive_worthy"):
            return []
        content = (
            "任务经验："
            f"用户需求={self._normalize_message(user_message)[:220]}；"
            f"交付结果摘要={self._normalize_message(answer)[:320]}"
        )
        return [content]

    def identify_intent(self, message: str) -> IntentResult:
        """识别是否需要调用工具。

        本地模型只作为兜底逻辑使用；外部模型客户端会覆盖此方法，优先使用大模型意图识别。
        """
        return self._identify_intent_by_rules(message)

    def _normalize_message(self, message: str) -> str:
        """清洗口语化任务文本。"""
        cleaned_message = message.replace("\u200b", "")
        cleaned_message = re.sub(r"\s+", " ", cleaned_message)
        return cleaned_message.strip()

    def _extract_constraints(self, message: str) -> dict[str, Any]:
        """用轻量规则抽取时间、对象、文件等关键约束。"""
        windows_paths = re.findall(r"[A-Za-z]:\\[^\s，。；,;]+", message)
        unix_paths = re.findall(r"(?<!\S)/(?:[A-Za-z_.~][^\s，。；,;]*)", message)
        file_paths = windows_paths + unix_paths
        dates = re.findall(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?", message)
        partial_dates = re.findall(r"(?<!\d)\d{1,2}[-/.月]\d{1,2}日?(?!\d)", message)
        relative_dates = [
            keyword
            for keyword in (
                "今天",
                "明天",
                "后天",
                "昨天",
                "本周",
                "本月",
                "下周",
                "下个月",
                "最新",
                "实时",
            )
            if keyword in message
        ]
        durations = re.findall(r"\d+\s*(?:天|日|周|个月|月|小时)", message)
        quoted_entities = re.findall(r"[“\"']([^”\"']{2,40})[”\"']", message)
        return {
            "entities": {"file_paths": file_paths, "quoted": quoted_entities},
            "dates": dates + partial_dates,
            "relative_dates": relative_dates,
            "durations": durations,
            "scope": "explicit" if file_paths or dates or relative_dates else "general",
        }

    def _infer_intent_subtype(
        self,
        tool_name: str,
        message: str = "",
        context_text: str = "",
    ) -> str:
        """根据工具路由推断细分意图类型。"""
        if self._is_capability_question(message):
            return "capability_question"
        if self._is_light_entertainment_chat(
            message,
        ) or self._is_contextual_light_entertainment_followup(message, context_text):
            return "light_entertainment"
        if self._is_realtime_inventory_booking_request(message):
            return "realtime_inventory_booking"
        mapping = {
            "search": "search",
            "code": "calculation",
            "file": "file_parse",
            "time": "time",
            "knowledge": "knowledge_query",
        }
        return mapping.get(tool_name, "chat")

    def _classify_intent_category(
        self,
        message: str,
        constraints: dict[str, Any],
        context_text: str = "",
    ) -> str:
        """将用户诉求归入信息问询、任务执行、交互闲聊三大类。"""
        if self._is_capability_question(message):
            return "information_inquiry"
        if self._is_contextual_light_entertainment_followup(message, context_text):
            return "interaction_chat"
        if self._is_interaction_chat(message):
            return "interaction_chat"
        if self._is_task_execution_request(message, constraints):
            return "task_execution"
        return "information_inquiry"

    def _normalize_intent_category(self, value: Any) -> str:
        """兼容模型可能返回的中英文顶层意图名称。"""
        normalized = str(value or "").strip().lower()
        mapping = {
            "information": "information_inquiry",
            "info": "information_inquiry",
            "information_inquiry": "information_inquiry",
            "信息问询": "information_inquiry",
            "query": "information_inquiry",
            "qa": "information_inquiry",
            "task": "task_execution",
            "task_execution": "task_execution",
            "execution": "task_execution",
            "任务执行": "task_execution",
            "execute": "task_execution",
            "chat": "interaction_chat",
            "smalltalk": "interaction_chat",
            "interaction_chat": "interaction_chat",
            "casual_chat": "interaction_chat",
            "交互闲聊": "interaction_chat",
            "闲聊": "interaction_chat",
        }
        return mapping.get(normalized, "")

    def _intent_confidence(self, intent_category: str, intent: IntentResult) -> float:
        """为结构化意图提供稳定的置信度。"""
        if intent.get("intent_source") == "rule_fast_path":
            return 0.9
        if intent_category == "interaction_chat":
            return 0.9
        if intent_category == "task_execution":
            return 0.78
        return 0.74 if intent.get("tool_name") else 0.66

    def _is_realtime_inventory_booking_request(self, message: str) -> bool:
        """识别需要实时库存、价格或预订状态的查询。"""
        inventory_keywords = (
            "高铁票",
            "火车票",
            "动车票",
            "列车票",
            "12306",
            "铁路票",
            "机票",
            "航班",
            "飞机",
            "航空",
            "酒店",
            "住宿",
            "房间",
            "房态",
            "房价",
        )
        action_keywords = (
            "查",
            "查询",
            "看一下",
            "看看",
            "买",
            "订",
            "预订",
            "预定",
            "余票",
            "票价",
            "车次",
            "舱位",
            "价格",
            "可订",
        )
        return any(keyword in message for keyword in inventory_keywords) and any(
            keyword in message for keyword in action_keywords
        )

    def _needs_clarification(
        self,
        message: str,
        context: list[dict[str, str]],
        constraints: dict[str, Any],
        intent_category: str = "",
    ) -> bool:
        """判断任务是否缺少最小执行信息。"""
        if intent_category in {"interaction_chat", "information_inquiry"}:
            return False
        vague_messages = {"处理一下", "帮我处理", "优化一下", "做一下", "分析一下"}
        if message in vague_messages and not context:
            return True
        references = ("这个", "它", "上面", "刚才", "之前")
        if any(reference in message for reference in references) and not context:
            return True

        context_text = self._context_text(context)
        if self._is_document_generation_request(message):
            has_topic = bool(context_text) or self._has_specific_business_object(
                message,
                constraints,
            )
            return not has_topic
        if self._is_ticket_or_booking_request(message):
            has_target = bool(context_text) or self._has_specific_business_object(
                message,
                constraints,
            )
            has_date = bool(constraints.get("dates") or constraints.get("relative_dates"))
            return not (has_target and has_date)
        if self._is_external_operation_request(message):
            has_target = bool(context_text) or self._has_specific_business_object(
                message,
                constraints,
            )
            return not has_target
        return False

    def _build_clarification_question(
        self,
        message: str,
        constraints: dict[str, Any],
    ) -> str:
        """根据任务类型生成具体澄清问题。"""
        if self._is_document_generation_request(message):
            return (
                "这个文档需要先补齐关键信息：请告诉我文档主题、使用场景、"
                "时间范围/对象，以及希望输出为正文、表格还是可下载文件。"
            )
        if self._is_ticket_or_booking_request(message):
            missing_parts: list[str] = []
            if not self._has_specific_business_object(message, constraints):
                missing_parts.append("目标场馆或官网名称")
            if not (constraints.get("dates") or constraints.get("relative_dates")):
                missing_parts.append("查询日期")
            missing_text = "、".join(missing_parts) or "目标和日期"
            return f"查询余票前还缺少{missing_text}，请补充后我再继续执行。"
        if self._is_external_operation_request(message):
            return "请补充要打开或访问的具体网站、页面 URL、小程序名称或业务对象。"
        return "请补充要处理的具体对象、范围、时间要求或期望输出格式。"

    def _context_text(self, context: list[dict[str, str]]) -> str:
        """拼接最近上下文内容用于澄清判断。"""
        return " ".join(str(item.get("content") or "") for item in context[-6:])

    def _is_clarification_followup(
        self,
        message: str,
        context: list[dict[str, str]],
    ) -> bool:
        """检测当前短消息是否为对上一轮澄清/追问的回答。"""
        if len(message) > 15:
            return False
        if not context or len(context) < 2:
            return False
        last = context[-1]
        if not isinstance(last, dict):
            return False
        role = str(last.get("role", "") or last.get("type", ""))
        content = str(last.get("content", ""))
        if role not in ("assistant", "ai", "model"):
            return False
        return "?" in content or "？" in content or "请补充" in content or "确认" in content

    def _merge_with_previous_task(
        self,
        message: str,
        context: list[dict[str, str]],
    ) -> str:
        """将简短回应与原任务描述合并，使意图分析能理解完整上下文。"""
        last_user_msg = ""
        for item in reversed(context):
            role = str(item.get("role", "") or item.get("type", ""))
            if role in ("user", "human"):
                last_user_msg = str(item.get("content", ""))
                break
        if not last_user_msg:
            return message
        return f"{last_user_msg}（{message}）"

    def _has_specific_business_object(
        self,
        message: str,
        constraints: dict[str, Any],
    ) -> bool:
        """判断消息中是否包含足够具体的业务对象。"""
        if constraints.get("entities", {}).get("file_paths"):
            return True
        if constraints.get("entities", {}).get("quoted"):
            return True
        specific_keywords = (
            "北京",
            "西安",
            "上海",
            "广州",
            "深圳",
            "国博",
            "国家博物馆",
            "故宫",
            "博物院",
            "合同",
            "会议",
            "项目",
            "客户",
            "产品",
            "订单",
            "报销",
            "招聘",
            "新能源",
            "车企",
            "电池",
            "欧盟",
            "法规",
            "合规",
            "供应链",
            "碳足迹",
            "市场准入",
        )
        return any(keyword in message for keyword in specific_keywords)

    def _is_document_generation_request(self, message: str) -> bool:
        """识别需要生成文档但可能缺少主题的任务。"""
        action_keywords = ("生成", "创建", "撰写", "写一份", "输出", "整理")
        document_keywords = (
            "文档",
            "报告",
            "方案",
            "计划",
            "日报",
            "周报",
            "月报",
            "日程表",
            "表格",
            "材料",
        )
        return any(keyword in message for keyword in action_keywords) and any(
            keyword in message for keyword in document_keywords
        )

    def _is_interaction_chat(self, message: str) -> bool:
        """识别无业务目标的寒暄、情绪表达和随口闲聊。"""
        normalized = re.sub(r"[，。！？!?,.\s~～]+", "", message.lower())
        chat_phrases = {
            "你好",
            "您好",
            "早上好",
            "中午好",
            "下午好",
            "晚上好",
            "嗨",
            "hi",
            "hello",
            "在吗",
            "你在吗",
            "谢谢",
            "感谢",
            "辛苦了",
            "哈哈",
            "呵呵",
            "再见",
            "拜拜",
            "聊聊天",
            "随便聊聊",
        }
        if normalized in chat_phrases:
            return True
        if len(normalized) <= 12 and any(phrase in normalized for phrase in chat_phrases):
            return True
        if self._is_light_entertainment_chat(message):
            return True
        return False

    def _infer_chat_subtype(self, message: str, context_text: str = "") -> str:
        """细分交互闲聊，便于路由是否需要上下文。"""
        if self._is_light_entertainment_chat(
            message,
        ) or self._is_contextual_light_entertainment_followup(message, context_text):
            return "light_entertainment"
        normalized = re.sub(r"[，。！？!?,.\s~～]+", "", message.lower())
        if normalized in {"谢谢", "感谢", "辛苦了"}:
            return "social_ack"
        if normalized in {"再见", "拜拜"}:
            return "farewell"
        return "greeting"

    def _is_light_entertainment_chat(self, message: str) -> bool:
        """识别轻量娱乐闲聊请求，不进入任务执行链路。"""
        normalized = re.sub(r"[，。！？!?,.\s~～]+", "", message.lower())
        patterns = (
            "讲个笑话",
            "讲一个笑话",
            "给我讲个笑话",
            "给我讲一个笑话",
            "说个笑话",
            "说一个笑话",
            "来个笑话",
            "逗我笑",
            "讲个段子",
            "说个段子",
        )
        return any(pattern in normalized for pattern in patterns)

    def _is_contextual_light_entertainment_followup(
        self,
        message: str,
        context_text: str,
    ) -> bool:
        """识别依赖上一轮娱乐请求的省略式追问。"""
        normalized = re.sub(r"[，。！？!?,.\s~～]+", "", message.lower())
        followup_patterns = (
            "重新讲一个",
            "重新说一个",
            "再讲一个",
            "再说一个",
            "再来一个",
            "换一个",
            "另一个",
            "讲另一个",
            "说另一个",
        )
        entertainment_context = ("笑话", "段子", "逗我笑", "讲一个办公场景的")
        return any(pattern in normalized for pattern in followup_patterns) and any(
            keyword in context_text for keyword in entertainment_context
        )

    def _is_capability_question(self, message: str) -> bool:
        """识别用户询问系统能力边界的问题。"""
        normalized = re.sub(r"[，。！？!?,.\s~～]+", "", message.lower())
        patterns = (
            "你可以干什么",
            "你能干什么",
            "你会干什么",
            "你有什么功能",
            "你有哪些功能",
            "你能做什么",
            "你可以做什么",
            "你能帮我什么",
            "你能帮我做什么",
            "你有什么能力",
            "介绍一下你的能力",
            "功能介绍",
        )
        return any(pattern in normalized for pattern in patterns)

    def _is_task_execution_request(
        self,
        message: str,
        constraints: dict[str, Any],
    ) -> bool:
        """识别要求智能体交付具体成果或执行事务操作的请求。"""
        if constraints.get("entities", {}).get("file_paths"):
            return True
        if self._is_pure_information_question(message):
            return False
        execution_keywords = (
            "生成",
            "创建",
            "撰写",
            "写一份",
            "写个",
            "编写",
            "制作",
            "整理",
            "汇总",
            "统计",
            "导出",
            "保存",
            "修改",
            "更新",
            "删除",
            "重构",
            "优化",
            "修复",
            "实现",
            "开发",
            "接入",
            "配置",
            "部署",
            "运行",
            "测试",
            "制定",
            "设计",
            "规划",
            "预订",
            "预定",
            "订票",
            "下单",
            "购买",
            "发送",
            "打开",
        )
        deliverable_keywords = (
            "报告",
            "文档",
            "表格",
            "报表",
            "代码",
            "脚本",
            "文件",
            "方案",
            "计划",
            "清单",
            "流程",
            "图表",
            "邮件",
            "摘要",
            "PPT",
            "幻灯片",
        )
        if any(keyword in message for keyword in execution_keywords):
            return True
        return any(keyword in message for keyword in deliverable_keywords) and any(
            keyword in message for keyword in ("做", "写", "给我", "帮我", "输出")
        )

    def _is_pure_information_question(self, message: str) -> bool:
        """识别只要求解释、查询、比较或了解信息的问题。"""
        information_patterns = (
            "什么是",
            "是什么",
            "为什么",
            "怎么理解",
            "如何理解",
            "解释一下",
            "介绍一下",
            "区别",
            "有哪些",
            "多少",
            "查询",
            "查一下",
            "规则",
            "资料",
            "概念",
            "含义",
            "原理",
        )
        transaction_keywords = (
            "预订",
            "预定",
            "订票",
            "下单",
            "购买",
            "生成",
            "创建",
            "制作",
            "写一份",
            "编写",
            "修改",
            "修复",
            "重构",
            "实现",
            "导出",
            "保存",
        )
        return any(pattern in message for pattern in information_patterns) and not any(
            keyword in message for keyword in transaction_keywords
        )

    def _is_ticket_or_booking_request(self, message: str) -> bool:
        """识别票务、预约类任务。"""
        return any(
            keyword in message
            for keyword in (
                "余票",
                "门票",
                "预约",
                "订票",
                "高铁票",
                "火车票",
                "动车票",
                "列车票",
                "12306",
                "铁路票",
                "机票",
                "航班",
                "飞机",
                "航空",
                "酒店",
                "住宿",
                "房间",
                "房态",
                "房价",
            )
        )

    def _is_external_operation_request(self, message: str) -> bool:
        """识别访问外部页面或应用的任务。"""
        return any(keyword in message for keyword in ("打开", "官网", "网页", "小程序"))

    def _is_browser_query(self, message: str) -> bool:
        """检查消息是否需要网页浏览工具。"""
        return bool(self._extract_url(message)) or any(
            keyword in message for keyword in ("打开网页", "访问网页", "浏览网页", "读取网页")
        )

    def _extract_url(self, message: str) -> str:
        """从消息中提取第一个 URL。"""
        match = re.search(r"https?://[^\s，。；,;）)]+", message)
        return match.group(0) if match else ""

    def _identify_intent_by_rules(self, message: str) -> IntentResult:
        """使用本地规则完成兜底意图识别。"""
        normalized_message = message.strip()
        if self._is_search_query(normalized_message):
            return {
                "tool_name": "search",
                "tool_input": {"query": normalized_message},
                "intent_source": "rule",
                "provider": "local",
            }
        if self._is_browser_query(normalized_message):
            url = self._extract_url(normalized_message)
            return {
                "tool_name": "browser" if url else "search",
                "tool_input": {"url": url} if url else {"query": normalized_message},
                "intent_source": "rule",
                "provider": "local",
            }
        if self._is_time_query(normalized_message):
            return {
                "tool_name": "time",
                "tool_input": {},
                "intent_source": "rule",
                "provider": "local",
            }
        if self._is_math_query(normalized_message):
            return {
                "tool_name": "code",
                "tool_input": {"expression": normalized_message},
                "intent_source": "rule",
                "provider": "local",
            }
        if self._is_file_query(normalized_message):
            file_paths = self._extract_constraints(normalized_message).get(
                "entities",
                {},
            ).get("file_paths", [])
            return {
                "tool_name": "file",
                "tool_input": {"file_path": file_paths[0]},
                "intent_source": "rule",
                "provider": "local",
            }
        if self._is_knowledge_query(normalized_message):
            return {
                "tool_name": "knowledge",
                "tool_input": {"query": normalized_message},
                "intent_source": "rule",
                "provider": "local",
            }
        return {
            "tool_name": "",
            "tool_input": {},
            "intent_source": "rule",
            "provider": "local",
        }

    def _answer_with_tool(self, user_message: str, tool_result: str) -> str:
        """根据工具上下文生成回复。"""
        return (
            "已完成工具调用，结果如下：\n\n"
            f"{tool_result}\n\n"
            "如需继续处理，我可以基于该结果进行总结、改写或下一步分析。"
            "如果需要进一步分析或整理这些信息，请直接告诉我。"
        )

    def _answer_directly(
        self,
        user_message: str,
        context: list[dict[str, str]],
    ) -> str:
        """在不调用工具时直接生成回复。"""
        if not user_message:
            return "请输入需要处理的问题。"
        if context:
            return (
                "我已结合当前会话上下文理解你的问题。当前本地模型未接入"
                "外部 LLM，建议配置正式模型密钥后获得更完整回答。"
                f"你的问题是：{user_message}"
            )
        return (
            "当前运行在本地规则模型模式，已接收问题："
            f"{user_message}。如涉及时间、计算、搜索或知识库，"
            "我会自动调用工具。"
        )

    def _is_time_query(self, message: str) -> bool:
        """检查消息是否为时间类问题。"""
        keywords = (
            "时间",
            "日期",
            "今天",
            "明天",
            "后天",
            "昨天",
            "现在",
            "timestamp",
        )
        return any(keyword in message for keyword in keywords)

    def _is_math_query(self, message: str) -> bool:
        """检查消息是否需要执行计算。"""
        keywords = ("计算", "算一下", "公式", "平方", "开方")
        has_operator = any(operator in message for operator in ("+", "-", "*", "/"))
        has_digit = any(character.isdigit() for character in message)
        return any(keyword in message for keyword in keywords) or (
            has_operator and has_digit
        )

    def _is_search_query(self, message: str) -> bool:
        """检查消息是否需要公网搜索。"""
        if self._extract_url(message):
            return False
        keywords = (
            "搜索",
            "联网",
            "最新",
            "新闻",
            "公开资料",
            "实时",
            "天气",
            "气温",
            "降雨",
            "空气质量",
            "预报",
            "官网",
            "网页",
            "余票",
            "高铁票",
            "火车票",
            "动车票",
            "列车票",
            "12306",
            "机票",
            "航班",
            "飞机",
            "航空",
            "酒店",
            "住宿",
            "房态",
            "房价",
            "预约",
        )
        return any(keyword in message for keyword in keywords)

    def _is_file_query(self, message: str) -> bool:
        """检查消息是否包含明确本地文件路径。"""
        file_paths = self._extract_constraints(message).get("entities", {}).get(
            "file_paths",
            [],
        )
        return bool(file_paths) and any(
            suffix in message.lower() for suffix in (".pdf", ".txt", ".docx")
        )

    def _is_knowledge_query(self, message: str) -> bool:
        """检查消息是否需要知识库检索。"""
        keywords = ("知识库", "文档", "资料", "合同", "规章", "手册", "根据")
        return any(keyword in message for keyword in keywords)

    def _is_public_info_query(self, message: str) -> bool:
        """检查消息是否关于外部公共信息（非企业内部），应走 search 而非 knowledge。"""
        keywords = ("博物馆", "文物", "展厅", "展馆", "景区", "景点", "公园", "游乐")
        return any(keyword in message for keyword in keywords)


class OpenAICompatibleModelClient(LocalModelClient):
    """OpenAI 兼容 Chat Completions 模型客户端基类。"""

    provider_name = "openai-compatible"
    _allowed_tools = {
        "",
        "search",
        "code",
        "file",
        "time",
        "knowledge",
        "browser",
    }

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name

    def identify_intent(self, message: str) -> IntentResult:
        """调用大模型完成意图识别。

        大模型必须返回结构化 JSON；解析失败、工具名非法或接口失败时回退到本地规则。
        """
        if not self._api_key:
            intent = super().identify_intent(message)
            intent["intent_source"] = "rule_no_api_key"
            intent["provider"] = self.provider_name
            return intent
        messages = self._build_intent_messages(message)
        try:
            raw_result = self._request_chat_completion(messages, temperature=0.0)
            return self._parse_intent_result(raw_result, message)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("{} 意图识别结果解析失败：{}", self.provider_name, exc)
        except httpx.HTTPError as exc:
            logger.warning("{} 意图识别请求失败：{}", self.provider_name, exc)
        intent = super().identify_intent(message)
        intent["intent_source"] = "rule_fallback"
        intent["provider"] = self.provider_name
        return intent

    def generate(
        self,
        user_message: str,
        tool_result: str | None = None,
        context: list[dict[str, str]] | None = None,
        memories: list[dict[str, Any]] | None = None,
        plan: list[PlanStep] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> str:
        """调用 OpenAI 兼容接口生成助手回答。

        参数:
            user_message: 最新用户消息。
            tool_result: 可选的工具执行结果。
            context: 最近聊天上下文。
            memories: 已检索的长期记忆片段。
            plan: 已定型的执行计划。
            tool_results: 多工具执行结果列表。
        返回:
            助手回复文本。
        异常:
            无。请求失败时返回本地兜底回答。
        """
        if not self._api_key:
            return super().generate(
                user_message=user_message,
                tool_result=tool_result,
                context=context,
                memories=memories,
                plan=plan,
                tool_results=tool_results,
            )
        if tool_results and not tool_result:
            merged_lines: list[str] = []
            for idx, r in enumerate(tool_results, start=1):
                if r.get("status") != "success":
                    continue
                content = str(r.get("content") or "").strip()
                if content:
                    merged_lines.append(f"[工具{idx}:{r.get('tool_name', 'unknown')}]\n{content}")
            tool_result = "\n\n".join(merged_lines)
        messages = self._build_messages(
            user_message=user_message,
            tool_result=tool_result,
            context=context or [],
            memories=memories or [],
            plan=plan or [],
        )
        try:
            return self._request_chat_completion(messages, temperature=0.2)
        except httpx.HTTPError as exc:
            logger.warning("{} 调用失败：{}", self.provider_name, exc)
            return super().generate(
                user_message=user_message,
                tool_result=tool_result,
                context=context,
                memories=memories,
                plan=plan,
                tool_results=tool_results,
            )

    def create_plan(
        self,
        analysis: TaskAnalysis,
        memories: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        tool_results: list[dict[str, Any]] | None = None,
    ) -> list[PlanStep]:
        """调用大模型生成 CoT/ToT 风格的结构化执行计划。"""
        if not self._api_key:
            return super().create_plan(analysis, memories, tool_specs, tool_results)
        system_prompt = (
            "你是企业智能 Agent 的规划节点。"
            "请基于任务意图、边界、记忆和工具清单拆解步骤，输出 JSON 对象。"
            "必须包含 steps 数组；每个步骤包含 id、phase、name、goal、depends_on、"
            "status，可选 tool_name、tool_input。phase 只能从 memory、planning、"
            "tools、action、reflection 中选择。只输出 JSON，不要 Markdown。\n\n"
            "重要规则：\n"
            "1. 如果当前用户消息没有明确指定目的地、时间等关键参数，必须创建澄清步骤"
            "（phase=planning, goal 包含'确认/补充'关键词）先向用户确认，"
            "不要从对话历史中臆测参数。\n"
            "2. knowledge 工具是企业内部知识库，只用于检索企业内部文档（如制度、"
            "流程、规范），不要将其用于通用模板查询或旅游规划查询。\n"
            "3. 每个工具调用必须有明确用途，不要添加冗余步骤。\n"
            "4. 如果最终输出信息完全来自 search 工具而非用户提供的内部资料，"
            "需要在回答中注明信息来源。"
        )
        payload = {
            "analysis": analysis,
            "memories": memories[:5],
            "tools": tool_specs,
        }
        if tool_results:
            payload["previous_tool_results"] = [
                {
                    "tool_name": r.get("tool_name"),
                    "status": r.get("status"),
                    "error": r.get("error_msg", ""),
                    "content_preview": str(r.get("content", ""))[:120],
                }
                for r in tool_results[-3:]
                if r.get("tool_name")
            ]
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            },
        ]
        try:
            raw_result = self._request_chat_completion(messages, temperature=0.1)
            plan = self._normalize_plan_result(raw_result, analysis, tool_specs)
            if plan:
                return plan
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("{} 规划结果解析失败：{}", self.provider_name, exc)
        except httpx.HTTPError as exc:
            logger.warning("{} 规划请求失败：{}", self.provider_name, exc)
        return super().create_plan(analysis, memories, tool_specs)

    def reflect(
        self,
        user_message: str,
        answer: str,
        plan: list[PlanStep],
        tool_results: list[dict[str, Any]],
        error_info: str = "",
    ) -> ReflectionResult:
        """调用大模型完成执行复盘。"""
        if not self._api_key:
            return super().reflect(user_message, answer, plan, tool_results, error_info)
        system_prompt = (
            "你是企业智能 Agent 的反思复盘节点。"
            "请对照用户需求、执行计划、工具结果和最终回答评估任务质量。"
            "只输出 JSON：score(0-1)、status、issues数组、improvements数组、"
            "requires_retry布尔值、retry_reason字符串、archive_worthy布尔值。"
            "如果存在关键约束缺失、错误假设、工具参数明显不匹配、无关RAG结果，"
            "requires_retry 必须为 true。"
            "如果问题是缺少实时外部专用工具能力，而不是用户约束缺失，"
            "请在 issues 中明确写“能力缺失”，不要要求用户补充已给出的出发地、目的地或日期。"
        )
        payload = {
            "user_message": user_message,
            "answer": answer,
            "plan": plan,
            "tool_results": tool_results,
            "error_info": error_info,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            },
        ]
        try:
            raw_result = self._request_chat_completion(messages, temperature=0.0)
            payload = self._loads_json_object(raw_result)
            return {
                "score": float(payload.get("score", 0.0)),
                "status": str(payload.get("status") or "unknown"),
                "issues": list(payload.get("issues") or []),
                "improvements": list(payload.get("improvements") or []),
                "requires_retry": bool(payload.get("requires_retry", False)),
                "retry_reason": str(payload.get("retry_reason") or ""),
                "archive_worthy": bool(payload.get("archive_worthy", False)),
            }
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("{} 复盘结果解析失败：{}", self.provider_name, exc)
        except httpx.HTTPError as exc:
            logger.warning("{} 复盘请求失败：{}", self.provider_name, exc)
        return super().reflect(user_message, answer, plan, tool_results, error_info)

    def extract_memory(
        self,
        user_message: str,
        answer: str,
        reflection: ReflectionResult,
    ) -> list[str]:
        """调用大模型萃取可复用长期记忆。"""
        if not self._api_key or not reflection.get("archive_worthy"):
            return super().extract_memory(user_message, answer, reflection)
        system_prompt = (
            "你是企业智能 Agent 的记忆归档节点。"
            "请只抽取对后续任务有复用价值的用户偏好、业务规则、有效经验。"
            "只输出 JSON：{\"memories\":[\"...\"]}，最多 3 条。"
        )
        payload = {
            "user_message": user_message,
            "answer": answer,
            "reflection": reflection,
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            },
        ]
        try:
            raw_result = self._request_chat_completion(messages, temperature=0.0)
            payload = self._loads_json_object(raw_result)
            memories = payload.get("memories") or []
            return [str(memory).strip() for memory in memories if str(memory).strip()][:3]
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("{} 记忆抽取结果解析失败：{}", self.provider_name, exc)
        except httpx.HTTPError as exc:
            logger.warning("{} 记忆抽取请求失败：{}", self.provider_name, exc)
        return super().extract_memory(user_message, answer, reflection)

    def _build_intent_messages(self, message: str) -> list[dict[str, str]]:
        """构建大模型意图识别消息。"""
        system_prompt = (
            "你是企业智能办公 Agent 的意图路由器，负责判断顶层意图分类和是否需要调用工具。"
            "你必须只输出一个 JSON 对象，不要输出 Markdown、解释或多余文本。"
            "intent_category 只能是：information_inquiry、task_execution、interaction_chat。"
            "分类规则："
            "information_inquiry=只获取知识、数据、资料、规则、文档内容或概念解释，不要求落地操作；"
            "task_execution=要求完成具体事务或交付成型成果，例如写代码、生成报告、整理文档、制作报表、预订、修改、统计；"
            "interaction_chat=寒暄、情绪交流、趣味闲谈，无实际业务目标。"
            "可选 tool_name 只能是：search、code、file、time、knowledge、browser、空字符串。"
            "工具说明："
            "search=查询公网、实时、最新或公开信息，例如天气、新闻、景点、博物馆、"
            "文物、展厅、旅游、交通、公开资料等一切外部信息；"
            "code=执行数学计算表达式；"
            "file=解析明确给出的本地文件路径；"
            "time=查询当前时间、日期、时间戳或日期偏移；"
            "knowledge=检索企业内部知识库文档（制度、流程、规范、合同等内部资料），"
            "禁止用于外部公共信息查询；"
            "browser=打开并读取明确 URL 的网页；"
            "其他通过 MCP 发现的远程工具会在规划阶段按真实工具名出现，"
            "意图识别阶段不要输出抽象的 mcp 作为工具名；"
            "空字符串=普通对话，不需要工具。"
            "如果选择 code，tool_input 使用 expression；"
            "如果选择 file，tool_input 使用 file_path；"
            "如果选择 time，可使用 offset_days；"
            "如果选择 knowledge，tool_input 使用 query；"
            "如果选择 browser，tool_input 使用 url。"
            "重要：不确定选哪个工具时优先选 search，不要盲目使用 knowledge。"
            "输出格式固定为："
            '{"intent_category":"information_inquiry","tool_name":"search","tool_input":{"query":"用户问题"}}。'
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

    def _parse_intent_result(self, raw_result: str, message: str) -> IntentResult:
        """解析并校验大模型意图识别结果。"""
        payload = self._loads_json_object(raw_result)
        tool_name = str(payload.get("tool_name") or "").strip().lower()
        if tool_name in {"none", "null", "chat", "llm", "answer"}:
            tool_name = ""
        if tool_name not in self._allowed_tools:
            raise ValueError(f"不支持的工具名：{tool_name}")

        tool_input = payload.get("tool_input") or {}
        if not isinstance(tool_input, dict):
            tool_input = {}
        return {
            "intent_category": self._normalize_intent_category(
                payload.get("intent_category"),
            ),
            "tool_name": tool_name,
            "tool_input": self._normalize_tool_input(tool_name, tool_input, message),
            "intent_source": "llm",
            "provider": self.provider_name,
        }

    def _normalize_plan_result(
        self,
        raw_result: str,
        analysis: TaskAnalysis,
        tool_specs: list[dict[str, Any]],
    ) -> list[PlanStep]:
        """解析并校验模型输出的执行计划。"""
        payload = self._loads_json_object(raw_result)
        raw_steps = payload.get("steps") or []
        if not isinstance(raw_steps, list):
            raise ValueError("规划输出缺少 steps 数组")
        available_tools = {str(spec.get("name")) for spec in tool_specs}
        normalized_steps: list[PlanStep] = []
        for index, raw_step in enumerate(raw_steps, start=1):
            if not isinstance(raw_step, dict):
                continue
            depends_on = raw_step.get("depends_on") or []
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            elif not isinstance(depends_on, list):
                depends_on = []
            step: PlanStep = {
                "id": str(raw_step.get("id") or f"step_{index}"),
                "phase": str(raw_step.get("phase") or "planning"),
                "name": str(raw_step.get("name") or f"步骤 {index}"),
                "goal": str(raw_step.get("goal") or ""),
                "depends_on": depends_on,
                "status": str(raw_step.get("status") or "pending"),
            }
            tool_name = str(raw_step.get("tool_name") or "").strip().lower()
            if tool_name in {"none", "null"}:
                tool_name = ""
            if tool_name:
                if tool_name not in available_tools:
                    raise ValueError(f"规划使用了未注册工具：{tool_name}")
                step["tool_name"] = tool_name
                step["tool_input"] = self._normalize_tool_input(
                    tool_name,
                    dict(raw_step.get("tool_input") or {}),
                    str(analysis.get("normalized_task") or ""),
                )
            normalized_steps.append(step)
        return normalized_steps

    def _loads_json_object(self, raw_result: str) -> dict[str, Any]:
        """从模型输出中提取 JSON 对象。"""
        cleaned_result = raw_result.strip()
        cleaned_result = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned_result,
            flags=re.IGNORECASE,
        ).strip()
        if not cleaned_result.startswith("{"):
            match = re.search(r"\{.*\}", cleaned_result, flags=re.DOTALL)
            if not match:
                raise ValueError("模型输出中未找到 JSON 对象")
            cleaned_result = match.group(0)
        payload = json.loads(cleaned_result)
        if not isinstance(payload, dict):
            raise ValueError("模型输出不是 JSON 对象")
        return payload

    def _normalize_tool_input(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        """按工具约束补齐和过滤入参。"""
        if tool_name == "search":
            return {"query": str(tool_input.get("query") or message).strip()}
        if tool_name == "code":
            return {"expression": str(tool_input.get("expression") or message).strip()}
        if tool_name == "knowledge":
            normalized_input: dict[str, Any] = {
                "query": str(tool_input.get("query") or message).strip()
            }
            if "top_k" in tool_input:
                normalized_input["top_k"] = int(tool_input["top_k"])
            if "min_score" in tool_input:
                normalized_input["min_score"] = float(tool_input["min_score"])
            return normalized_input
        if tool_name == "browser":
            url = str(tool_input.get("url") or self._extract_url(message)).strip()
            if not url:
                raise ValueError("browser 工具缺少 url")
            return {"url": url, "action": str(tool_input.get("action") or "read")}
        if tool_name == "time":
            normalized_input = {}
            if "offset_days" in tool_input:
                normalized_input["offset_days"] = int(tool_input["offset_days"])
            return normalized_input
        if tool_name == "file":
            file_path = str(tool_input.get("file_path") or "").strip()
            if not file_path:
                raise ValueError("file 工具缺少 file_path")
            return {"file_path": file_path}
        return dict(tool_input)

    def _build_messages(
        self,
        user_message: str,
        tool_result: str | None,
        context: list[dict[str, str]],
        memories: list[dict[str, Any]],
        plan: list[PlanStep],
    ) -> list[dict[str, str]]:
        """构建 OpenAI 兼容消息列表。"""
        messages = [
            {
                "role": "system",
                "content": (
                    "你是企业级智能办公 Agent 的行动执行节点。"
                    "回答必须准确、简洁、可执行；如果上下文来自工具结果，"
                    "必须优先基于工具结果作答，并显式说明无法确认的部分。"
                    "如果工具结果为空或无效，主动向用户说明并提供替代方案建议。"
                    "回答末尾如果上下文支持，给出 1 条后续可执行建议。"
                ),
            }
        ]
        messages.extend(context[-10:])
        if memories:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "以下是与本轮任务相关的长期记忆，按需使用：\n"
                        f"{json.dumps(memories, ensure_ascii=False, default=str)}"
                    ),
                }
            )
        if plan:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "本轮任务执行计划如下，回答时对齐目标但不要机械复述：\n"
                        f"{json.dumps(plan, ensure_ascii=False, default=str)}"
                    ),
                }
            )
        if tool_result:
            messages.append(
                {
                    "role": "user",
                    "content": f"以下是工具调用结果，请基于它回答：\n{tool_result}",
                }
            )
        messages.append({"role": "user", "content": user_message})
        return messages

    def _request_chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        """请求 Chat Completions 接口。"""
        langchain_result = self._request_chat_completion_with_langchain(
            messages=messages,
            temperature=temperature,
        )
        if langchain_result:
            return langchain_result
        return self._request_chat_completion_with_httpx(messages, temperature)

    def _request_chat_completion_with_langchain(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        """优先使用 LangChain ChatModel 调用 OpenAI 兼容接口。"""
        try:
            from langchain_core.messages import AIMessage
            from langchain_core.messages import HumanMessage
            from langchain_core.messages import SystemMessage
            from langchain_openai import ChatOpenAI
        except ImportError:
            return ""

        settings = get_settings()
        lc_messages = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        try:
            chat_model = ChatOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                model=self._model_name,
                temperature=temperature,
                timeout=settings.request_timeout,
            )
            result = chat_model.invoke(lc_messages)
        except Exception as exc:
            logger.warning("{} LangChain 调用失败，回退 httpx：{}", self.provider_name, exc)
            return ""
        return str(result.content)

    def _request_chat_completion_with_httpx(
        self,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        """使用 httpx 直连 Chat Completions 接口，作为 LangChain 兜底。"""
        settings = get_settings()
        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        with httpx.Client(timeout=settings.request_timeout) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]

    def _missing_key_message(self) -> str:
        """返回缺少密钥时的提示信息。"""
        return (
            f"当前选择了 {self.provider_name}，但尚未配置对应 API Key。"
            "请在项目根目录 `.env` 中补充密钥后重启服务。"
        )


class OpenAIModelClient(OpenAICompatibleModelClient):
    """OpenAI 模型客户端接入位置。"""

    provider_name = "OpenAI"


class DeepSeekModelClient(OpenAICompatibleModelClient):
    """DeepSeek 模型客户端接入位置。"""

    provider_name = "DeepSeek"


class QwenModelClient(OpenAICompatibleModelClient):
    """通义千问 OpenAI 兼容模式客户端接入位置。"""

    provider_name = "通义千问"


def get_model_client() -> LocalModelClient:
    """根据配置返回模型客户端。

    返回:
        当前模型供应商对应的模型客户端。
    异常:
        无。不支持的供应商会回退到本地模型。
    """
    settings = get_settings()
    provider = settings.model_provider.lower()
    if provider == "openai":
        return OpenAIModelClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model_name=settings.openai_model,
        )
    if provider == "deepseek":
        return DeepSeekModelClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model_name=settings.deepseek_model,
        )
    if provider in {"qwen", "dashscope", "tongyi"}:
        return QwenModelClient(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            model_name=settings.qwen_model,
        )
    return LocalModelClient()
