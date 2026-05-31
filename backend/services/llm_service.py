"""模型抽象层。"""

import hashlib
import json
import re
from typing import Any

import httpx
from loguru import logger

from config.settings import get_settings
from memory.store import redis_kv

_LLM_CACHE_TTL = 300  # LLM 结果缓存 5 分钟


IntentResult = dict[str, Any]
TaskAnalysis = dict[str, Any]
PlanStep = dict[str, Any]

# ------------------------------------------------------------------
# 动态向量样本：从工具注册表按描述+参数字段自动生成
# ------------------------------------------------------------------

_VECTOR_DIM = 1024

# 根据工具 input_schema 字段名自动补充的领域样本模板
_FIELD_EXEMPLAR_PATTERNS: dict[str, list[str]] = {
    "city": ["查询城市天气", "天气怎么样", "明天会下雨吗", "温度湿度"],
    "query": ["搜索一下", "帮我查", "查询资料", "查找信息"],
    "expression": ["计算", "算一下", "等于多少", "数学计算", "公式"],
    "to": ["发邮件给", "发送邮件到", "邮件通知", "写邮件"],
    "file_path": ["读取文件", "打开文件", "查看文档", "文件内容"],
    "url": ["打开网页", "访问网站", "浏览页面", "读取网址"],
    "offset_days": ["现在几点", "今天日期", "当前时间", "星期几"],
}


def _text_hash(text: str) -> str:
    """简单的字符串哈希，用于向量维度索引映射。"""
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _cjk_tokens(chars: list[str]) -> list[str]:
    """CJK 字符 unigram + bigram。"""
    if not chars:
        return []
    result = list(chars)
    for i in range(len(chars) - 1):
        result.append(chars[i] + chars[i + 1])
    return result


def _tokenize(text: str) -> list[str]:
    """中英文混合分词：英文按词，中文按 unigram+bigram。"""
    tokens: list[str] = []
    current_word = ""
    cjk_chars: list[str] = []
    for char in text:
        if char.isascii() and char.isalnum():
            if cjk_chars:
                tokens.extend(_cjk_tokens(cjk_chars))
                cjk_chars = []
            current_word += char
        elif char.strip():
            if current_word:
                tokens.append(current_word)
                current_word = ""
            cjk_chars.append(char)
        else:
            if current_word:
                tokens.append(current_word)
                current_word = ""
            if cjk_chars:
                tokens.extend(_cjk_tokens(cjk_chars))
                cjk_chars = []
    if current_word:
        tokens.append(current_word)
    if cjk_chars:
        tokens.extend(_cjk_tokens(cjk_chars))
    return tokens


def _embed_text(text: str) -> list[float]:
    """将文本转为确定性归一化向量（基于哈希的轻量 embedding）。"""
    import math
    vector = [0.0] * _VECTOR_DIM
    normalized_text = text.lower()
    if not normalized_text:
        return vector
    for token in _tokenize(normalized_text):
        index = int(_text_hash(token), 16) % _VECTOR_DIM
        vector[index] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    return max(0.0, min(1.0, dot))  # 已归一化向量，余弦 = 内积


# 延迟加载的动态向量样本缓存（从工具注册表生成）
_EXEMPLAR_EMBEDDINGS_CACHE: dict[str, list[list[float]]] | None = None


def _build_exemplar_embeddings() -> dict[str, list[list[float]]]:
    """从工具注册表动态构建向量样本，支持 MCP 动态工具。"""
    embeddings: dict[str, list[list[float]]] = {}
    try:
        from services.tool_service import get_tool_registry

        for spec in get_tool_registry().list_specs():
            label = spec.name
            texts = [spec.description, spec.name]
            # 根据 input_schema 字段名补充领域样本
            for field, patterns in _FIELD_EXEMPLAR_PATTERNS.items():
                if field in spec.input_schema:
                    texts.extend(patterns)
            embeddings[label] = [_embed_text(t) for t in set(texts)]
    except Exception:
        pass
    return embeddings


def _get_exemplar_embeddings() -> dict[str, list[list[float]]]:
    """获取向量样本（延迟加载 + 缓存）。"""
    global _EXEMPLAR_EMBEDDINGS_CACHE
    if _EXEMPLAR_EMBEDDINGS_CACHE is None:
        _EXEMPLAR_EMBEDDINGS_CACHE = _build_exemplar_embeddings()
    return _EXEMPLAR_EMBEDDINGS_CACHE



class LocalModelClient:
    """本地模型兜底实现。"""

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
        memories: list[dict[str, Any]] | None = None,
        memory_ctx: Any = None,
    ) -> TaskAnalysis:
        """分层混合意图识别：Layer 1 关键词 → Layer 2 向量 → Layer 3 LLM。

        memory_ctx 承载短期记忆（对话历史）和长期记忆（向量检索），
        三层架构全部使用该上下文增强意图理解。
        """
        context = context or []
        normalized_message = self._normalize_message(message)
        constraints = self._extract_constraints(normalized_message)

        # Layer 1: 关键词+正则前置粗筛（支持上下文感知的弱匹配降级）
        layer1_result = self._layer1_fast_match(normalized_message, memory_ctx=memory_ctx)
        if layer1_result is not None:
            intent = layer1_result
            logger.info(
                "[意图识别] Layer1 命中 | tool={} source={}",
                intent.get("tool_name") or "(无)",
                intent.get("intent_source"),
            )
        else:
            # Layer 2: 向量 Embedding 分类（支持上下文增强查询）
            best_label, best_score = self._layer2_vector_classify(
                normalized_message, memory_ctx=memory_ctx,
            )
            L2_USE_THRESHOLD = 0.80   # 高分直接使用
            L2_HINT_THRESHOLD = 0.50   # 中等分数作为 LLM 提示

            if best_score >= L2_USE_THRESHOLD and best_label:
                intent = self._intent_from_label(best_label, normalized_message)
                intent["intent_source"] = "layer2_vector"
                intent["l2_score"] = best_score
                logger.info(
                    "[意图识别] Layer1 未命中 → Layer2 高分直用 | "
                    "label={} score={:.4f} tool={}",
                    best_label, best_score,
                    intent.get("tool_name") or "(无)",
                )
            elif best_score >= L2_HINT_THRESHOLD and best_label:
                # Layer 3: LLM 精判（带向量提示 + 长期记忆 + 对话上下文）
                intent = self.identify_intent(
                    normalized_message, memories=memories, memory_ctx=memory_ctx,
                )
                intent["intent_source"] = "layer3_llm"
                provider = intent.get("provider", "")
                intent["provider"] = f"{provider}|vec_hint={best_label}({best_score:.3f})"
                logger.info(
                    "[意图识别] Layer1 未命中 → Layer2 中等分({:.4f}) → Layer3 LLM(带提示) | "
                    "vec_label={} llm_tool={}",
                    best_score, best_label,
                    intent.get("tool_name") or "(无)",
                )
            else:
                # Layer 3: LLM 精判（无向量提示，向量分太低不可靠）
                intent = self.identify_intent(
                    normalized_message, memories=memories, memory_ctx=memory_ctx,
                )
                intent["intent_source"] = "layer3_llm"
                logger.info(
                    "[意图识别] Layer1 未命中 → Layer2 低分({:.4f}) → Layer3 LLM(无提示) | "
                    "llm_tool={}",
                    best_score if best_label else 0.0,
                    intent.get("tool_name") or "(无)",
                )

        # 后处理：公共信息查询不应走 knowledge，闲聊兜底避免误触发工具
        if intent.get("tool_name") == "knowledge" and self._is_public_info_query(
            normalized_message,
        ):
            logger.info("[意图识别] 后处理: knowledge → 空（公共信息查询）")
            intent["tool_name"] = ""
            intent["tool_input"] = {}
        if self._is_interaction_chat(normalized_message):
            intent["tool_name"] = ""
            intent["tool_input"] = {}

        confidence = self._intent_confidence(intent)
        logger.info(
            "[意图识别] 最终结果 | tool={} confidence={:.2f} source={}",
            intent.get("tool_name") or "(无)",
            confidence,
            intent.get("intent_source", "?"),
        )

        return {
            "normalized_task": normalized_message,
            "tool_name": intent.get("tool_name", ""),
            "tool_input": intent.get("tool_input", {}),
            "intent_source": intent.get("intent_source", ""),
            "constraints": constraints,
            "entities": constraints.get("entities", {}),
            "confidence": confidence,
            "provider": intent.get("provider", "local"),
        }

    def _layer2_vector_classify(
        self,
        message: str,
        memory_ctx: Any = None,
    ) -> tuple[str | None, float]:
        """Layer 2: 向量相似度分类 — 动态样本集，余弦得分即为置信度。

        支持上下文增强查询：当消息过短或检测到多轮跟进时，
        将对话摘要拼入 query 文本以获得更准确的向量匹配。
        """
        if not message.strip():
            return None, 0.0

        # 上下文增强：短消息或跟进语 → 拼入对话摘要提升匹配精度
        is_short = len(message) < 5
        is_follow_up = bool(getattr(memory_ctx, "is_follow_up", False)) if memory_ctx else False
        if is_short or is_follow_up:
            context_text = ""
            if memory_ctx and hasattr(memory_ctx, "to_text_summary"):
                context_text = memory_ctx.to_text_summary()
            if context_text:
                augmented = f"{context_text}\n当前: {message}"
                query_vec = _embed_text(augmented)
            else:
                query_vec = _embed_text(message)
        else:
            query_vec = _embed_text(message)

        best_label: str | None = None
        best_score = 0.0
        for label, exemplar_vecs in _get_exemplar_embeddings().items():
            for ev in exemplar_vecs:
                score = _cosine_similarity(query_vec, ev)
                if score > best_score:
                    best_score = score
                    best_label = label
        return best_label, round(best_score, 4)

    def _intent_from_label(self, label: str, message: str) -> IntentResult:
        """将向量分类的 label 转换为完整 IntentResult（label 即工具名）。"""
        tool_name = label

        # 动态从工具注册表获取入参键
        input_keys: list[str] = []
        try:
            from services.tool_service import get_tool_registry

            registry = get_tool_registry()
            tool = registry.get(tool_name)
            input_keys = list(tool.input_schema.keys())
        except Exception:
            pass  # 注册表不可用时使用空入参

        tool_input: dict[str, Any] = {}
        if "city" in input_keys:
            tool_input["city"] = self._extract_city(message)
        if "query" in input_keys:
            tool_input["query"] = message
        if "expression" in input_keys:
            tool_input["expression"] = message
        if "to" in input_keys:
            tool_input["to"] = ""
        if "subject" in input_keys:
            tool_input["subject"] = message
        if "body" in input_keys:
            tool_input["body"] = message
        if "file_path" in input_keys:
            file_paths = self._extract_constraints(message).get(
                "entities",
                {},
            ).get("file_paths", [])
            tool_input["file_path"] = file_paths[0] if file_paths else message
        if "url" in input_keys:
            tool_input["url"] = self._extract_url(message) or message
        return {
            "intent_category": "task_execution",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "intent_source": "layer2_vector",
            "provider": "local",
        }

    def _layer1_fast_match(self, message: str, memory_ctx: Any = None) -> IntentResult | None:
        """Layer 1: 关键词 + 正则前置粗筛（极速拦截 30%~50% 强特征请求）。

        支持上下文感知的弱匹配降级：
          - 裸邮箱（无"发邮件"等动词）+ 对话跟进 → 返回 None，穿透给 L2/L3
          - 明确动作关键词 → 直接命中
        """
        # 闲聊 / 能力问询 → 不需要工具（始终不需要上下文）
        if self._is_capability_question(message):
            return {
                "intent_category": "information_inquiry",
                "tool_name": "",
                "tool_input": {},
                "intent_source": "layer1_keyword",
                "provider": "local",
            }
        if self._is_interaction_chat(message):
            return {
                "intent_category": "interaction_chat",
                "tool_name": "",
                "tool_input": {},
                "intent_source": "layer1_keyword",
                "provider": "local",
            }

        # 复合任务检测：消息同时提到多个工具关键词 → 不拦截，交给高层
        if self._is_compound_task(message):
            return None

        # ── 上下文感知的弱匹配降级 ──────────────────────────────────
        # 裸邮箱检测：有邮箱正则但无动词 → 弱匹配
        is_bare_email = False
        if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", message):
            email_verbs = ("发邮件", "发送邮件", "写邮件", "发送给", "发送到")
            if not any(v in message for v in email_verbs):
                is_bare_email = True

        # 裸城市名检测：只有一个城市名但无天气关键词 → 弱匹配
        is_bare_city = False
        if not is_bare_email:
            weather_keywords = ("天气", "气温", "降雨", "湿度", "风力", "预报", "温度", "空气质量", "降雪")
            if not any(kw in message for kw in weather_keywords):
                # 检查是否有任意城市关键词但没有天气动词
                from tools.weather_tool import CITY_COORDS  # type: ignore
                for city_name in CITY_COORDS:
                    if city_name in message:
                        is_bare_city = True
                        break

        # 如果检测到弱匹配，检查 memory_ctx 中是否有跟进上下文
        if (is_bare_email or is_bare_city) and memory_ctx is not None:
            is_follow_up = getattr(memory_ctx, "is_follow_up", False)
            last_tool = getattr(memory_ctx, "last_tool_used", "")
            if is_follow_up and last_tool:
                logger.info(
                    "[意图识别] Layer1 弱匹配降级 | bare={} last_tool={} | 穿透至 L2/L3",
                    "email" if is_bare_email else "city",
                    last_tool,
                )
                return None

        # 强特征工具请求 — 直接命中，不走后续层
        if self._is_weather_query(message):
            return {
                "intent_category": "task_execution",
                "tool_name": "weather",
                "tool_input": {"city": self._extract_city(message)},
                "intent_source": "layer1_keyword",
                "provider": "local",
            }
        if self._is_time_query(message):
            return {
                "intent_category": "task_execution",
                "tool_name": "time",
                "tool_input": {},
                "intent_source": "layer1_keyword",
                "provider": "local",
            }
        if self._is_math_query(message):
            return {
                "intent_category": "task_execution",
                "tool_name": "code",
                "tool_input": {"expression": message},
                "intent_source": "layer1_keyword",
                "provider": "local",
            }
        if self._is_email_query(message):
            return {
                "intent_category": "task_execution",
                "tool_name": "email",
                "tool_input": {"to": "", "subject": message, "body": message},
                "intent_source": "layer1_keyword",
                "provider": "local",
            }
        if self._is_file_query(message):
            file_paths = self._extract_constraints(message).get(
                "entities", {},
            ).get("file_paths", [])
            return {
                "intent_category": "task_execution",
                "tool_name": "file" if file_paths else "",
                "tool_input": {"file_path": file_paths[0]} if file_paths else {},
                "intent_source": "layer1_keyword",
                "provider": "local",
            }
        return None

    def create_plan(
        self,
        analysis: TaskAnalysis,
        memories: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]],
        tool_results: list[dict[str, Any]] | None = None,
        memory_ctx: Any = None,
    ) -> list[PlanStep]:
        """根据意图、记忆和工具能力生成可调度计划。"""
        if tool_results is not None:
            pass  # 签名兼容，LLM 子类中使用
        tool_name = str(analysis.get("tool_name") or "")
        tool_input = dict(analysis.get("tool_input") or {})
        available_tools = {str(spec.get("name")) for spec in tool_specs}
        if tool_name and tool_name not in available_tools:
            tool_name = ""
            tool_input = {}

        # 跨轮次参数推断：当前工具缺少必要参数时，从 memory_ctx 补全
        if memory_ctx is not None and tool_name:
            result_preview = getattr(memory_ctx, "last_tool_result_preview", "")
            if result_preview:
                # email body 为空，但上一轮有工具执行结果 → 注入
                if tool_name == "email":
                    body = tool_input.get("body", "")
                    if not body or len(str(body).strip()) < 10:
                        tool_input = dict(tool_input)
                        tool_input["body"] = str(result_preview)[:500]
                # 其他工具也可同理扩展

        plan: list[PlanStep] = [
            {
                "id": "understand",
                "phase": "memory",
                "name": "任务理解",
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
            ]
        )
        return plan

    def identify_intent(self, message: str, memories: list[dict[str, Any]] | None = None, memory_ctx: Any = None) -> IntentResult:
        """兜底意图识别（无 LLM 时返回空意图，由三层架构处理）。"""
        _ = memories
        _ = memory_ctx  # 本地模式不使用记忆，LLM 子类接管时使用
        return {
            "tool_name": "",
            "tool_input": {},
            "intent_source": "rule_default",
            "provider": "local",
        }

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

    def _intent_confidence(self, intent: IntentResult) -> float:
        """为结构化意图提供稳定的置信度（按识别层级递减）。
        L1/L3 使用固定分，L2 使用余弦相似度实际得分。
        """
        source = intent.get("intent_source", "")
        if source == "layer1_keyword":
            return 0.95
        if source == "layer2_vector":
            return float(intent.get("l2_score", 0.88))
        if source == "layer3_llm":
            return 0.82
        if source == "rule_default":
            return 0.85
        return 0.74 if intent.get("tool_name") else 0.66

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
        return False

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

    def _is_browser_query(self, message: str) -> bool:
        """检查消息是否需要网页浏览工具。"""
        return bool(self._extract_url(message)) or any(
            keyword in message for keyword in ("打开网页", "访问网页", "浏览网页", "读取网页")
        )

    def _extract_url(self, message: str) -> str:
        """从消息中提取第一个 URL。"""
        match = re.search(r"https?://[^\s，。；,;）)]+", message)
        return match.group(0) if match else ""

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
                "我已结合之前对话理解你的问题。当前本地模型未接入"
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

    def _is_weather_query(self, message: str) -> bool:
        """检查消息是否需要查询天气。"""
        keywords = ("天气", "气温", "降雨", "降雪", "湿度", "风力", "空气质量", "预报", "温度")
        return any(keyword in message for keyword in keywords)

    def _is_compound_task(self, message: str) -> bool:
        """检测消息是否涉及多个工具关键词（复合任务），用于 Layer 1 跳过拦截。

        当前支持的复合检测组合：
          - weather + email（查天气并发邮件）
          - weather + knowledge / + file / + browser 等
        """
        tool_groups = [
            {"weather", "气温", "降雨", "天气", "温度", "湿度", "风力", "预报"},
            {"email", "发邮件", "发送邮件", "写邮件", "邮件", "发送给", "发送到"},
            {"time", "时间", "日期", "现在", "几点"},
            {"knowledge", "知识库", "文档", "合同", "规章", "手册"},
            {"browser", "打开网页", "访问", "浏览"},
            {"file", "文件", "读取", "打开文件", "pdf", ".txt"},
            {"code", "计算", "公式", "平方"},
        ]
        matched_groups = 0
        for group in tool_groups:
            if any(kw in message for kw in group):
                matched_groups += 1
                if matched_groups >= 2:
                    return True
        return False

    def _extract_city(self, message: str) -> str:
        """从天气查询消息中用正则提取中文城市名。

        优先匹配常见地名词典，再通过位置特征提取。
        """
        import re
        # 常见中国城市名（地名词典）
        known_cities = sorted((
            "北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安",
            "南京", "重庆", "天津", "苏州", "长沙", "郑州", "东莞", "青岛",
            "沈阳", "宁波", "昆明", "大连", "厦门", "合肥", "佛山", "福州",
            "哈尔滨", "济南", "温州", "长春", "石家庄", "常州", "泉州",
            "南宁", "贵阳", "南昌", "太原", "烟台", "嘉兴", "南通", "金华",
            "珠海", "惠州", "徐州", "海口", "乌鲁木齐", "绍兴", "中山",
            "台州", "兰州", "三亚", "呼和浩特", "银川", "西宁", "拉萨",
            "洛阳", "襄阳", "宜昌", "芜湖", "赣州", "遵义", "安庆",
            "保定", "邯郸", "秦皇岛", "张家口", "大同", "开封", "南阳",
            "柳州", "桂林", "珠海", "湛江", "汕头", "三亚",
        ), key=len, reverse=True)

        # 优先匹配已知城市名
        for city in known_cities:
            if city in message:
                return city

        # 兜底：捕获 "XXX的天气" / "XXX天气" / "XXX气温" 中的城市名
        m = re.search(r"([一-鿿]{2,4})(?:的?天气|的?气温|的?温度|的?降雨|的?预报)", message)
        if m:
            return m.group(1)

        return message

    def _is_email_query(self, message: str) -> bool:
        """检查消息是否需要发送邮件。"""
        keywords = ("发邮件", "发送邮件", "写邮件", "发一封邮件", "邮件通知", "发送给", "发送到")
        if any(keyword in message for keyword in keywords):
            return True
        # 检测邮件地址模式
        if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", message):
            return True
        return False

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
        "weather",
        "code",
        "file",
        "time",
        "knowledge",
        "browser",
        "email",
    }

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        fallback_client: LocalModelClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._fallback_client = fallback_client

    def _fallback_identify_intent(self, message: str, memory_ctx: Any = None) -> IntentResult:
        """尝试 fallback 客户端识别意图，失败则走本地规则。"""
        if self._fallback_client is not None:
            try:
                intent = self._fallback_client.identify_intent(message, memory_ctx=memory_ctx)
                intent["intent_source"] = "fallback_model"
                intent["provider"] = getattr(self._fallback_client, "provider_name", "fallback")
                return intent
            except Exception as exc:
                logger.warning("fallback 意图识别也失败：{}", exc)
        intent = super().identify_intent(message, memory_ctx=memory_ctx)
        intent["intent_source"] = "rule_fallback"
        intent["provider"] = self.provider_name
        return intent

    def _fallback_generate(
        self,
        user_message: str,
        tool_result: str | None = None,
        context: list[dict[str, str]] | None = None,
        memories: list[dict[str, Any]] | None = None,
        plan: list[PlanStep] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> str:
        """尝试 fallback 客户端生成回答，失败则走本地规则。"""
        if self._fallback_client is not None:
            try:
                return self._fallback_client.generate(
                    user_message=user_message,
                    tool_result=tool_result,
                    context=context,
                    memories=memories,
                    plan=plan,
                    tool_results=tool_results,
                )
            except Exception as exc:
                logger.warning("fallback 生成也失败：{}", exc)
        return super().generate(
            user_message=user_message,
            tool_result=tool_result,
            context=context,
            memories=memories,
            plan=plan,
            tool_results=tool_results,
        )

    # ------------------------------------------------------------------
    # LLM 结果缓存（RedisKV）
    # ------------------------------------------------------------------

    @staticmethod
    def _build_intent_cache_key(
        message: str,
        memories: list[dict[str, Any]] | None = None,
        memory_ctx: Any = None,
    ) -> str:
        """构建意图识别缓存键。"""
        import hashlib
        parts = [message]
        if memories:
            mem_preview = "|".join(
                str(m.get("text", ""))[:60] for m in memories[:3]
            )
            parts.append(hashlib.md5(mem_preview.encode()).hexdigest()[:12])
        if memory_ctx is not None:
            parts.append(getattr(memory_ctx, "last_user_msg", ""))
            parts.append(getattr(memory_ctx, "last_tool_used", ""))
        raw = "||".join(parts)
        return "llm:intent:" + hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_generate_cache_key(
        user_message: str,
        tool_result: str | None,
        context: list[dict[str, str]] | None,
    ) -> str:
        """构建生成回答缓存键。"""
        import hashlib
        parts = [user_message]
        if tool_result:
            parts.append(tool_result[:120])
        if context:
            ctx_snap = "|".join(
                m.get("content", "")[:60] for m in context[-3:]
            )
            parts.append(hashlib.md5(ctx_snap.encode()).hexdigest()[:12])
        raw = "||".join(parts)
        return "llm:generate:" + hashlib.md5(raw.encode("utf-8")).hexdigest()

    def identify_intent(self, message: str, memories: list[dict[str, Any]] | None = None, memory_ctx: Any = None) -> IntentResult:
        """调用大模型完成意图识别（带 Redis 缓存）。

        大模型必须返回结构化 JSON；解析失败、工具名非法或接口失败时回退到本地规则。
        """
        if not self._api_key:
            return self._fallback_identify_intent(message, memory_ctx=memory_ctx)

        # ── Redis 缓存查询 ──────────────────────────────────────────
        cache_key = self._build_intent_cache_key(message, memories, memory_ctx)
        cached = redis_kv.get(cache_key)
        if cached is not None:
            logger.debug("[LLM缓存] {} identify_intent 命中", self.provider_name)
            return cached  # type: ignore[return-value]

        messages = self._build_intent_messages(message, memories=memories, memory_ctx=memory_ctx)
        try:
            raw_result = self._request_chat_completion(messages, temperature=0.0)
            result = self._parse_intent_result(raw_result, message)
            redis_kv.set(cache_key, result, ttl=_LLM_CACHE_TTL)
            return result
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("{} 意图识别结果解析失败：{}", self.provider_name, exc)
        except httpx.HTTPError as exc:
            logger.warning("{} 意图识别请求失败：{}", self.provider_name, exc)
        return self._fallback_identify_intent(message)

    def generate(
        self,
        user_message: str,
        tool_result: str | None = None,
        context: list[dict[str, str]] | None = None,
        memories: list[dict[str, Any]] | None = None,
        plan: list[PlanStep] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> str:
        """调用 OpenAI 兼容接口生成助手回答（带 Redis 缓存）。

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
            return self._fallback_generate(
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

        # ── Redis 缓存查询 ──────────────────────────────────────────
        cache_key = self._build_generate_cache_key(user_message, tool_result, context)
        cached = redis_kv.get(cache_key)
        if cached is not None:
            logger.debug("[LLM缓存] {} generate 命中", self.provider_name)
            return str(cached)

        messages = self._build_messages(
            user_message=user_message,
            tool_result=tool_result,
            context=context or [],
            memories=memories or [],
            plan=plan or [],
        )
        try:
            result = self._request_chat_completion(messages, temperature=0.2)
            redis_kv.set(cache_key, result, ttl=_LLM_CACHE_TTL)
            return result
        except httpx.HTTPError as exc:
            logger.warning("{} 调用失败：{}", self.provider_name, exc)
            return self._fallback_generate(
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
        memory_ctx: Any = None,
    ) -> list[PlanStep]:
        """调用大模型生成 CoT（思维链）/ToT（树状思维）风格的结构化执行计划。"""
        if not self._api_key:
            return super().create_plan(
                analysis, memories, tool_specs, tool_results, memory_ctx=memory_ctx,
            )
        system_prompt = (
            "你是智能 Agent 的规划节点。"
            "请基于任务意图、记忆和工具清单拆解步骤，输出 JSON 对象。"
            "必须包含 steps 数组；每个步骤包含 id、phase、name、goal、depends_on、"
            "status，可选 tool_name、tool_input。phase 只能从 memory、planning、"
            "tools、action 中选择。只输出 JSON，不要 Markdown。\n\n"
            "重要规则：\n"
            "1. 如果当前用户消息没有明确指定目的地、时间等关键参数，必须创建澄清步骤"
            "（phase=planning, goal 包含'确认/补充'关键词）先向用户确认，"
            "如果短期记忆里面有明确信息可以使用，没有的禁止臆测 / 编造参数。\n"
            "2. knowledge 工具是外部知识库，只用于检索知识库，不要将其用于通用模板查询或旅游规划等查询。\n"
            "3. 每个工具调用必须有明确用途，不要添加冗余步骤。\n"
            "4. 如果最终输出信息完全来自外部工具而非用户提供的内部资料，"
            "需要在回答中注明信息来源。\n"
            "5. 如果某步骤的 tool_input 依赖前序工具的执行结果（如邮件正文需要填入天气查询结果），"
            "禁止使用「step_1 的结果」等占位符，应将对应参数留空或填写简短的描述性文字。"
            "系统会在所有工具执行完毕后，将完整结果汇总给生成节点。"
        )
        payload = {
            "analysis": analysis,
            "memories": memories[:5],
            "tools": tool_specs,
        }
        if memory_ctx is not None and hasattr(memory_ctx, "to_plan_context"):
            plan_ctx = memory_ctx.to_plan_context()
            if plan_ctx:
                payload["memory_context"] = plan_ctx
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
        return super().create_plan(
            analysis, memories, tool_specs, tool_results, memory_ctx=memory_ctx,
        )

    def _build_intent_messages(
        self,
        message: str,
        memories: list[dict[str, Any]] | None = None,
        memory_ctx: Any = None,
    ) -> list[dict[str, str]]:
        """构建大模型意图识别消息。"""
        system_prompt = (
            "你是 Agent 的意图路由器，负责判断是否需要调用工具。"
            "你必须只输出一个 JSON 对象，不要输出 Markdown、解释或多余文本。"
            "可选 tool_name 只能是：weather、email、code、file、time、knowledge、browser、空字符串。"
            "工具说明："
            "weather=查询指定城市的实时天气和天气预报；"
            "email=发送邮件，需要指定收件人(to)、主题(subject)和正文(body)；"
            "code=执行数学计算表达式；"
            "file=解析明确给出的本地文件路径；"
            "time=查询当前时间、日期、时间戳或日期偏移；"
            "knowledge=检索企业内部知识库文档（制度、流程、规范、合同等内部资料），"
            "禁止用于外部公共信息查询；"
            "browser=打开并读取明确 URL 的网页；"
            "其他通过 MCP 发现的远程工具会在规划阶段按真实工具名出现，"
            "意图识别阶段不要输出抽象的 mcp 作为工具名；"
            "空字符串=普通对话，不需要工具。"
            "如果选择 weather，tool_input 使用 city；"
            "如果选择 email，tool_input 使用 to、subject、body，可选 cc；"
            "如果选择 code，tool_input 使用 expression；"
            "如果选择 file，tool_input 使用 file_path；"
            "如果选择 time，可使用 offset_days；"
            "如果选择 knowledge，tool_input 使用 query；"
            "如果选择 browser，tool_input 使用 url。"
            "重要：不确定选哪个工具时优先选 weather，不要盲目使用 knowledge。"
            "如果用户发送的是零散信息（如只有邮箱地址），请结合【对话历史】判断真实意图。"
            "输出格式固定为："
            '{"tool_name":"weather","tool_input":{"city":"城市名称"}}。'
        )
        messages = [{"role": "system", "content": system_prompt}]
        # 注入记忆上下文（对话历史 + 长期记忆）
        if memory_ctx is not None:
            ctx_text = ""
            if hasattr(memory_ctx, "to_llm_context"):
                ctx_text = memory_ctx.to_llm_context()
            if ctx_text:
                messages.append({
                    "role": "system",
                    "content": ctx_text,
                })
        if memories:
            messages.append({
                "role": "system",
                "content": (
                    "以下是该用户历史交互中的长期记忆，"
                    "可参考用户的历史行为模式来判断当前意图：\n"
                    f"{json.dumps(memories, ensure_ascii=False, default=str)}"
                ),
            })
        messages.append({"role": "user", "content": message})
        return messages

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
        if tool_name == "weather":
            return {"city": str(tool_input.get("city") or message).strip()}
        if tool_name == "email":
            return {
                "to": str(tool_input.get("to", "")).strip(),
                "subject": str(tool_input.get("subject", "")).strip(),
                "body": str(tool_input.get("body") or message).strip(),
                "cc": str(tool_input.get("cc", "")).strip(),
            }
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
                    "你是智能助手。"
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



def _build_fallback_client(settings: Any) -> LocalModelClient | None:
    """构建兜底模型客户端，当前使用本地 DeepSeek（Ollama）。"""
    if settings.deepseek_api_key and settings.deepseek_base_url and settings.deepseek_model:
        return DeepSeekModelClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model_name=settings.deepseek_model,
        )
    return None


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
            fallback_client=_build_fallback_client(settings),
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
            fallback_client=_build_fallback_client(settings),
        )
    return LocalModelClient()
