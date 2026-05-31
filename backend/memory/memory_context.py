"""统一记忆上下文。

在 mem_pre 节点中构建，封装短期记忆（对话历史）和长期记忆（向量检索），
为下游所有节点（意图理解、规划、工具执行、回答生成）提供"先回忆，后理解"的基础。
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

# ---------------------------------------------------------------------------
# 助理消息中用于推断工具使用的关键词
# ---------------------------------------------------------------------------
_TOOL_SIGNATURES: dict[str, list[str]] = {
    "weather": ["天气", "°C", "湿度", "风力", "晴", "雨", "雪", "预报"],
    "email": ["邮件已发送", "已发送邮件", "邮件发送成功"],
    "code": [
        "计算结果",
        "计算",
        "=",
    ],
    "time": ["北京时间", "当前时间", "日期", "今天是"],
    "knowledge": ["知识库", "文档内容", "根据文档", "规章制度"],
    "browser": ["网页内容", "页面内容", "网页访问"],
}


def _infer_last_tool(assistant_msg: str) -> str:
    """从助理回复内容推断上一轮使用的工具名称。"""
    for tool_name, keywords in _TOOL_SIGNATURES.items():
        if any(kw in assistant_msg for kw in keywords):
            return tool_name
    return ""


def _extract_last_exchange(
    messages: list[dict[str, str]],
) -> tuple[str, str, str, list[str]]:
    """提取最近一轮对话的完整信息。

    Returns:
        (last_user_msg, last_assistant_msg, last_tool_used, past_tool_chain)
    """
    last_user = ""
    last_assistant = ""
    tool_chain: list[str] = []

    # 从后往前遍历，提取最后一轮 user+assistant 交换
    for msg in reversed(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant" and not last_assistant:
            last_assistant = content
            inferred = _infer_last_tool(content)
            if inferred:
                tool_chain.insert(0, inferred)
        elif role == "user" and not last_user:
            last_user = content
        if last_user and last_assistant:
            break

    # 从整个历史中构建工具链
    for msg in messages:
        if msg.get("role") == "assistant":
            inferred = _infer_last_tool(msg.get("content", ""))
            if inferred and (not tool_chain or tool_chain[-1] != inferred):
                tool_chain.append(inferred)

    return last_user, last_assistant, tool_chain[0] if tool_chain else "", tool_chain


@dataclass
class MemoryContext:
    """统一记忆上下文。

    由 mem_pre_node 每轮从 AgentState 构建一次，供理解、规划、工具执行、回答节点只读消费。
    """

    # ── 原始数据 ──────────────────────────────────────────────────
    conversation: list[dict[str, str]] = field(default_factory=list)
    """最近 N 轮对话历史（state.messages）。"""

    long_term: list[dict[str, Any]] = field(default_factory=list)
    """向量检索到的长期记忆（state.relevant_memories）。"""

    # ── 派生字段（由 build() 自动计算） ────────────────────────────
    last_user_msg: str = ""
    """上一轮用户消息文本。"""

    last_assistant_msg: str = ""
    """上一轮助理回复文本。"""

    last_tool_used: str = ""
    """上一轮使用的工具名，如 'weather'；空串表示未使用工具。"""

    last_tool_succeeded: bool = False
    """上一轮工具是否执行成功。"""

    last_tool_result_preview: str = ""
    """上一轮工具结果摘要（前 200 字符）。"""

    is_follow_up: bool = False
    """本轮是否是多轮对话中的跟进（即 history 中有完整的一问一答）。"""

    past_tool_chain: list[str] = field(default_factory=list)
    """本轮会话中使用过的工具列表（按使用顺序）。"""

    # ── 工厂方法 ──────────────────────────────────────────────────

    @staticmethod
    def build(
        messages: list[dict[str, str]],
        relevant_memories: list[dict[str, Any]],
    ) -> MemoryContext:
        """从 AgentState 构建统一记忆上下文。

        Args:
            messages: state["messages"]，对话历史。
            relevant_memories: state["relevant_memories"]，向量检索到的长期记忆。

        Returns:
            填充完整的 MemoryContext 实例。
        """
        messages = list(messages or [])
        relevant_memories = list(relevant_memories or [])

        last_user, last_assistant, last_tool, tool_chain = _extract_last_exchange(messages)

        is_follow_up = False
        user_count = sum(1 for m in messages if m.get("role") == "user")
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        if user_count >= 1 and assistant_count >= 1:
            is_follow_up = True

        # 从助理回复中提取工具结果预览
        result_preview = ""
        if last_assistant:
            result_preview = last_assistant[:200]

        return MemoryContext(
            conversation=messages[-10:],
            long_term=relevant_memories,
            last_user_msg=last_user,
            last_assistant_msg=last_assistant,
            last_tool_used=last_tool,
            last_tool_succeeded=bool(last_tool),
            last_tool_result_preview=result_preview,
            is_follow_up=is_follow_up,
            past_tool_chain=tool_chain,
        )

    # ── 视图方法 ──────────────────────────────────────────────────

    def to_text_summary(self) -> str:
        """紧凑的纯文本摘要，供 L1 弱匹配判定和 L2 查询增强使用。"""
        parts: list[str] = []
        if self.conversation:
            lines = []
            for msg in self.conversation[-4:]:
                role = "用户" if msg.get("role") == "user" else "助理"
                content = msg.get("content", "")
                lines.append(f"{role}: {content[:150]}")
            parts.append("【对话历史】\n" + "\n".join(lines))
        if self.long_term:
            mem_lines = []
            for m in self.long_term[:3]:
                score = m.get("score", 0)
                text = str(m.get("text", ""))[:100]
                mem_lines.append(f"- [{score}] {text}")
            if mem_lines:
                parts.append("【长期记忆】\n" + "\n".join(mem_lines))
        return "\n\n".join(parts)

    def to_llm_context(self) -> str:
        """结构化上下文文本，供 L3 LLM prompt 注入使用。"""
        parts: list[str] = []
        # 对话历史
        if self.last_user_msg:
            history_lines = [f"用户: {self.last_user_msg}"]
            if self.last_assistant_msg:
                history_lines.append(f"助理: {self.last_assistant_msg[:300]}")
            if self.is_follow_up and len(self.conversation) > 2:
                # 附加上一轮之前的上下文
                prev_lines = []
                for msg in self.conversation[:-2]:
                    role = "用户" if msg.get("role") == "user" else "助理"
                    prev_lines.append(f"{role}: {msg.get('content', '')[:200]}")
                if prev_lines:
                    history_lines = prev_lines + history_lines
            parts.append("【对话历史】\n" + "\n".join(history_lines))
        # 长期记忆
        if self.long_term:
            mem_lines = []
            for m in self.long_term:
                score = m.get("score", 0)
                text = str(m.get("text", ""))[:200]
                mem_lines.append(f"- [{score:.4f}] {text}")
            if mem_lines:
                parts.append("【长期记忆】\n" + "\n".join(mem_lines))
        # 上一轮工具信息
        if self.last_tool_used:
            tool_info = f"上一轮调用了 {self.last_tool_used} 工具"
            if self.last_tool_succeeded:
                tool_info += "（成功）"
            else:
                tool_info += "（失败）"
            if self.last_tool_result_preview:
                tool_info += f"\n结果摘要: {self.last_tool_result_preview[:200]}"
            parts.append(f"【上一轮工具】\n{tool_info}")
        return "\n\n".join(parts)

    def to_plan_context(self) -> str:
        """供 planning 节点使用的上下文，侧重已执行的操作。"""
        parts: list[str] = []
        if self.last_tool_used:
            parts.append(
                f"上一轮已执行工具: {self.last_tool_used}"
                f"（{'成功' if self.last_tool_succeeded else '失败'}）"
            )
            if self.last_tool_result_preview:
                parts.append(f"结果预览: {self.last_tool_result_preview[:200]}")
        if self.past_tool_chain:
            parts.append(f"本轮会话工具链: {' → '.join(self.past_tool_chain)}")
        if not parts:
            parts.append("无历史工具调用记录")
        return "\n".join(parts)
