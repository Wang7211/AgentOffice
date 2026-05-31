"""Agent 状态定义。"""

from typing import Any
from typing import TypedDict


class AgentState(TypedDict):
    """Agent 图节点之间共享的运行状态。"""

    messages: list[dict[str, str]]  # 对话历史（短期记忆）
    task_desc: str  # 原始用户输入
    normalized_task: str  # 规范化任务描述
    intent: dict[str, Any]  # 意图分析结果
    need_tool: bool  # planning 决策：是否需要调用工具
    relevant_memories: list[dict[str, Any]]  # 检索到的长期记忆
    tool_name: str  # 工具名
    tool_input: dict[str, Any]  # 工具入参
    tool_result: str  # 工具执行结果文本
    tool_calls: list[dict[str, Any]]  # 工具调用记录（供日志/持久化使用）
    tool_results: list[dict[str, Any]]  # 工具执行结果列表
    step_count: int  # 步数计数
    error_info: str  # 错误信息
    answer: str  # 最终回答
    session_id: str  # 会话ID
    memory_context: Any  # MemoryContext 对象（运行时构建，见 memory.memory_context）
