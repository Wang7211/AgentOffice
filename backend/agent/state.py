"""Agent 状态定义。"""

from typing import Any
from typing import TypedDict


class AgentState(TypedDict):
    """Agent 图节点之间共享的运行状态。"""

    messages: list[dict[str, str]]
    task_desc: str
    normalized_task: str
    intent: dict[str, Any]
    constraints: dict[str, Any]
    task_status: str
    boundary: dict[str, Any]
    clarification_question: str
    relevant_memories: list[dict[str, Any]]
    plan: list[dict[str, Any]]
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: str
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    step_count: int
    error_info: str
    answer: str
    reflection: dict[str, Any]
    reflection_retry_count: int
    archived_memory_ids: list[str]
    session_id: str
