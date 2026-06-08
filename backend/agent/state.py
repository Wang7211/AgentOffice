"""Agent runtime state definitions."""

from typing import Any
from typing import TypedDict


class AgentState(TypedDict):
    """Shared state passed between Agent graph nodes."""

    messages: list[dict[str, str]]
    task_desc: str
    normalized_task: str
    intent: dict[str, Any]
    need_tool: bool
    relevant_memories: list[dict[str, Any]]
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: str
    plan: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    step_count: int
    max_steps: int
    error_info: str
    answer: str
    session_id: str
    user_id: int
    memory_context: Any
