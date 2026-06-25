"""Agent runtime state definitions."""

from typing import Any
from typing import TypedDict


class AgentState(TypedDict):
    """Shared state passed between Agent graph nodes."""

    messages: list[dict[str, str]]
    task_desc: str
    normalized_task: str
    understanding: dict[str, Any]
    capability_context: dict[str, Any]
    task_contract: dict[str, Any]
    task_evaluation: dict[str, Any]
    short_term_summary: str
    relevant_memories: list[dict[str, Any]]
    plan: list[dict[str, Any]]
    current_step_id: str
    tool_calls: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    recent_observations: list[dict[str, Any]]
    resolved_references: list[dict[str, Any]]
    replan_requested: bool
    replan_context: dict[str, Any]
    replan_count: int
    max_replans: int
    step_count: int
    max_steps: int
    error_info: str
    answer: str
    session_id: str
    user_id: int
    memory_context: Any
