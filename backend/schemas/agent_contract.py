"""Shared contracts for data passed between agent planning nodes."""

from __future__ import annotations

from typing import Any


PlanStep = dict[str, Any]

INTERNAL_PLAN_IDS = {"understand", "plan"}
INTERNAL_PLAN_KINDS = {"understand", "plan"}


def safe_dict(value: Any) -> dict[str, Any]:
    """Return a dictionary only when the input already has dictionary shape."""
    return dict(value) if isinstance(value, dict) else {}


def normalize_depends_on(value: Any) -> list[str]:
    """Normalize dependency declarations without splitting strings into chars."""
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def filter_task_steps(plan: list[dict[str, Any]]) -> list[PlanStep]:
    """Remove graph-internal planning steps from executable task plans."""
    removed_ids = set(INTERNAL_PLAN_IDS)
    for step in plan:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "")
        kind = str(step.get("kind") or "")
        if step_id in INTERNAL_PLAN_IDS or kind in INTERNAL_PLAN_KINDS:
            removed_ids.add(step_id)

    task_steps: list[PlanStep] = []
    for step in plan:
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("id") or "")
        kind = str(step.get("kind") or "")
        if step_id in removed_ids or kind in INTERNAL_PLAN_KINDS:
            continue
        item = dict(step)
        item["depends_on"] = [
            dep
            for dep in normalize_depends_on(item.get("depends_on"))
            if dep not in removed_ids
        ]
        task_steps.append(item)
    return task_steps
