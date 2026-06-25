"""Agent contract normalization tests."""

from schemas.agent_contract import filter_task_steps
from schemas.agent_contract import normalize_depends_on
from schemas.agent_contract import safe_dict


def test_normalize_depends_on_keeps_string_as_one_dependency() -> None:
    assert normalize_depends_on("plan") == ["plan"]
    assert normalize_depends_on(["weather", 123, ""]) == ["weather", "123"]
    assert normalize_depends_on({"bad": "shape"}) == []


def test_filter_task_steps_removes_internal_steps_and_dependencies() -> None:
    plan = [
        {"id": "understand", "kind": "understand", "status": "completed"},
        {"id": "plan", "kind": "plan", "status": "completed"},
        {
            "id": "weather",
            "kind": "tool",
            "depends_on": ["understand", "plan"],
            "status": "pending",
        },
        {
            "id": "respond",
            "kind": "respond",
            "depends_on": "weather",
            "status": "pending",
        },
    ]

    task_steps = filter_task_steps(plan)

    assert [step["id"] for step in task_steps] == ["weather", "respond"]
    assert task_steps[0]["depends_on"] == []
    assert task_steps[1]["depends_on"] == ["weather"]


def test_safe_dict_rejects_non_dict_shapes() -> None:
    assert safe_dict({"ok": True}) == {"ok": True}
    assert safe_dict(["bad"]) == {}
