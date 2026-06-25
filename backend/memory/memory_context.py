"""Two-layer memory context used by the Agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class MemoryContext:
    """Read-only memory view with short-term and long-term layers."""

    short_term: list[dict[str, str]] = field(default_factory=list)
    """Recent raw messages in the current session window."""

    short_term_summary: str = ""
    """Compacted summary for messages dropped from the sliding window."""

    long_term: list[dict[str, Any]] = field(default_factory=list)
    """Retrieved cross-session facts, preferences, and durable knowledge."""

    recent_observations: list[dict[str, Any]] = field(default_factory=list)
    """Recent structured tool results reusable by follow-up turns."""

    @staticmethod
    def build(
        messages: list[dict[str, str]],
        relevant_memories: list[dict[str, Any]],
        short_term_summary: str = "",
        recent_observations: list[dict[str, Any]] | None = None,
    ) -> MemoryContext:
        """Build a two-layer memory context from runtime state."""
        messages = list(messages or [])
        relevant_memories = list(relevant_memories or [])
        return MemoryContext(
            short_term=messages[-10:],
            short_term_summary=str(short_term_summary or ""),
            long_term=relevant_memories,
            recent_observations=list(recent_observations or [])[-10:],
        )

    @property
    def conversation(self) -> list[dict[str, str]]:
        """Backward-compatible alias for recent short-term messages."""
        return self.short_term

    @property
    def last_user_msg(self) -> str:
        """Last user message in the short-term window."""
        for msg in reversed(self.short_term):
            if msg.get("role") == "user":
                return str(msg.get("content") or "")
        return ""

    def to_llm_context(self) -> str:
        """Render memory as prompt context."""
        parts: list[str] = []
        if self.short_term_summary:
            parts.append(f"【短期记忆摘要】\n{self.short_term_summary[:1200]}")
        if self.short_term:
            lines = []
            for msg in self.short_term[-8:]:
                role = "用户" if msg.get("role") == "user" else "助理"
                lines.append(f"{role}: {str(msg.get('content') or '')[:300]}")
            parts.append("【短期记忆窗口】\n" + "\n".join(lines))
        if self.long_term:
            mem_lines = []
            for m in self.long_term:
                score = m.get("score", 0)
                text = str(m.get("text", ""))[:300]
                mem_lines.append(f"- [{score:.4f}] {text}")
            if mem_lines:
                parts.append("【长期记忆】\n" + "\n".join(mem_lines))
        if self.recent_observations:
            observation_lines = []
            for observation in self.recent_observations[-5:]:
                if observation.get("status") != "success":
                    continue
                tool_name = str(observation.get("tool_name") or "unknown")
                tool_input = observation.get("tool_input") or {}
                content = str(observation.get("content") or "")[:500]
                observation_lines.append(
                    f"- [{tool_name}] input={tool_input} result={content}"
                )
            if observation_lines:
                parts.append("Recent tool results:\n" + "\n".join(observation_lines))
        return "\n\n".join(parts)

    def to_text_summary(self) -> str:
        """Render a compact text summary for non-planning consumers."""
        return self.to_llm_context()

    def to_plan_context(self) -> str:
        """Planning consumes the same two-layer memory view."""
        return self.to_llm_context()
