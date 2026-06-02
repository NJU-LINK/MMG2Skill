"""Agent state tracking across predict calls."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentState:
    """Tracks the agent's trajectory history."""

    history: list[dict] = field(default_factory=list)

    def record(self, obs_content: list[dict], response: str, action: str):
        """Record a step in the history.

        Args:
            obs_content: Encoded observation (VLM content blocks from BenchmarkKit).
            response: LLM response text.
            action: Parsed action string.
        """
        self.history.append({
            "obs_content": obs_content,
            "response": response,
            "action": action,
        })

    def get_recent_history(self, max_length: int = 3) -> list[dict]:
        """Get recent history entries for trajectory context."""
        if not self.history:
            return []
        return self.history[-max_length:]
