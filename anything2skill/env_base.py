"""Abstract environment interface for benchmark compatibility."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anything2skill.benchmark_kit import TaskDescriptor


class EnvironmentInterface(ABC):
    """Abstract base for all benchmark environments.

    Future environments (WebArena, AndroidWorld, etc.) should implement this.
    """

    @abstractmethod
    def reset(self, task: TaskDescriptor) -> dict:
        """Reset environment for a new task.

        Args:
            task: TaskDescriptor (or subclass) containing task data.
                  Each benchmark's env wrapper extracts what it needs
                  from the subclass fields.

        Returns:
            Initial observation dict.
        """

    @abstractmethod
    def step(self, action: str, pause: float = 2.0) -> tuple[dict, float, bool, dict]:
        """Execute an action in the environment.

        Args:
            action: Action string (domain-specific code or framework signal).
                Framework-level signals that all envs MUST handle:
                - "DONE": agent considers the task complete. Set done=True.
                - "FAIL": agent considers the task infeasible. Set done=True.
                Other signals (e.g. "WAIT") are domain-specific.
            pause: Pause after execution in seconds.

        Returns:
            Tuple of (observation, reward, done, info).
        """

    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate task completion.

        Returns:
            Score (typically 0.0 or 1.0).
        """

    @abstractmethod
    def close(self):
        """Clean up environment resources."""

    @property
    def vm_ip(self) -> str:
        """VM IP address (if applicable)."""
        return ""
