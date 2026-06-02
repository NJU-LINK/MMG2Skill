"""BaseAgent: abstract interface for all agent modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from anything2skill.agent.message_builder import MessageBuilder
from anything2skill.agent.state import AgentState

if TYPE_CHECKING:
    from anything2skill.benchmark_kit import BenchmarkKit
    from anything2skill.parser.data_types import Skills
    from anything2skill.vlm.client import VLMClient


class BaseAgent(ABC):
    """Abstract base class for agent implementations."""

    def __init__(
        self,
        vlm: VLMClient,
        skills: Skills,
        kit: BenchmarkKit,
        history_window: int = 3,
        result_dir: str | None = None,
        llm_params: dict | None = None,
    ):
        self.vlm = vlm
        self.skills = skills
        self.kit = kit
        self.msg = MessageBuilder(kit)
        self.history_window = history_window
        self.result_dir = result_dir
        self.llm_params = llm_params or {}
        self.state = AgentState()
        self._predict_count: int = 0
        self._last_predict_info: dict = {}

    @abstractmethod
    def predict(self, instruction: str, obs: dict) -> tuple[str, list[str]]:
        """Generate the next action(s) given an observation.

        Args:
            instruction: The task instruction string.
            obs: Observation dict from the environment.

        Returns:
            Tuple of (response_text, list_of_action_strings).
        """

    @property
    def last_predict_info(self) -> dict:
        """Metadata from the most recent predict() call.

        Populated by subclasses; merged into traj.jsonl by the runner.
        """
        return self._last_predict_info
