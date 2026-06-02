"""VanillaAgent: baseline agent without skills.

Single VLM call per predict(): observe -> build prompt (no skills) -> VLM -> parse.
Equivalent to OSWorld's PromptAgent baseline. Uses only the domain system prompt,
task instruction, observation, and conversation history.
"""

from __future__ import annotations

import logging

from anything2skill.agent.base import BaseAgent
from anything2skill.benchmark_kit import BenchmarkKit
from anything2skill.parser.data_types import Skills
from anything2skill.vlm.client import VLMClient

logger = logging.getLogger("anything2skill.agent")


class VanillaAgent(BaseAgent):
    """Baseline agent that operates without skills or tutorials.

    All domain-specific behavior is delegated to the BenchmarkKit.
    No skills, no planner, no step tracking -- the VLM decides everything
    based solely on the task instruction and current observation.
    """

    def __init__(
        self,
        vlm: VLMClient,
        skills: Skills,
        kit: BenchmarkKit,
        history_window: int = 3,
        result_dir: str | None = None,
        llm_params: dict | None = None,
    ):
        super().__init__(vlm, skills, kit, history_window, result_dir, llm_params)

    def predict(self, instruction: str, obs: dict) -> tuple[str, list[str]]:
        # 1. Observe
        obs_content = self.kit.encode_observation(obs)

        # 2. Build messages (no skills)
        history = self.state.get_recent_history(self.history_window)
        messages = self.msg.build_vanilla_messages(
            obs_content, history, instruction,
        )

        # 3. VLM call
        response = self.vlm.chat(messages, **self.llm_params)

        # 4. Parse
        actions = self.kit.parse_actions(response) or ["WAIT"]
        logger.info("Action: %s", actions[0])

        # 5. Record
        self.state.record(obs_content, response, actions[0])
        self._predict_count += 1
        self._last_predict_info = {
            "predict_num": self._predict_count,
            "phase": "act",
            "total_history": len(self.state.history),
        }

        return response, actions
