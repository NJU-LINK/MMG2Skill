"""VanillaTutorialAgent: raw-tutorial baseline (no skill extraction).

Ablation variant of VanillaAgent: the original tutorial body and images
are injected directly into the first user turn instead of going through
VLM-based skill extraction. Used to measure the contribution of the
extraction step itself.
"""

from __future__ import annotations

import logging

from anything2skill.agent.base import BaseAgent
from anything2skill.benchmark_kit import BenchmarkKit
from anything2skill.parser.data_types import Skills, TutorialMaterial
from anything2skill.vlm.client import VLMClient

logger = logging.getLogger("anything2skill.agent")


class VanillaTutorialAgent(BaseAgent):
    """Baseline agent fed raw tutorial material instead of extracted skills."""

    def __init__(
        self,
        vlm: VLMClient,
        skills: Skills,
        kit: BenchmarkKit,
        tutorial: TutorialMaterial | None,
        max_images: int | None = None,
        history_window: int = 3,
        result_dir: str | None = None,
        llm_params: dict | None = None,
    ):
        super().__init__(vlm, skills, kit, history_window, result_dir, llm_params)
        self.tutorial = tutorial
        self.max_images = max_images

    def predict(self, instruction: str, obs: dict) -> tuple[str, list[str]]:
        obs_content = self.kit.encode_observation(obs)

        history = self.state.get_recent_history(self.history_window)
        messages = self.msg.build_vanilla_tutorial_messages(
            self.tutorial,
            self.max_images,
            obs_content,
            history,
            instruction,
        )

        response = self.vlm.chat(messages, **self.llm_params)

        actions = self.kit.parse_actions(response) or ["WAIT"]
        logger.info("Action: %s", actions[0])

        self.state.record(obs_content, response, actions[0])
        self._predict_count += 1
        self._last_predict_info = {
            "predict_num": self._predict_count,
            "phase": "act",
            "total_history": len(self.state.history),
        }

        return response, actions
