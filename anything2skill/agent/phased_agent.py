"""PhasedAgent: Plan -> Act/Reflect two-phase agent.

Uses a SoftPlanner to evaluate state against skills, then dispatches:
- EXECUTE: generate action (planner-guided, no skills context)
- REFLECT: re-examine skills to diagnose and correct errors
- DONE: task complete
"""

from __future__ import annotations

import logging

from anything2skill.agent.base import BaseAgent
from anything2skill.agent.planner import PlanAction, PlanDecision, SoftPlanner
from anything2skill.benchmark_kit import BenchmarkKit
from anything2skill.parser.data_types import Skills
from anything2skill.vlm.client import VLMClient

logger = logging.getLogger("anything2skill.agent")


class PhasedAgent(BaseAgent):
    """Two-phase agent: Plan -> Act/Reflect.

    The planner (SoftPlanner) evaluates the current state against available
    skills and decides EXECUTE/REFLECT/DONE. No hard SOP step tracking.
    """

    def __init__(
        self,
        vlm: VLMClient,
        skills: Skills,
        kit: BenchmarkKit,
        planner: SoftPlanner | None = None,
        history_window: int = 3,
        result_dir: str | None = None,
        llm_params: dict | None = None,
    ):
        super().__init__(vlm, skills, kit, history_window, result_dir, llm_params)
        self.planner = planner or SoftPlanner(vlm, kit, llm_params=self.llm_params)

    def predict(self, instruction: str, obs: dict) -> tuple[str, list[str]]:
        # 1. Observe
        obs_content = self.kit.encode_observation(obs)

        # 2. Plan
        decision = self.planner.plan(
            self.skills, self.state, obs_content, instruction,
        )

        # 3. Dispatch
        if decision.action == PlanAction.DONE:
            logger.info("Planner decided DONE: %s", decision.reasoning)
            self._update_predict_info("done", decision)
            return "DONE", ["DONE"]

        if decision.action == PlanAction.FAIL:
            logger.info("Planner decided FAIL: %s", decision.reasoning)
            self._update_predict_info("fail", decision)
            return "FAIL", ["FAIL"]

        if decision.action == PlanAction.REFLECT:
            return self._do_reflect(obs_content, instruction, decision)

        # PlanAction.EXECUTE
        return self._do_execute(obs_content, instruction, decision)

    def _do_execute(
        self,
        obs_content: list[dict],
        instruction: str,
        decision: PlanDecision,
    ) -> tuple[str, list[str]]:
        messages = self.msg.build_executor_messages(obs_content, instruction, decision)
        response = self.vlm.chat(messages, **self.llm_params)
        actions = self.kit.parse_actions(response) or ["WAIT"]
        logger.info("Execute: %s", actions[0])

        self.state.record(obs_content, response, actions[0])
        self._update_predict_info("execute", decision)

        return response, actions

    def _do_reflect(
        self,
        obs_content: list[dict],
        instruction: str,
        decision: PlanDecision,
    ) -> tuple[str, list[str]]:
        history = self.state.get_recent_history(self.history_window)
        messages = self.msg.build_reflect_messages(
            self.skills, obs_content, history, instruction, decision,
        )
        response = self.vlm.chat(messages, **self.llm_params)
        actions = self.kit.parse_actions(response) or ["WAIT"]
        logger.info("Reflect: %s", actions[0])

        self.state.record(obs_content, response, actions[0])
        self._update_predict_info("reflect", decision)

        return response, actions

    def _update_predict_info(self, phase: str, decision: PlanDecision):
        self._predict_count += 1
        self._last_predict_info = {
            "predict_num": self._predict_count,
            "phase": phase,
            "total_history": len(self.state.history),
            "planner": {
                "reasoning": decision.reasoning,
                "guidance": decision.guidance,
                "raw_response": decision.raw_response,
            },
        }
