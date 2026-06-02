"""Soft planner: VLM-driven planning without hard SOP state tracking.

The planner analyzes the current state against available skills and decides:
- EXECUTE: Move forward (SOP on-track or autonomous)
- REFLECT: SOP is correct but previous action failed, re-examine SOP
- DONE: Task complete
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from anything2skill.agent.state import AgentState
from anything2skill.parser.data_types import Skills
from anything2skill.vlm.client import VLMClient

if TYPE_CHECKING:
    from anything2skill.agent.message_builder import MessageBuilder
    from anything2skill.benchmark_kit import BenchmarkKit

logger = logging.getLogger("anything2skill.agent.planner")


class PlanAction(Enum):
    """Terminal action the agent loop should take."""

    EXECUTE = "execute"
    REFLECT = "reflect"
    DONE = "done"
    FAIL = "fail"


@dataclass
class PlanDecision:
    """Outcome of a planning step."""

    action: PlanAction
    reasoning: str = ""
    guidance: str = ""
    raw_response: str = ""


class SoftPlanner:
    """VLM-powered planner that evaluates state against skills.

    No step index tracking. The VLM judges:
    - Whether current state aligns with a skill's SOP
    - Whether the previous action failed (needs reflection)
    - Whether the task is complete
    """

    def __init__(self, vlm: VLMClient, kit: BenchmarkKit, llm_params: dict | None = None):
        self.vlm = vlm
        self.kit = kit
        self.llm_params = llm_params or {}
        # Lazy-init to avoid circular imports at module level
        self._msg: MessageBuilder | None = None

    @property
    def msg(self) -> MessageBuilder:
        if self._msg is None:
            from anything2skill.agent.message_builder import MessageBuilder
            self._msg = MessageBuilder(self.kit)
        return self._msg

    def plan(
        self,
        skills: Skills,
        state: AgentState,
        obs_content: list[dict],
        instruction: str,
    ) -> PlanDecision:
        msgs = self.msg.build_planner_messages(
            skills, state, obs_content, instruction,
        )
        resp = self.vlm.chat(msgs, **self.llm_params)
        decision = self._parse_response(resp)
        decision.raw_response = resp
        return decision

    def _parse_response(self, resp: str) -> PlanDecision:
        parsed = _extract_json(resp)

        action_str = parsed.get("action", "execute")
        try:
            action = PlanAction(action_str)
        except ValueError:
            logger.warning("Unknown plan action '%s', defaulting to EXECUTE", action_str)
            action = PlanAction.EXECUTE

        return PlanDecision(
            action=action,
            reasoning=parsed.get("reasoning", ""),
            guidance=parsed.get("guidance", ""),
        )


def _extract_json(text: str) -> dict:
    """Extract JSON from VLM response."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to extract JSON from planner response: %s", text[:200])
    return {}
