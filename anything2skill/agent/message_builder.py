"""MessageBuilder: framework-managed message assembly.

Assembles VLM messages by combining the BenchmarkKit's domain-specific prompts
with framework-level templates.  All message orchestration, skills formatting,
planner prompt construction, and skill-extraction prompt building lives here
so that BenchmarkKit implementers never touch message structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anything2skill.agent.prompts import (
    EXECUTOR_SYSTEM_TMPL,
    PLANNER_SYSTEM_TMPL,
    REFLECTOR_SYSTEM_TMPL,
    SIMPLE_ACTION_SYSTEM_TMPL,
    SKILL_EXTRACTION_SYSTEM_TMPL,
    VANILLA_ACTION_SYSTEM_TMPL,
    VANILLA_TUTORIAL_ACTION_SYSTEM_TMPL,
)
from anything2skill.agent.skill_utils import format_skills_for_prompt
from anything2skill.parser.data_types import Skills, TutorialMaterial

if TYPE_CHECKING:
    from anything2skill.agent.planner import PlanDecision
    from anything2skill.agent.state import AgentState
    from anything2skill.benchmark_kit import BenchmarkKit


class MessageBuilder:
    """Assemble VLM messages from BenchmarkKit prompts + framework templates.

    Kit developers provide domain text via ``system_prompt``,
    ``bridging_text``, etc.  This class handles the rest: multi-turn
    history embedding, skills formatting, planner prompt construction,
    and skill-extraction prompt composition.
    """

    def __init__(self, kit: BenchmarkKit) -> None:
        self.kit = kit

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_action_messages(
        self,
        skills: Skills,
        obs_content: list[dict],
        history: list[dict],
        instruction: str,
    ) -> list[dict]:
        """Build messages for SimpleAgent (multi-turn, skills in first user turn).

        First user turn: skills (text + images) + instruction + bridging + obs.
        Subsequent user turns: bridging + obs.
        """
        system_text = SIMPLE_ACTION_SYSTEM_TMPL.strip().format(
            domain_system_prompt=self.kit.system_prompt,
        )
        first_content = self._build_skills_block(skills)
        first_content.append(self._task_with_bridging(instruction))
        return self._build_multiturn(system_text, first_content, history, obs_content)

    def build_vanilla_messages(
        self,
        obs_content: list[dict],
        history: list[dict],
        instruction: str,
    ) -> list[dict]:
        """Build messages for VanillaAgent (no skills, OSWorld-style).

        Instruction is placed in the first user turn.  All user turns
        (history + current) share the same format: bridging_text + obs.
        """
        system_text = VANILLA_ACTION_SYSTEM_TMPL.strip().format(
            domain_system_prompt=self.kit.system_prompt,
        )
        first_content = [self._task_with_bridging(instruction)]
        return self._build_multiturn(system_text, first_content, history, obs_content)

    def build_vanilla_tutorial_messages(
        self,
        tutorial: TutorialMaterial | None,
        max_images: int | None,
        obs_content: list[dict],
        history: list[dict],
        instruction: str,
    ) -> list[dict]:
        """Build messages for VanillaTutorialAgent (raw tutorial, no skills).

        First user turn: ``## Reference Tutorial`` block (body + capped images)
        + instruction + bridging + obs.  Subsequent user turns: bridging + obs.
        """
        system_text = VANILLA_TUTORIAL_ACTION_SYSTEM_TMPL.strip().format(
            domain_system_prompt=self.kit.system_prompt,
        )
        first_content = self._build_tutorial_block(tutorial, max_images)
        first_content.append(self._task_with_bridging(instruction))
        return self._build_multiturn(system_text, first_content, history, obs_content)

    def build_executor_messages(
        self,
        obs_content: list[dict],
        instruction: str,
        decision: PlanDecision,
    ) -> list[dict]:
        """Build messages for PhasedAgent EXECUTE (no skills, no history).

        Single user turn: instruction + planner reasoning/guidance + bridging + current obs.
        """
        system_text = EXECUTOR_SYSTEM_TMPL.strip().format(
            domain_system_prompt=self.kit.system_prompt,
        )

        user_content: list[dict] = [self._task_heading(instruction)]
        user_content.append({
            "type": "text",
            "text": (
                f"## Planner Decision\n"
                f"Reasoning: {decision.reasoning}\n"
                f"Guidance: {decision.guidance}\n\n"
                f"{self.kit.bridging_text}"
            ),
        })
        user_content.extend(obs_content)

        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ]

    def build_reflect_messages(
        self,
        skills: Skills,
        obs_content: list[dict],
        history: list[dict],
        instruction: str,
        decision: PlanDecision,
    ) -> list[dict]:
        """Build messages for PhasedAgent REFLECT (all skills + history, single turn).

        Single user turn: skills + instruction + planner guidance + history steps + current obs.
        """
        system_text = REFLECTOR_SYSTEM_TMPL.strip().format(
            domain_system_prompt=self.kit.system_prompt,
            domain_reflection_guidance=self.kit.reflection_guidance,
        )

        user_content = self._build_skills_block(skills)
        user_content.append(self._task_heading(instruction))

        user_content.append({
            "type": "text",
            "text": (
                f"## Planner Decision\n"
                f"Reasoning: {decision.reasoning}\n"
                f"Guidance: {decision.guidance}"
            ),
        })

        user_content.extend(self._build_history_steps(history, obs_content))

        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ]

    def build_planner_messages(
        self,
        skills: Skills,
        state: AgentState,
        obs_content: list[dict],
        instruction: str,
    ) -> list[dict]:
        """Build messages for the soft planner (single user turn).

        Single user turn: skills + instruction + history steps + current obs.
        """
        system_text = PLANNER_SYSTEM_TMPL.strip().format(
            domain_planner_guidance=self.kit.planner_guidance,
        )

        user_content = self._build_skills_block(skills)
        user_content.append(self._task_heading(instruction))

        history = state.get_recent_history()
        user_content.extend(self._build_history_steps(history, obs_content))

        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ]

    def build_skill_extraction_messages(
        self,
        tutorial_content: str,
        instruction: str,
        image_entries: list[tuple[str, str]],
        content_type: str = "html",
    ) -> list[dict]:
        """Build messages for skill extraction from tutorials.

        ``content_type`` selects how the tutorial body is announced to the
        VLM:

        - ``"html"``: keep the legacy ``TASK: ...\\nTUTORIAL:\\n{body}``
          framing so the model reads structured prose.
        - ``"screenshot"``: ``tutorial_content`` is empty; render only
          ``TASK: ...\\nTutorial provided as screenshots below:`` to avoid
          a dangling ``TUTORIAL:`` label with nothing under it.

        Other values raise ``NotImplementedError`` so the video modality
        cannot silently fall through.
        """
        system_text = SKILL_EXTRACTION_SYSTEM_TMPL.strip().format(
            domain_guidance=self.kit.skill_extraction_guidance,
        )

        if content_type == "html":
            intro = f"TASK: {instruction}\n\nTUTORIAL:\n{tutorial_content}"
        elif content_type == "screenshot":
            intro = (
                f"TASK: {instruction}\n\n"
                "Tutorial provided as screenshots below:"
            )
        else:
            raise NotImplementedError(
                f"build_skill_extraction_messages: unsupported "
                f"content_type={content_type!r}",
            )

        user_content: list[dict] = [{"type": "text", "text": intro}]

        if image_entries:
            user_content.append(
                {
                    "type": "text",
                    "text": f"\nThe tutorial includes {len(image_entries)} images shown below:",
                }
            )
            for filename, data_url in image_entries:
                user_content.append(
                    {"type": "text", "text": f"[Image: {filename}]"}
                )
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    }
                )

        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _task_heading(self, instruction: str) -> dict:
        """Build '# Your Task' text block (without bridging text)."""
        return {"type": "text", "text": f"## Your Task\n{instruction}"}

    def _task_with_bridging(self, instruction: str) -> dict:
        """Build '# Your Task' text block followed by bridging text."""
        return {"type": "text", "text": f"## Your Task\n{instruction}\n\n{self.kit.bridging_text}"}

    def _build_skills_block(self, skills: Skills) -> list[dict]:
        """Build content blocks: skills text with inline images."""
        formatted = format_skills_for_prompt(skills)
        if not formatted:
            return []

        content: list[dict] = [
            {"type": "text", "text": "## Reference Skills"},
        ]
        content.extend(formatted)
        return content

    def _build_tutorial_block(
        self,
        tutorial: TutorialMaterial | None,
        max_images: int | None,
    ) -> list[dict]:
        """Build content blocks: raw tutorial body + capped tutorial images.

        Used by VanillaTutorialAgent (the no-skill-extraction ablation).
        ``max_images=None`` means no cap. Empty tutorial -> empty list.

        Dispatch by ``tutorial.content_type``:

        - ``"html"``: render ``## Reference Tutorial`` heading + body text +
          images.
        - ``"screenshot"``: skip the body block (it is empty) and render
          ``## Reference Tutorial (screenshots)`` + images only.
        - other: ``NotImplementedError`` (video reserved).
        """
        if tutorial is None:
            return []

        from anything2skill.vlm.client import encode_image_files

        image_entries = encode_image_files(tutorial.image_paths, max_images)

        if tutorial.content_type == "html":
            blocks: list[dict] = [
                {"type": "text", "text": "## Reference Tutorial"},
                {"type": "text", "text": tutorial.body},
            ]
        elif tutorial.content_type == "screenshot":
            blocks = [
                {"type": "text", "text": "## Reference Tutorial (screenshots)"},
            ]
        else:
            raise NotImplementedError(
                f"_build_tutorial_block: unsupported "
                f"content_type={tutorial.content_type!r}",
            )

        if image_entries:
            blocks.append({
                "type": "text",
                "text": f"\nTutorial images ({len(image_entries)}):",
            })
            for filename, data_url in image_entries:
                blocks.append({"type": "text", "text": f"[Image: {filename}]"})
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": "high"},
                })
        return blocks

    def _build_multiturn(
        self,
        system_text: str,
        first_user_content: list[dict],
        history: list[dict],
        obs_content: list[dict],
    ) -> list[dict]:
        """Build multi-turn messages (SimpleAgent / VanillaAgent).

        First user turn uses *first_user_content* + obs.
        Subsequent user turns use bridging_text + obs.
        """
        messages: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
        ]

        all_obs: list[tuple[list[dict], str | None]] = [
            (e["obs_content"], e["response"]) for e in history
        ]
        all_obs.append((obs_content, None))

        for i, (obs, response) in enumerate(all_obs):
            if i == 0:
                user_content = list(first_user_content)
            else:
                user_content = [{"type": "text", "text": self.kit.bridging_text}]
            user_content.extend(obs)
            messages.append({"role": "user", "content": user_content})

            if response is not None:
                messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": response.strip() if response else "No valid action",
                        }
                    ],
                })

        return messages

    def _build_history_steps(
        self, history: list[dict], current_obs: list[dict],
    ) -> list[dict]:
        """Build step-aggregated history + current observation content blocks.

        Each history step: ### Step N / Action: ... / Observation: [screenshot].
        Ends with ## Current Observation + current obs.
        """
        content: list[dict] = []

        if history:
            content.append({"type": "text", "text": "## Recent Steps"})
            for i, entry in enumerate(history):
                content.append({
                    "type": "text",
                    "text": f"### Step {i + 1}\nAction: {entry.get('action', 'N/A')}\nObservation:",
                })
                content.extend(entry["obs_content"])

        content.append({"type": "text", "text": "## Current Observation"})
        content.extend(current_obs)

        return content
