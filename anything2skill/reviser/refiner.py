"""ReviserRefiner: tutorial-aware VLM skill refinement.

Inputs:
    - instruction
    - current Skills object
    - RootCauseAnalysis (raw XML is re-used verbatim in the prompt)
    - TutorialMaterial (body + image paths)

Output:
    - new Skills (parsed back from VLM markdown via
      ``parser.skill_extractor._parse_skills_markdown``).

Image map is built from **both** the original skill images and the tutorial
images; tutorial entries win when filenames collide, because the refiner is
aligning skills to the tutorial.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape as _xml_escape

from anything2skill.reviser.data_types import RootCauseAnalysis
from anything2skill.reviser.prompts import (
    REVISER_REFINE_SYSTEM_TMPL,
    REVISER_REFINE_USER_TMPL,
)

if TYPE_CHECKING:
    from anything2skill.benchmark_kit import BenchmarkKit
    from anything2skill.parser.data_types import Skills, TutorialMaterial
    from anything2skill.vlm.client import VLMClient

logger = logging.getLogger("anything2skill.reviser")


class ReviserRefiner:
    """Phase-2 refiner: rewrite skills given root-cause XML + tutorial."""

    def __init__(
        self,
        vlm: VLMClient,
        kit: BenchmarkKit,
        *,
        max_tokens: int = 4000,
        temperature: float | None = None,
        tutorial_image_cap: int = 8,
        include_tutorial_in_refine: bool = True,
    ):
        self.vlm = vlm
        self.kit = kit
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tutorial_image_cap = max(0, int(tutorial_image_cap))
        self.include_tutorial_in_refine = bool(include_tutorial_in_refine)

    def refine(
        self,
        skills: Skills,
        root_cause: RootCauseAnalysis,
        tutorial: TutorialMaterial,
        instruction: str,
        history: list[RootCauseAnalysis] | None = None,
    ) -> Skills:
        """Produce refined skills by calling the VLM.

        When ``root_cause.issues`` is empty the prompt's ``likely_success``
        block steers the VLM into reinforce mode (absorb ``<what_worked>``
        into silent skills, no other edits); with issues present the usual
        edit-intensity rules in ``prompts.py`` apply.
        """
        from anything2skill.parser.data_types import Skills as SkillsType
        from anything2skill.parser.skill_extractor import _parse_skills_markdown

        original_skills_md = _format_skills_as_markdown(skills)
        root_cause_xml = root_cause.raw_xml.strip() or _serialize_root_cause_fallback(root_cause)

        image_entries, image_map = self._build_image_inputs(skills, tutorial)

        system_prompt = REVISER_REFINE_SYSTEM_TMPL.format(
            domain_reviser_guidance=(self.kit.reviser_guidance or "").strip(),
        ).strip()

        history_block = _format_history_block(history)
        trajectory_summary_block = _format_trajectory_summary_block(
            root_cause.trajectory_summary
        )

        if not self.include_tutorial_in_refine:
            tutorial_body_slot = "(tutorial body omitted per config)"
        elif tutorial.content_type == "screenshot":
            tutorial_body_slot = "(tutorial provided as screenshots below)"
        else:
            tutorial_body_slot = tutorial.body

        user_text = REVISER_REFINE_USER_TMPL.format(
            instruction=instruction,
            history_block=history_block,
            root_cause_xml=root_cause_xml,
            trajectory_summary_block=trajectory_summary_block,
            original_skills=original_skills_md,
            tutorial_body=tutorial_body_slot,
        ).strip()

        user_content: list[dict] = [{"type": "text", "text": user_text}]
        if image_entries:
            user_content.append(
                {
                    "type": "text",
                    "text": f"\nTutorial / skill images available ({len(image_entries)}):",
                }
            )
            for filename, data_url in image_entries:
                user_content.append({"type": "text", "text": f"[Image: {filename}]"})
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    }
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            response = self.vlm.chat(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error("[Reviser] Refiner VLM call failed: %s", e)
            return skills

        refined_list = _parse_skills_markdown(response, image_map)
        if not refined_list:
            logger.warning(
                "[Reviser] Refinement produced no parseable skills, keeping originals"
            )
            return skills

        logger.info(
            "[Reviser] Refined %d skills (was %d)",
            len(refined_list), len(skills.skills),
        )
        return SkillsType(
            task_id=skills.task_id,
            instruction=instruction,
            skills=refined_list,
            raw_content=skills.raw_content,
            image_dir=skills.image_dir,
        )

    # ------------------------------------------------------------------

    def _build_image_inputs(
        self,
        skills: Skills,
        tutorial: TutorialMaterial,
    ) -> tuple[list[tuple[str, str]], dict[str, str]]:
        """Build the (filename, data_url) list plus the filename→abs_path map.

        Both the original skill images and the tutorial images contribute;
        when filenames collide, the tutorial wins (refiner is aligning
        skills to the tutorial). Tutorial images are capped by
        ``tutorial_image_cap``.
        """
        from anything2skill.vlm.client import encode_image_files

        image_map: dict[str, str] = {}

        # Existing skill images first — so the tutorial overlay wins.
        for skill in skills.skills:
            for img_path in skill.images:
                name = Path(img_path).name
                if name not in image_map and Path(img_path).is_file():
                    image_map[name] = img_path

        capped = self._select_tutorial_images(tutorial.image_paths or [])
        for img_path in capped:
            name = Path(img_path).name
            if Path(img_path).is_file():
                image_map[name] = img_path  # tutorial wins on collision

        image_entries: list[tuple[str, str]] = []
        if self.include_tutorial_in_refine:
            image_entries = encode_image_files(list(image_map.values()))

        return image_entries, image_map

    def _select_tutorial_images(self, image_paths: list[str]) -> list[str]:
        """Apply include/cap policy to the tutorial image list.

        Returns the subset of ``image_paths`` the refiner should surface.
        Logs a one-line info when the cap actually truncated the list so the
        operator can notice missing pages.
        """
        if not self.include_tutorial_in_refine:
            return []
        if self.tutorial_image_cap <= 0:
            return list(image_paths)
        capped = image_paths[: self.tutorial_image_cap]
        if len(image_paths) > len(capped):
            logger.info(
                "[Reviser] Tutorial image list truncated to %d (cap=%d)",
                len(capped), self.tutorial_image_cap,
            )
        return capped


def _format_skills_as_markdown(skills: Skills) -> str:
    """Render a Skills object into the extractor's markdown shape."""
    sections: list[str] = []
    for skill in skills.skills:
        lines = [f"# {skill.name}"]
        if skill.description:
            lines.append(f"> {skill.description}")
        lines.append("")
        lines.append(skill.content)
        sections.append("\n".join(lines))
    return "\n\n---\n\n".join(sections)


def _serialize_root_cause_fallback(rc: RootCauseAnalysis) -> str:
    """Reconstruct ``<root_cause>`` XML when raw_xml was not preserved.

    Text bodies are escaped so that evidence / rationale containing
    ``&`` / ``<`` / ``>`` doesn't produce malformed XML that the regex
    fallback would then have to salvage.
    """
    out = ["<root_cause>"]
    out.append(
        f"  <trajectory_summary>{_xml_escape(rc.trajectory_summary)}"
        "</trajectory_summary>"
    )
    out.append("  <what_worked>")
    for item in rc.what_worked:
        out.append(f"    <item>{_xml_escape(item)}</item>")
    out.append("  </what_worked>")
    out.append("  <issues>")
    for issue in rc.issues:
        out.append("    <issue>")
        out.append(f"      <where>{_xml_escape(issue.get('where', ''))}</where>")
        out.append(
            f"      <evidence>{_xml_escape(issue.get('evidence', ''))}</evidence>"
        )
        out.append(f"      <cause>{_xml_escape(issue.get('cause', ''))}</cause>")
        out.append("    </issue>")
    out.append("  </issues>")
    # Hardcode double quotes on the attribute to match _OUTCOME_RE, and
    # escape embedded quotes in the value (the escape helper's attribute
    # overload converts " → &quot; and & → &amp;).
    value_escaped = _xml_escape(
        rc.outcome_assessment, {'"': "&quot;"},
    )
    out.append(
        f'  <outcome_assessment value="{value_escaped}">'
        f"{_xml_escape(rc.outcome_rationale)}</outcome_assessment>"
    )
    out.append("</root_cause>")
    return "\n".join(out)


def _format_history_block(history: list[RootCauseAnalysis] | None) -> str:
    """Render chronological prior root_cause XMLs for the refiner prompt.

    Returns an empty string when ``history`` is empty/None, so the outer
    template renders without a dangling section header.
    """
    if not history:
        return ""
    lines = ["## Past attempts root cause (chronological)"]
    for i, rc in enumerate(history, start=1):
        xml = rc.raw_xml.strip() or _serialize_root_cause_fallback(rc)
        lines.append(f"### Attempt {i}")
        lines.append(xml)
    lines.append("")  # trailing blank so the next section has spacing
    return "\n".join(lines) + "\n"


def _format_trajectory_summary_block(summary: str) -> str:
    """Render the optional trajectory summary section."""
    text = (summary or "").strip()
    if not text:
        return ""
    return f"\n## Trajectory Summary (current attempt)\n{text}\n"
