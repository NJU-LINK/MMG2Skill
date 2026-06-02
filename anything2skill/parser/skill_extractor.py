"""Extract skills from tutorial materials using VLM.

The VLM outputs skills as markdown sections. Each skill starts with
a `# skill-name` header, followed by `> description` blockquote,
then the SOP content with inline ``![alt](filename)`` image references.
Skills are separated by `---`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from anything2skill.parser.data_types import Skill, Skills, TutorialMaterial
from anything2skill.vlm.client import VLMClient, encode_image_files

if TYPE_CHECKING:
    from anything2skill.agent.message_builder import MessageBuilder
    from anything2skill.benchmark_kit import BenchmarkKit

logger = logging.getLogger("anything2skill.parser")


def extract_skills(
    tutorial: TutorialMaterial,
    vlm: VLMClient,
    instruction: str,
    kit: BenchmarkKit,
    llm_params: dict | None = None,
    max_images: int | None = None,
) -> Skills:
    """Extract skills from a tutorial using VLM.

    Args:
        max_images: Maximum number of images to send to VLM. ``None`` means no cap.

    Returns:
        Skills object containing extracted skills.
    """
    image_paths = tutorial.image_paths
    if max_images is not None and len(image_paths) > max_images:
        logger.warning(
            "Tutorial %s has %d images, truncating to %d",
            tutorial.task_id, len(image_paths), max_images,
        )
        image_paths = image_paths[:max_images]

    # (filename, data_url) for VLM; filename->path map for response parsing.
    image_entries = encode_image_files(image_paths)
    path_by_name = {Path(p).name: p for p in image_paths}
    image_map = {fname: path_by_name[fname] for fname, _ in image_entries}

    # Use MessageBuilder for prompt construction
    from anything2skill.agent.message_builder import MessageBuilder
    msg_builder = MessageBuilder(kit)
    messages = msg_builder.build_skill_extraction_messages(
        tutorial.body, instruction, image_entries,
        content_type=tutorial.content_type,
    )

    response = vlm.chat(messages, **(llm_params or {}))

    skill_list = _parse_skills_markdown(response, image_map)

    image_dir = None
    if image_paths:
        image_dir = str(Path(image_paths[0]).parent)

    skills = Skills(
        task_id=tutorial.task_id,
        instruction=instruction,
        skills=skill_list,
        raw_content=tutorial.body,
        image_dir=image_dir,
    )

    logger.info(
        "Extracted %d skills for task %s", len(skill_list), tutorial.task_id
    )
    return skills


def _parse_skills_markdown(
    response: str,
    image_map: dict[str, str],
) -> list[Skill]:
    """Parse VLM markdown response into Skill objects.

    Expected format:
        # skill-name
        > Brief description of what this skill does

        ## Steps
        1. Do this
           ![description of what the image shows](image-1.png)
        2. Do that

        ---

        # another-skill
        > Another description
        ...

    Each skill section starts with `# name` and ends at the next `---` or
    next `# ` header or end of text.
    """
    sections = _split_into_sections(response)

    if not sections:
        logger.warning("Could not parse skill sections, using entire response as one skill")
        return [
            Skill(
                name="task-procedure",
                description="General procedure for the task",
                content=response.strip(),
                images=list(image_map.values()),
            )
        ]

    skills = []
    for name, body in sections:
        description, content, skill_images = _parse_skill_body(body, image_map)
        skills.append(
            Skill(
                name=name,
                description=description,
                content=content,
                images=skill_images,
            )
        )

    return skills


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (name, body) sections.

    Looks for `# skill-name` headers. Sections end at `---`, next `# `,
    or end of text.
    """
    pattern = r"^# +(.+)$"
    matches = list(re.finditer(pattern, text, re.MULTILINE))

    if not matches:
        return []

    sections = []
    for i, match in enumerate(matches):
        name = match.group(1).strip()
        name = re.sub(r"[^a-zA-Z0-9\s-]", "", name)
        name = re.sub(r"\s+", "-", name).lower().strip("-")

        start = match.end()
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        body = text[start:end]
        body = re.sub(r"\n---\s*$", "", body).strip()

        if name and body:
            sections.append((name, body))

    return sections


def _parse_skill_body(
    body: str, image_map: dict[str, str]
) -> tuple[str, str, list[str]]:
    """Parse a skill section body into (description, content, images).

    Extracts:
    - description from `> blockquote` line(s) at the start
    - images from inline ``![alt](filename)`` references in content
    - content is everything else (image references are kept inline)
    """
    lines = body.split("\n")
    description_lines = []
    content_lines = []
    past_description = False

    for line in lines:
        stripped = line.strip()

        # Extract description from blockquote
        if not past_description and stripped.startswith(">"):
            description_lines.append(stripped.lstrip("> ").strip())
            continue

        if description_lines and not stripped:
            past_description = True
            continue

        if not description_lines and not stripped:
            continue

        past_description = True
        content_lines.append(line)

    description = " ".join(description_lines).strip()
    content = "\n".join(content_lines).strip()

    # Extract image paths from inline ![alt](filename) references
    skill_images: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", content):
        filename = match.group(1).strip()
        path = image_map.get(filename)
        if path and path not in seen:
            skill_images.append(path)
            seen.add(path)

    return description, content, skill_images
