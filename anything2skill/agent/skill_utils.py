"""Shared skill formatting utilities.

These functions operate on the generic Skills/Skill data types and are used
by the MessageBuilder and agents.  They are domain-agnostic.
"""

from __future__ import annotations

import re
from pathlib import Path

from anything2skill.parser.data_types import Skill, Skills
from anything2skill.vlm.client import encode_image_file


def format_skills_for_prompt(skills: Skills) -> list[dict]:
    """Format all skills as interleaved text/image content blocks.

    Inline ``![alt](filename)`` references in skill content are resolved
    to ``[Image: filename]`` label + ``image_url`` blocks.  Text surrounding
    image references is emitted as ``text`` blocks.

    Returns:
        List of VLM content blocks (``{"type": "text", ...}`` or
        ``{"type": "image_url", ...}``).
    """
    blocks: list[dict] = []
    for skill in skills.skills:
        blocks.append({
            "type": "text",
            "text": f"### {skill.name}\n> {skill.description}\n",
        })
        blocks.extend(_format_skill_content_blocks(skill, skills.image_dir))
    return blocks


def _format_skill_content_blocks(
    skill: Skill, image_dir: str | None,
) -> list[dict]:
    """Split a single skill's content into interleaved text + image blocks.

    For each ``![alt](filename)`` in *content*:
    - preceding text → ``{"type": "text"}``
    - the reference  → ``{"type": "text", "text": "[Image: filename]"}``
                      + ``{"type": "image_url", ...}``

    If an image file cannot be found the reference is kept as plain text.
    """
    content = skill.content
    pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    blocks: list[dict] = []
    last_end = 0

    for match in pattern.finditer(content):
        filename = match.group(2).strip()
        resolved = _resolve_image(filename, image_dir)

        # Text up to (and including) the image reference
        text_end = match.end()
        preceding = content[last_end:text_end].strip()
        if preceding:
            blocks.append({"type": "text", "text": preceding})
        last_end = text_end

        # Insert actual image block right after the markdown reference
        if resolved:
            blocks.append({
                "type": "image_url",
                "image_url": {"url": resolved, "detail": "high"},
            })

    # Remaining text after last image reference
    remaining = content[last_end:].strip()
    if remaining:
        blocks.append({"type": "text", "text": remaining})

    return blocks


def _resolve_image(filename: str, image_dir: str | None) -> str | None:
    """Resolve a filename to a base64 data URL, or None if not found."""
    if not image_dir:
        return None
    img_path = Path(image_dir) / filename
    if not img_path.exists():
        return None
    try:
        return encode_image_file(str(img_path))
    except Exception:
        return None
