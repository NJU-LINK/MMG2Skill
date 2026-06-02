"""Data structures for tutorials and skills."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TutorialMaterial:
    """Raw tutorial material loaded from disk.

    The body can be any format (HTML, markdown, plain text, etc.).
    The content_type field indicates the format so downstream consumers
    (VLM skill extraction) can handle it appropriately.
    """

    task_id: str
    instruction: str
    content_type: str  # "html" | "screenshot" | "video" (reserved)
    body: str  # Raw tutorial content for html; empty for screenshot
    image_paths: list[str] = field(default_factory=list)  # Absolute paths to images
    source_url: str = ""


@dataclass
class Skill:
    """A single skill extracted from a tutorial, stored as SKILL.md.

    Follows Claude Code skill format: YAML frontmatter (name, description,
    images) + markdown body containing the SOP procedure.
    """

    name: str  # kebab-case, used as directory name
    description: str  # Brief description for agent skill selection
    content: str  # Markdown body (SOP procedure)
    images: list[str] = field(default_factory=list)  # Absolute paths to tutorial images


@dataclass
class Skills:
    """All skills extracted from a tutorial for a given task."""

    task_id: str
    instruction: str
    skills: list[Skill] = field(default_factory=list)
    raw_content: str = ""  # Original tutorial content for fallback
    image_dir: str | None = None  # tutorial/images/ absolute path
