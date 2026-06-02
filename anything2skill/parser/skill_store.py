"""Skill persistence: save, load, and cache management.

Cache structure:
  skills_cache/{tutorial_type}/{model}/{bench}/{task_id}/{skill_name}/SKILL.md

The leading ``{tutorial_type}`` partition mirrors
``data_tutorial/{tutorial_type}/...`` so caches built from html and
screenshot tutorials never collide. SKILL.md follows Claude Code format
with YAML frontmatter. Images are referenced back to
data_tutorial/{tutorial_type}/{bench}/{task_id}/tutorial/images/.
"""

from __future__ import annotations

import fcntl
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from anything2skill.parser.data_types import Skill, Skills, TutorialMaterial
from anything2skill.parser.skill_extractor import extract_skills
from anything2skill.vlm.client import VLMClient

if TYPE_CHECKING:
    from anything2skill.benchmark_kit import BenchmarkKit

logger = logging.getLogger("anything2skill.parser")


def _skills_dir(
    cache_dir: str,
    tutorial_type: str,
    model: str,
    benchmark: str,
    task_id: str,
) -> Path:
    """Build the skills cache directory path.

    Returns: cache_dir/{tutorial_type}/{model}/{benchmark}/{task_id}/
    """
    return Path(cache_dir) / tutorial_type / model / benchmark / task_id


def _serialize_skill_md(skill: Skill, image_dir: str | None = None) -> str:
    """Serialize a Skill to SKILL.md content with YAML frontmatter.

    Image paths are persisted as absolute paths so multi-tutorial skills
    (e.g. per-task wiki + shared beginner's guide) reload correctly without
    needing a single ``image_dir`` root. The ``image_dir`` parameter is kept
    for API compatibility but no longer consulted.
    """
    lines = ["---"]
    lines.append(f"name: {skill.name}")
    lines.append(f"description: {skill.description}")

    if skill.images:
        lines.append("images:")
        for img_path in skill.images:
            abs_path = str(Path(img_path).resolve())
            lines.append(f"  - {abs_path}")

    lines.append("---")
    lines.append("")
    lines.append(skill.content)

    return "\n".join(lines)


def _parse_skill_md(
    text: str, skill_dir: Path, image_dir: str | None = None,
) -> Skill:
    """Parse a SKILL.md file into a Skill object.

    Frontmatter ``images:`` entries are expected to be absolute paths
    (written by :func:`_serialize_skill_md`). Missing files are skipped with
    a warning — callers should not rely on ``image_dir`` fallback.
    """
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return Skill(
            name=skill_dir.name,
            description="",
            content=text.strip(),
        )

    fm_text = match.group(1)
    body = text[match.end():].strip()

    # Parse frontmatter
    name = skill_dir.name
    description = ""
    image_entries: list[str] = []

    in_images = False
    for line in fm_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("name:"):
            name = stripped.split(":", 1)[1].strip()
            in_images = False
        elif stripped.startswith("description:"):
            description = stripped.split(":", 1)[1].strip()
            in_images = False
        elif stripped == "images:":
            in_images = True
        elif in_images and stripped.startswith("- "):
            image_entries.append(stripped[2:].strip())
        else:
            in_images = False

    images: list[str] = []
    for entry in image_entries:
        img_path = Path(entry)
        if img_path.is_file():
            images.append(str(img_path.resolve()))
        else:
            logger.warning("Skill image not found: %s", img_path)

    return Skill(
        name=name,
        description=description,
        content=body,
        images=images,
    )


def save_skills(
    skills: Skills,
    cache_dir: str,
    tutorial_type: str,
    model: str,
    benchmark: str,
) -> Path:
    """Save Skills to disk as SKILL.md files.

    Creates: cache_dir/{tutorial_type}/{model}/{benchmark}/{task_id}/{skill_name}/SKILL.md

    Returns:
        Path to the skills model directory.
    """
    out_dir = _skills_dir(
        cache_dir, tutorial_type, model, benchmark, skills.task_id,
    )

    for skill in skills.skills:
        skill_dir = out_dir / skill.name
        skill_dir.mkdir(parents=True, exist_ok=True)

        content = _serialize_skill_md(skill, skills.image_dir)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(content, encoding="utf-8")

    logger.info("Saved %d skills to %s", len(skills.skills), out_dir)
    return out_dir


def load_skills(
    task_id: str,
    cache_dir: str,
    tutorial_type: str,
    model: str,
    benchmark: str,
    raw_content: str = "",
    instruction: str = "",
    image_dir: str | None = None,
) -> Skills | None:
    """Load Skills from cached SKILL.md files.

    Returns:
        Skills if cached, None otherwise.
    """
    skills_root = _skills_dir(
        cache_dir, tutorial_type, model, benchmark, task_id,
    )
    if not skills_root.is_dir():
        return None

    skill_list: list[Skill] = []
    for skill_dir in sorted(skills_root.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        try:
            text = skill_md.read_text(encoding="utf-8")
            skill = _parse_skill_md(text, skill_dir, image_dir)
            skill_list.append(skill)
        except Exception as e:
            logger.warning("Failed to load skill %s: %s", skill_dir.name, e)

    if not skill_list:
        return None

    logger.info("Loaded %d cached skills for task %s", len(skill_list), task_id)
    return Skills(
        task_id=task_id,
        instruction=instruction,
        skills=skill_list,
        raw_content=raw_content,
        image_dir=image_dir,
    )


def get_or_extract_skills(
    tutorial: TutorialMaterial,
    vlm: VLMClient,
    instruction: str,
    cache_dir: str,
    tutorial_type: str,
    benchmark: str,
    model: str,
    force_regenerate: bool = False,
    kit: BenchmarkKit = None,
    llm_params: dict | None = None,
    max_images: int | None = None,
) -> Skills:
    """Get Skills from cache or extract via VLM.

    Args:
        tutorial: Tutorial material. Its ``task_id`` is the cache key —
            when a tutorial is shared across tasks, the caller rebinds it
            via ``load_tutorials(..., task_id=task.task_id)`` before
            reaching here.
        vlm: VLM client.
        instruction: Task instruction.
        cache_dir: Skills cache root directory (e.g., "skills_cache").
        tutorial_type: Modality partition for the cache layout
            (``html`` / ``screenshot`` / ``video``). Caches built from
            different modalities never collide.
        benchmark: Benchmark name (e.g., "osworld").
        model: Model name (for per-model caching).
        force_regenerate: If True, always regenerate.
        kit: BenchmarkKit providing skill extraction prompt.
        llm_params: Unified VLM call parameters.
        max_images: Maximum images to send to VLM. ``None`` means no cap.

    Returns:
        Skills (loaded from cache or freshly extracted).
    """
    # Determine image_dir from tutorial
    image_dir = None
    if tutorial.image_paths:
        image_dir = str(Path(tutorial.image_paths[0]).parent)

    # Use a file lock to prevent parallel workers from extracting simultaneously.
    lock_dir = _skills_dir(
        cache_dir, tutorial_type, model, benchmark, tutorial.task_id,
    ).parent
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f".{tutorial.task_id}.lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if not force_regenerate:
                cached = load_skills(
                    tutorial.task_id, cache_dir, tutorial_type, model, benchmark,
                    raw_content=tutorial.body,
                    instruction=instruction,
                    image_dir=image_dir,
                )
                if cached is not None:
                    return cached

            logger.info(
                "Extracting skills for task %s (type=%s, model=%s, force=%s)",
                tutorial.task_id, tutorial_type, model, force_regenerate,
            )
            skills = extract_skills(
                tutorial, vlm, instruction, kit=kit, llm_params=llm_params,
                max_images=max_images,
            )
            save_skills(skills, cache_dir, tutorial_type, model, benchmark)
            return skills
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
