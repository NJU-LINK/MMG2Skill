"""Skill directory I/O used by :class:`ReviserRunner`."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from anything2skill.parser.data_types import Skills
from anything2skill.parser.skill_store import _parse_skill_md, _serialize_skill_md

logger = logging.getLogger("anything2skill.reviser")


def load_skills_from_dir(
    skills_root: str | Path,
    task_id: str,
    instruction: str,
    image_dir: str | None = None,
) -> Skills | None:
    """Load Skills from ``<skills_root>/<skill-name>/SKILL.md`` files.

    Uses :func:`_parse_skill_md` (YAML frontmatter, absolute image paths) —
    **not** :func:`_parse_skills_markdown`, which parses the VLM's
    ``# skill-name`` markdown output and is a totally different format.
    """
    root = Path(skills_root)
    if not root.is_dir():
        return None

    skill_list = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        skill_md = sub / "SKILL.md"
        if not skill_md.is_file():
            continue
        try:
            text = skill_md.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to read %s: %s", skill_md, e)
            continue
        try:
            skill_list.append(_parse_skill_md(text, sub, image_dir))
        except Exception as e:
            logger.warning("Failed to parse %s: %s", skill_md, e)

    if not skill_list:
        return None

    return Skills(
        task_id=task_id,
        instruction=instruction,
        skills=skill_list,
        image_dir=image_dir,
    )


def save_skills_to_dir(skills: Skills, skills_root: str | Path) -> None:
    """Persist each skill as ``<skills_root>/<skill-name>/SKILL.md``.

    Any existing ``skills_root`` is removed first so the final contents
    exactly match ``skills``. Frontmatter uses absolute paths (see
    ``_serialize_skill_md``).
    """
    root = Path(skills_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    for skill in skills.skills:
        skill_dir = root / skill.name
        skill_dir.mkdir()
        content = _serialize_skill_md(skill)
        _atomic_write_text(skill_dir / "SKILL.md", content)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write *content* atomically: temp file + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
