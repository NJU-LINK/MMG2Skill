"""Tests for skill formatting utilities (inline image injection)."""

import os
import tempfile

from anything2skill.agent.skill_utils import (
    _format_skill_content_blocks,
    format_skills_for_prompt,
)
from anything2skill.parser.data_types import Skill, Skills


def _make_image(tmpdir: str, name: str) -> str:
    """Create a valid 10x10 PNG file and return its path."""
    from PIL import Image

    path = os.path.join(tmpdir, name)
    img = Image.new("RGB", (10, 10), color=(0, 0, 0))
    img.save(path, format="PNG")
    return path


class TestFormatSkillContentBlocks:
    def test_with_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_image(tmpdir, "shot-1.png")
            skill = Skill(
                name="test",
                description="desc",
                content=(
                    "## Steps\n"
                    "1. Do this\n"
                    "   ![Menu visible](shot-1.png)\n"
                    "2. Do that"
                ),
            )
            blocks = _format_skill_content_blocks(skill, tmpdir)

            types = [b["type"] for b in blocks]
            # text (before img) + text (original markdown ref) + image_url + text (after img)
            assert "image_url" in types

            # Check original markdown reference is preserved
            texts = [b["text"] for b in blocks if b["type"] == "text"]
            assert any("![Menu visible](shot-1.png)" in t for t in texts)

    def test_no_images(self):
        skill = Skill(
            name="test",
            description="desc",
            content="## Steps\n1. Do this\n2. Do that",
        )
        blocks = _format_skill_content_blocks(skill, None)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert "Do this" in blocks[0]["text"]

    def test_missing_image_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Don't create the actual image file
            skill = Skill(
                name="test",
                description="desc",
                content="1. Step\n   ![alt](missing.png)\n2. Next",
            )
            blocks = _format_skill_content_blocks(skill, tmpdir)

            types = [b["type"] for b in blocks]
            assert "image_url" not in types

            # Original markdown reference kept as text
            all_text = " ".join(b["text"] for b in blocks if b["type"] == "text")
            assert "![alt](missing.png)" in all_text


class TestFormatSkillsForPrompt:
    def test_returns_list_of_dicts(self):
        skills = Skills(
            task_id="t1",
            instruction="Do it",
            skills=[
                Skill(name="s1", description="First", content="Step 1"),
                Skill(name="s2", description="Second", content="Step 2"),
            ],
        )
        blocks = format_skills_for_prompt(skills)
        assert isinstance(blocks, list)
        assert all(isinstance(b, dict) for b in blocks)

        # Should contain skill headers
        texts = [b["text"] for b in blocks if b["type"] == "text"]
        assert any("### s1" in t for t in texts)
        assert any("### s2" in t for t in texts)

    def test_empty_skills(self):
        skills = Skills(task_id="t1", instruction="Do it", skills=[])
        blocks = format_skills_for_prompt(skills)
        assert blocks == []
