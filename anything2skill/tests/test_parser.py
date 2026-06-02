"""Tests for tutorial loading, skill data types, and skill extraction parsing."""

import json
import logging
import os
import tempfile
from pathlib import Path

import pytest

from anything2skill.parser.data_types import Skill, Skills, TutorialMaterial
from anything2skill.parser.skill_extractor import _parse_skill_body
from anything2skill.parser.skill_store import (
    _parse_skill_md,
    _serialize_skill_md,
    load_skills,
    save_skills,
)
from anything2skill.parser.tutorial_loader import load_tutorial


class TestTutorialLoading:
    def test_load_html(self):
        """Loads raw HTML and reads metadata from metadata.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_id = "test-task-001"
            tutorial_dir = os.path.join(tmpdir, task_id, "tutorial")
            os.makedirs(tutorial_dir)

            with open(os.path.join(tutorial_dir, "page.html"), "w") as f:
                f.write("<html><body><h1>Tutorial</h1></body></html>")
            with open(os.path.join(tutorial_dir, "metadata.json"), "w") as f:
                json.dump({
                    "task_id": "test-task-001",
                    "instruction": "Turn on Bluetooth",
                    "source_url": "https://example.com",
                }, f)

            tutorial = load_tutorial(task_id, tmpdir, "html")
            assert tutorial.task_id == "test-task-001"
            assert tutorial.instruction == "Turn on Bluetooth"
            assert tutorial.content_type == "html"
            assert "<h1>Tutorial</h1>" in tutorial.body

    def test_load_tutorial_with_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_id = "test-task-002"
            tutorial_dir = os.path.join(tmpdir, task_id, "tutorial")
            images_dir = os.path.join(tutorial_dir, "images")
            os.makedirs(images_dir)

            with open(os.path.join(images_dir, "screenshot.png"), "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

            with open(os.path.join(tutorial_dir, "page.html"), "w") as f:
                f.write("<html><body><p>Steps</p></body></html>")

            tutorial = load_tutorial(task_id, tmpdir, "html")
            assert len(tutorial.image_paths) == 1
            assert tutorial.image_paths[0].endswith("screenshot.png")

    def test_html_with_local_image_paths(self):
        """HTML with already-rewritten local image paths loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_id = "test-rewrite"
            tutorial_dir = os.path.join(tmpdir, task_id, "tutorial")
            images_dir = os.path.join(tutorial_dir, "images")
            os.makedirs(images_dir)

            with open(os.path.join(images_dir, "step1.png"), "wb") as f:
                f.write(b"\x89PNG" + b"\x00" * 100)

            # HTML already has local paths (rewritten at download time)
            html = '<html><body><img src="images/step1.png"></body></html>'
            with open(os.path.join(tutorial_dir, "page.html"), "w") as f:
                f.write(html)

            tutorial = load_tutorial(task_id, tmpdir, "html")
            assert 'images/step1.png' in tutorial.body

    def test_load_missing_tutorial(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_tutorial("nonexistent", tmpdir, "html")

    def test_load_screenshot_only(self, tmp_path):
        """Screenshot type: images/ + metadata.json, no page.html, body == ''."""
        task_id = "task-screenshot"
        tutorial_dir = tmp_path / task_id / "tutorial"
        images_dir = tutorial_dir / "images"
        images_dir.mkdir(parents=True)
        for name in ("frame_002.png", "frame_001.png", "frame_003.png"):
            _make_png(images_dir / name)
        (tutorial_dir / "metadata.json").write_text(
            json.dumps({
                "task_id": task_id,
                "instruction": "Capture a screenshot",
                "source_url": "https://example.com",
                "content_type": "screenshot",
            }),
            encoding="utf-8",
        )

        tutorial = load_tutorial(task_id, str(tmp_path), "screenshot")
        assert tutorial.content_type == "screenshot"
        assert tutorial.body == ""
        assert [Path(p).name for p in tutorial.image_paths] == [
            "frame_001.png", "frame_002.png", "frame_003.png",
        ]

    def test_html_requires_page_html(self, tmp_path):
        """tutorial_type='html' on an images-only dir raises FileNotFoundError."""
        task_id = "task-no-html"
        tutorial_dir = tmp_path / task_id / "tutorial"
        images_dir = tutorial_dir / "images"
        images_dir.mkdir(parents=True)
        _make_png(images_dir / "frame_001.png")

        with pytest.raises(FileNotFoundError):
            load_tutorial(task_id, str(tmp_path), "html")

    def test_screenshot_requires_images(self, tmp_path):
        """tutorial_type='screenshot' on an html-only dir raises FileNotFoundError."""
        task_id = "task-no-images"
        tutorial_dir = tmp_path / task_id / "tutorial"
        tutorial_dir.mkdir(parents=True)
        (tutorial_dir / "page.html").write_text(
            "<html><body>x</body></html>", encoding="utf-8",
        )

        with pytest.raises(FileNotFoundError):
            load_tutorial(task_id, str(tmp_path), "screenshot")

    def test_metadata_content_type_mismatch_warns(self, tmp_path, caplog):
        """metadata content_type conflict: caller value wins, warning is logged."""
        task_id = "task-mismatch"
        tutorial_dir = tmp_path / task_id / "tutorial"
        images_dir = tutorial_dir / "images"
        images_dir.mkdir(parents=True)
        _make_png(images_dir / "frame_001.png")
        (tutorial_dir / "metadata.json").write_text(
            json.dumps({
                "task_id": task_id,
                "instruction": "x",
                "content_type": "html",
            }),
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING, logger="anything2skill.parser"):
            tutorial = load_tutorial(task_id, str(tmp_path), "screenshot")

        assert tutorial.content_type == "screenshot"
        assert any(
            "content_type" in rec.message and "conflicts" in rec.message
            for rec in caplog.records
        )


class TestSkillDataTypes:
    def test_skill_creation(self):
        skill = Skill(
            name="open-terminal",
            description="Open a terminal on Ubuntu",
            content="## Steps\n1. Right-click desktop\n2. Select Open Terminal",
            images=["/path/to/img.png"],
        )
        assert skill.name == "open-terminal"
        assert skill.description == "Open a terminal on Ubuntu"
        assert "Right-click" in skill.content
        assert len(skill.images) == 1

    def test_skills_collection(self):
        skills = Skills(
            task_id="test-001",
            instruction="Do the thing",
            skills=[
                Skill(name="step-a", description="First step", content="Do A"),
                Skill(name="step-b", description="Second step", content="Do B"),
            ],
            raw_content="# Tutorial",
            image_dir="/path/to/images",
        )
        assert len(skills.skills) == 2
        assert skills.skills[0].name == "step-a"
        assert skills.task_id == "test-001"

    def test_skill_default_images(self):
        skill = Skill(name="test", description="test", content="test")
        assert skill.images == []


class TestParseSkillBody:
    """Tests for _parse_skill_body() inline image extraction."""

    def test_inline_images(self):
        body = (
            "> Open a terminal\n"
            "\n"
            "## Steps\n"
            "1. Right-click on desktop\n"
            "   ![Context menu showing options](shot-1.png)\n"
            "2. Select Open Terminal\n"
            "   ![Terminal window opened](shot-3.png)\n"
        )
        image_map = {
            "shot-1.png": "/imgs/shot-1.png",
            "shot-3.png": "/imgs/shot-3.png",
        }
        desc, content, images = _parse_skill_body(body, image_map)
        assert desc == "Open a terminal"
        assert "![Context menu showing options](shot-1.png)" in content
        assert "![Terminal window opened](shot-3.png)" in content
        assert images == ["/imgs/shot-1.png", "/imgs/shot-3.png"]

    def test_no_images(self):
        body = "> Do something\n\n## Steps\n1. Step one\n2. Step two\n"
        desc, content, images = _parse_skill_body(body, {})
        assert desc == "Do something"
        assert "Step one" in content
        assert images == []

    def test_dedup_images(self):
        body = (
            "> Desc\n"
            "\n"
            "1. Step\n"
            "   ![first ref](same.png)\n"
            "2. Step\n"
            "   ![second ref](same.png)\n"
        )
        image_map = {"same.png": "/imgs/same.png"}
        _, _, images = _parse_skill_body(body, image_map)
        assert images == ["/imgs/same.png"]

    def test_unresolved_image(self):
        body = "> Desc\n\n1. Step\n   ![alt](missing.png)\n"
        image_map = {}  # missing.png not in map
        _, content, images = _parse_skill_body(body, image_map)
        assert images == []
        assert "![alt](missing.png)" in content


def _make_png(path: Path) -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128)


class TestSkillStoreFrontmatter:
    """Serialize / parse must round-trip absolute image paths across tutorials."""

    def test_serialize_reload_roundtrip_multi_tutorial(self, tmp_path):
        tutA = tmp_path / "tutorial_A" / "images"
        tutB = tmp_path / "tutorial_B" / "images"
        tutA.mkdir(parents=True)
        tutB.mkdir(parents=True)
        img_a = tutA / "a.png"
        img_b = tutB / "b.png"
        _make_png(img_a)
        _make_png(img_b)

        skill = Skill(
            name="compound",
            description="uses images from two tutorials",
            content="## Steps\n1. Look at a.png\n2. Check b.png",
            images=[str(img_a), str(img_b)],
        )
        skills = Skills(
            task_id="t1",
            instruction="do it",
            skills=[skill],
            raw_content="",
            image_dir=str(tutA),
        )

        cache_dir = tmp_path / "skills_cache"
        save_skills(skills, str(cache_dir), "html", "gpt-4o", "osworld")

        loaded = load_skills(
            task_id="t1",
            cache_dir=str(cache_dir),
            tutorial_type="html",
            model="gpt-4o",
            benchmark="osworld",
            image_dir=str(tutA),  # unused internally now
        )
        assert loaded is not None
        assert len(loaded.skills) == 1
        images = loaded.skills[0].images
        assert len(images) == 2
        assert str(img_a.resolve()) in images
        assert str(img_b.resolve()) in images

    def test_parse_skill_md_missing_image_logs_warning(self, tmp_path, caplog):
        """Absolute path in frontmatter that doesn't exist on disk: warn, don't raise."""
        skill_dir = tmp_path / "broken-skill"
        skill_dir.mkdir()

        # Write a SKILL.md referencing a path that doesn't exist
        missing_path = tmp_path / "does-not-exist" / "ghost.png"
        md_text = (
            "---\n"
            "name: broken-skill\n"
            "description: Points to a missing image\n"
            "images:\n"
            f"  - {missing_path}\n"
            "---\n"
            "\n"
            "## Steps\n"
            "1. Look at nothing\n"
        )

        with caplog.at_level(logging.WARNING, logger="anything2skill.parser"):
            skill = _parse_skill_md(md_text, skill_dir)

        assert skill.name == "broken-skill"
        assert skill.description == "Points to a missing image"
        assert skill.images == []
        assert any(
            "Skill image not found" in record.message for record in caplog.records
        )

    def test_serialize_uses_absolute_path_even_when_input_relative(self, tmp_path):
        real_img = tmp_path / "images" / "x.png"
        real_img.parent.mkdir(parents=True)
        _make_png(real_img)

        # Pass a relative path to the skill (simulating Path(img).resolve() upstream)
        skill = Skill(
            name="s",
            description="d",
            content="body",
            images=[str(real_img)],
        )
        md = _serialize_skill_md(skill)
        assert f"  - {real_img.resolve()}" in md
        assert "images:" in md
