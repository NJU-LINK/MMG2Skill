"""Tests for MessageBuilder dispatch on tutorial.content_type.

Two builders fan out by modality:

- ``_build_tutorial_block`` (used by VanillaTutorialAgent's first user turn)
- ``build_skill_extraction_messages`` (used by SimpleAgent / PhasedAgent
  via :func:`parser.skill_extractor.extract_skills`)

The legacy html path must keep its current rendering; the new screenshot
path must skip the (empty) tutorial body label and instead announce the
screenshots.
"""

from __future__ import annotations

import pytest

from anything2skill.agent.message_builder import MessageBuilder
from anything2skill.benchmarks.osworld.kit import OSWorldKit
from anything2skill.parser.data_types import TutorialMaterial


_kit = OSWorldKit()


def _flatten_text_blocks(blocks: list[dict]) -> str:
    return "\n".join(b["text"] for b in blocks if b.get("type") == "text")


def _make_tutorial(
    content_type: str, body: str, image_paths: list[str] | None = None,
) -> TutorialMaterial:
    return TutorialMaterial(
        task_id="t1",
        instruction="x",
        content_type=content_type,
        body=body,
        image_paths=list(image_paths or []),
    )


class TestBuildTutorialBlockDispatch:
    """`_build_tutorial_block` must dispatch on tutorial.content_type."""

    def test_html_renders_body(self):
        mb = MessageBuilder(_kit)
        tut = _make_tutorial("html", "<html>steps</html>")
        blocks = mb._build_tutorial_block(tut, max_images=None)
        text = _flatten_text_blocks(blocks)
        assert "## Reference Tutorial" in text
        # html: the body is a separate text block (no "(screenshots)" suffix).
        assert "<html>steps</html>" in text
        assert "(screenshots)" not in text

    def test_screenshot_skips_body(self, tmp_path, monkeypatch):
        # Stub encoder so we don't need real PNG bytes.
        from anything2skill.vlm import client as vlm_client
        monkeypatch.setattr(
            vlm_client, "encode_image_file",
            lambda p: f"data:image/png;base64,FAKE_{p}",
        )

        img = tmp_path / "frame_001.png"
        img.write_bytes(b"\x89PNG")
        mb = MessageBuilder(_kit)
        tut = _make_tutorial("screenshot", body="", image_paths=[str(img)])
        blocks = mb._build_tutorial_block(tut, max_images=None)
        text = _flatten_text_blocks(blocks)

        assert "## Reference Tutorial (screenshots)" in text
        # Must not render an empty body line that would dangle a label.
        assert "## Reference Tutorial\n" not in text
        # An image_url block should be present for the frame.
        assert any(b.get("type") == "image_url" for b in blocks)

    def test_unknown_content_type_raises(self):
        mb = MessageBuilder(_kit)
        tut = _make_tutorial("video", body="")
        with pytest.raises(NotImplementedError):
            mb._build_tutorial_block(tut, max_images=None)


class TestSkillExtractionMessagesDispatch:
    """`build_skill_extraction_messages` must dispatch on content_type."""

    def test_html_keeps_legacy_label(self):
        mb = MessageBuilder(_kit)
        msgs = mb.build_skill_extraction_messages(
            tutorial_content="<html>step 1</html>",
            instruction="do thing",
            image_entries=[],
            content_type="html",
        )
        user_text = msgs[1]["content"][0]["text"]
        assert "TASK: do thing" in user_text
        assert "TUTORIAL:" in user_text
        assert "<html>step 1</html>" in user_text

    def test_screenshot_drops_tutorial_label(self):
        mb = MessageBuilder(_kit)
        msgs = mb.build_skill_extraction_messages(
            tutorial_content="",
            instruction="do thing",
            image_entries=[],
            content_type="screenshot",
        )
        user_text = msgs[1]["content"][0]["text"]
        assert "TASK: do thing" in user_text
        # No empty TUTORIAL: line; the screenshots phrase replaces it.
        assert "TUTORIAL:" not in user_text
        assert "screenshots below" in user_text

    def test_unknown_content_type_raises(self):
        mb = MessageBuilder(_kit)
        with pytest.raises(NotImplementedError):
            mb.build_skill_extraction_messages(
                tutorial_content="",
                instruction="x",
                image_entries=[],
                content_type="video",
            )
