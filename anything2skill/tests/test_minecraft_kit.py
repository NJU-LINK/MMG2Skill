from __future__ import annotations

"""Tests for the Minecraft benchmark kit."""

import base64
import io
import json

from PIL import Image

from anything2skill.benchmarks.minecraft.kit import MinecraftKit, MinecraftTask

_kit = MinecraftKit()


def _make_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10), color=(0, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_parse_actions_done():
    assert _kit.parse_actions("DONE") == ["DONE"]


def test_parse_actions_fail():
    assert _kit.parse_actions("FAIL") == ["FAIL"]


def test_parse_actions_wait():
    assert _kit.parse_actions("WAIT") == ["WAIT"]


def test_parse_actions_single_action():
    response = "I need to move.\n<action>move(10, 5)</action>"
    assert _kit.parse_actions(response) == ["move(10, 5)"]


def test_parse_actions_compound_action():
    response = "<action>move(0, 5) and click(left)</action>"
    assert _kit.parse_actions(response) == ["move(0, 5) and click(left)"]


def test_parse_actions_multiple_blocks():
    response = "<action>press(w)</action>\n<action>click(left)</action>"
    assert _kit.parse_actions(response) == ["press(w)", "click(left)"]


def test_parse_actions_done_in_tag():
    assert _kit.parse_actions("<action>DONE</action>") == ["DONE"]


def test_parse_actions_no_action_fallback():
    assert _kit.parse_actions("I am thinking about what to do...") == ["WAIT"]


def test_parse_actions_case_insensitive_signal():
    assert _kit.parse_actions("done") == ["DONE"]


def test_parse_actions_ignores_thinking_tokens():
    # Regression: Kimi-class models emit `</think>` on the same line as the
    # action; the old `^ACTION:` anchor failed on this and swallowed every
    # action as WAIT. With `<action>` tags the thinking stream is orthogonal.
    response = (
        "<think>The cursor is near the wheat; click to pick it up.</think> "
        "<action>move(10, 20)</action>"
    )
    assert _kit.parse_actions(response) == ["move(10, 20)"]


def test_parse_actions_case_insensitive_tag():
    assert _kit.parse_actions("<Action>WAIT</Action>") == ["WAIT"]


def test_parse_actions_action_content_spanning_newlines():
    response = "<action>move(0,\n  5)</action>"
    assert _kit.parse_actions(response) == ["move(0,\n  5)"]


def test_parse_actions_rejects_legacy_format():
    # Old `ACTION: click(left)` syntax must no longer be recognized — the
    # switch is a clean cut, not a dual-format shim.
    assert _kit.parse_actions("ACTION: click(left)") == ["WAIT"]


def test_parse_actions_empty_tag_falls_back_to_wait():
    assert _kit.parse_actions("<action></action>") == ["WAIT"]


def test_parse_actions_whitespace_only_tag_falls_back_to_wait():
    # Whitespace-only content must not slip through as an empty action atom.
    assert _kit.parse_actions("<action>   </action>") == ["WAIT"]


def test_parse_actions_skips_empty_blocks_between_real_actions():
    response = "<action>press(w)</action><action>   </action><action>click(left)</action>"
    assert _kit.parse_actions(response) == ["press(w)", "click(left)"]


def test_parse_actions_signal_trails_real_actions():
    # Regression: model emits a batch of real actions then `<action>DONE</action>`
    # as "I think the task is now complete". Previously we short-circuited on the
    # first signal and dropped every preceding action. Expect: real actions pass
    # through in order, signal becomes the terminal atom; runner's done-handling
    # breaks the loop when env.step("DONE") fires.
    response = (
        "<action>move(-80, 30)</action>"
        "<action>click(left)</action>"
        "<action>move(60, -40)</action>"
        "<action>DONE</action>"
    )
    assert _kit.parse_actions(response) == [
        "move(-80, 30)",
        "click(left)",
        "move(60, -40)",
        "DONE",
    ]


def test_encode_observation():
    screenshot_png = _make_png_bytes()
    obs_dict = {"screenshot_png": screenshot_png, "status": {}}

    encoded = _kit.encode_observation(obs_dict)

    assert isinstance(encoded, list)
    assert len(encoded) == 1

    (image_block,) = encoded
    assert image_block["type"] == "image_url"
    data_url = image_block["image_url"]["url"]
    assert data_url.startswith("data:image/png;base64,")
    assert image_block["image_url"]["detail"] == "high"

    encoded_bytes = data_url.split(",", 1)[1]
    assert base64.b64decode(encoded_bytes) == screenshot_png


def test_system_prompt_contains_key_elements():
    prompt = _kit.system_prompt.lower()
    assert "<action>" in prompt
    assert "</action>" in prompt
    assert "done" in prompt
    assert "fail" in prompt
    assert "move" in prompt
    assert "click" in prompt
    assert "press" in prompt
    # Thinking structure must NOT be prescribed (no `thought:` label etc.).
    assert "thought:" not in prompt


def test_get_result_subdir():
    task = MinecraftTask(
        task_id="collect_wood_zero",
        instruction="Collect wood.",
        task_type="survival",
        difficulty="zero",
        task_config={},
    )
    assert _kit.get_result_subdir(task) == "survival/collect_wood_zero"


def test_reviser_guidance_mentions_action_format_and_signals():
    text = _kit.reviser_guidance
    lower = text.lower()
    assert "<action>" in lower
    assert "done" in lower
    assert "fail" in lower
    assert "wait" in lower


def test_load_saved_observation_returns_only_screenshot(tmp_path):
    # status.json is developer-only debug data — it must NEVER reach the
    # reviser VLM, even when a status file is present on disk.
    step_dir = tmp_path / "step_1_20260420@120000000000"
    step_dir.mkdir()
    (step_dir / "screenshot.png").write_bytes(_make_png_bytes())
    (step_dir / "status.json").write_text(
        '{"inventory": {"dirt": 3}, "gui_open": false}',
        encoding="utf-8",
    )

    blocks = _kit.load_saved_observation(step_dir)

    assert len(blocks) == 1
    assert blocks[0]["type"] == "image_url"
    assert blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")
    serialized = json.dumps(blocks)
    assert "dirt" not in serialized
    assert "gui_open" not in serialized
    assert "status" not in serialized.lower()


def test_load_saved_observation_missing_screenshot_returns_empty(tmp_path):
    # No screenshot → empty list; analyzer will degrade to text-only
    # reasoning instead of crashing. status.json alone is not a substitute.
    step_dir = tmp_path / "step_missing"
    step_dir.mkdir()
    (step_dir / "status.json").write_text(
        '{"inventory": {"dirt": 3}}', encoding="utf-8"
    )
    assert _kit.load_saved_observation(step_dir) == []
