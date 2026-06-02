"""Tests for agent core: action parsing, state management."""

import logging

import pytest

from anything2skill.benchmarks.osworld.kit import OSWorldKit

_kit = OSWorldKit()
parse_actions = _kit.parse_actions
from anything2skill.agent.message_builder import MessageBuilder
from anything2skill.agent.state import AgentState
from anything2skill.parser.data_types import Skills, TutorialMaterial
from anything2skill.runner import _effective_max_attempts


class TestActionParser:
    def test_parse_special_wait(self):
        assert parse_actions("```WAIT```") == ["WAIT"]

    def test_parse_special_done(self):
        assert parse_actions("```DONE```") == ["DONE"]

    def test_parse_special_fail(self):
        assert parse_actions("```FAIL```") == ["FAIL"]

    def test_parse_special_fail_with_reason(self):
        assert parse_actions("```FAIL\nnot possible\n```") == ["FAIL"]

    def test_parse_special_done_with_reason(self):
        assert parse_actions("```DONE\ncompleted\n```") == ["DONE"]

    def test_parse_python_code(self):
        response = 'Some thinking...\n```python\nimport pyautogui\npyautogui.click(100, 200)\n```'
        actions = parse_actions(response)
        assert len(actions) == 1
        assert "pyautogui.click(100, 200)" in actions[0]

    def test_parse_code_with_trailing_done(self):
        response = '```python\nimport pyautogui\npyautogui.click(100, 200)\nDONE\n```'
        actions = parse_actions(response)
        assert len(actions) == 2
        assert "pyautogui.click(100, 200)" in actions[0]
        assert actions[1] == "DONE"

    def test_parse_no_code_blocks(self):
        response = "Just some random text without code blocks"
        actions = parse_actions(response)
        assert actions == []

    def test_parse_bare_special(self):
        assert parse_actions("DONE") == ["DONE"]
        assert parse_actions("WAIT") == ["WAIT"]
        assert parse_actions("FAIL") == ["FAIL"]


class TestAgentState:
    def test_initial_state(self):
        state = AgentState()
        assert len(state.history) == 0

    def test_record(self):
        state = AgentState()
        obs_content = [{"type": "text", "text": "obs"}]
        state.record(obs_content, "thinking...", "pyautogui.click(1,1)")
        assert len(state.history) == 1
        assert state.history[0]["obs_content"] == obs_content
        assert state.history[0]["response"] == "thinking..."
        assert state.history[0]["action"] == "pyautogui.click(1,1)"

    def test_recent_history(self):
        state = AgentState()
        for i in range(5):
            obs = [{"type": "text", "text": f"obs_{i}"}]
            state.record(obs, f"response_{i}", f"action_{i}")

        history = state.get_recent_history(3)
        assert len(history) == 3
        assert history[0]["response"] == "response_2"
        assert history[-1]["response"] == "response_4"

    def test_recent_history_empty(self):
        state = AgentState()
        assert state.get_recent_history(3) == []

    def test_recent_history_fewer_than_max(self):
        state = AgentState()
        obs = [{"type": "text", "text": "obs"}]
        state.record(obs, "r", "a")
        history = state.get_recent_history(5)
        assert len(history) == 1


def _flatten_user_text(messages: list[dict]) -> list[str]:
    """Concatenate all user-turn text blocks into a list, in turn order."""
    result = []
    for m in messages:
        if m["role"] != "user":
            continue
        chunks = []
        for blk in m["content"]:
            if blk.get("type") == "text":
                chunks.append(blk["text"])
        result.append("\n".join(chunks))
    return result


class TestVanillaTutorialMessages:
    """build_vanilla_tutorial_messages should inject the raw tutorial in
    the first user turn only and never reach for skill-extraction artifacts.
    """

    def _make_tutorial(self, image_paths=None) -> TutorialMaterial:
        return TutorialMaterial(
            task_id="t1",
            instruction="open terminal",
            content_type="html",
            body="<html>step 1: right-click desktop</html>",
            image_paths=list(image_paths or []),
        )

    def test_first_turn_contains_tutorial_block(self):
        mb = MessageBuilder(_kit)
        tut = self._make_tutorial()
        obs = [{"type": "text", "text": "<screenshot>"}]
        messages = mb.build_vanilla_tutorial_messages(
            tutorial=tut,
            max_images=None,
            obs_content=obs,
            history=[],
            instruction="open terminal",
        )

        assert messages[0]["role"] == "system"
        sys_text = messages[0]["content"][0]["text"]
        assert "RAW source material" in sys_text  # distinguishing phrase

        first_user_text = _flatten_user_text(messages)[0]
        assert "## Reference Tutorial" in first_user_text
        assert "right-click desktop" in first_user_text
        assert "## Reference Skills" not in first_user_text

    def test_subsequent_turns_skip_tutorial(self):
        mb = MessageBuilder(_kit)
        tut = self._make_tutorial()
        obs = [{"type": "text", "text": "<screenshot>"}]
        history = [
            {"obs_content": [{"type": "text", "text": "<old>"}],
             "response": "thinking", "action": "click"},
            {"obs_content": [{"type": "text", "text": "<old2>"}],
             "response": "thinking2", "action": "click2"},
        ]
        messages = mb.build_vanilla_tutorial_messages(
            tutorial=tut,
            max_images=None,
            obs_content=obs,
            history=history,
            instruction="open terminal",
        )
        user_texts = _flatten_user_text(messages)
        assert len(user_texts) >= 2
        assert "## Reference Tutorial" in user_texts[0]
        for later in user_texts[1:]:
            assert "## Reference Tutorial" not in later
            assert "right-click desktop" not in later

    def test_tutorial_none_yields_no_block(self):
        mb = MessageBuilder(_kit)
        obs = [{"type": "text", "text": "<screenshot>"}]
        messages = mb.build_vanilla_tutorial_messages(
            tutorial=None,
            max_images=None,
            obs_content=obs,
            history=[],
            instruction="open terminal",
        )
        first_user_text = _flatten_user_text(messages)[0]
        assert "## Reference Tutorial" not in first_user_text
        assert "## Reference Skills" not in first_user_text

    def test_max_images_caps_image_blocks(self, tmp_path, monkeypatch):
        # Three fake png paths; encode is monkey-patched so we don't need real PNGs.
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n")
            paths.append(str(p))

        # Stub encode_image_file in the location MessageBuilder imports it from.
        from anything2skill.vlm import client as vlm_client

        monkeypatch.setattr(
            vlm_client, "encode_image_file",
            lambda p: f"data:image/png;base64,FAKE_{p}",
        )

        mb = MessageBuilder(_kit)
        tut = self._make_tutorial(image_paths=paths)
        obs = [{"type": "text", "text": "<screenshot>"}]
        messages = mb.build_vanilla_tutorial_messages(
            tutorial=tut,
            max_images=1,
            obs_content=obs,
            history=[],
            instruction="open terminal",
        )
        # Count image_url blocks in the first user turn.
        first_user_blocks = messages[1]["content"]
        image_blocks = [b for b in first_user_blocks if b.get("type") == "image_url"]
        assert len(image_blocks) == 1


class TestEffectiveMaxAttempts:
    def test_vanilla_tutorial_clamps_to_one(self, caplog):
        with caplog.at_level(logging.WARNING, logger="anything2skill"):
            n = _effective_max_attempts({"max_attempts": 3}, "vanilla_tutorial")
        assert n == 1
        assert any("vanilla_tutorial" in r.getMessage() for r in caplog.records)

    def test_vanilla_clamps_to_one(self):
        assert _effective_max_attempts({"max_attempts": 5}, "vanilla") == 1

    def test_simple_passes_through(self):
        assert _effective_max_attempts({"max_attempts": 4}, "simple") == 4

    def test_phased_passes_through(self):
        assert _effective_max_attempts({"max_attempts": 2}, "phased") == 2

    def test_default_floor_is_one(self):
        assert _effective_max_attempts({}, "vanilla_tutorial") == 1
        assert _effective_max_attempts({"max_attempts": 0}, "simple") == 1
