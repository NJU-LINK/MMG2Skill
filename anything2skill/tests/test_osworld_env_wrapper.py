from __future__ import annotations

from anything2skill.benchmarks.osworld.env_wrapper import OSWorldEnvWrapper


class _FakeEnv:
    def __init__(self):
        self.calls: list[tuple[object, float]] = []

    def step(self, action, pause):
        self.calls.append((action, pause))
        return {"screenshot": b""}, 1.0, False, {"existing": True}


def test_wait_action_preserves_signal_and_uses_wait_pause():
    fake_env = _FakeEnv()
    wrapper = OSWorldEnvWrapper(fake_env, wait_action_duration=20.0)

    _obs, _reward, _done, info = wrapper.step("WAIT", pause=2.0)

    assert fake_env.calls == [("WAIT", 10.0)]
    assert info["requested_action"] == "WAIT"
    assert info["executed_action"] == "WAIT"
    assert info["action_rewritten"] is False
    assert info["existing"] is True


def test_text_action_is_rewritten_when_enabled():
    fake_env = _FakeEnv()
    wrapper = OSWorldEnvWrapper(fake_env)
    action = "import pyautogui\npyautogui.write('ab')"

    _obs, _reward, _done, info = wrapper.step(action, pause=2.0)

    executed_action, pause = fake_env.calls[0]
    assert pause == 2.0
    assert executed_action != action
    assert "pyautogui.press('a')" in executed_action
    assert "pyautogui.press('b')" in executed_action
    assert info["requested_action"] == action
    assert info["executed_action"] == executed_action
    assert info["action_rewritten"] is True


def test_kimi_model_rewrites_normalized_coordinates():
    fake_env = _FakeEnv()
    wrapper = OSWorldEnvWrapper(
        fake_env,
        model_name="kimi2.6",
        screen_size=(1920, 1080),
    )
    action = "import pyautogui\npyautogui.click(0.5, 0.5)"

    _obs, _reward, _done, info = wrapper.step(action, pause=2.0)

    executed_action, _pause = fake_env.calls[0]
    assert "pyautogui.click(960, 540)" in executed_action
    assert info["kimi_coord_rewrite_enabled"] is True
    assert info["kimi_action_rewritten"] is True
    assert info["action_rewritten"] is True


def test_non_kimi_model_does_not_rewrite_normalized_coordinates():
    fake_env = _FakeEnv()
    wrapper = OSWorldEnvWrapper(
        fake_env,
        model_name="gpt-4o",
        screen_size=(1920, 1080),
    )
    action = "import pyautogui\npyautogui.click(0.5, 0.5)"

    _obs, _reward, _done, info = wrapper.step(action, pause=2.0)

    executed_action, _pause = fake_env.calls[0]
    assert executed_action == action
    assert info["kimi_coord_rewrite_enabled"] is False
    assert info["kimi_action_rewritten"] is False
    assert info["claude_coord_rewrite_enabled"] is False
    assert info["claude_action_rewritten"] is False
    assert info["action_rewritten"] is False


def test_claude_model_rescales_pixel_coordinates():
    fake_env = _FakeEnv()
    wrapper = OSWorldEnvWrapper(
        fake_env,
        model_name="claude-sonnet-4-6",
        screen_size=(1920, 1080),
    )
    action = "import pyautogui\npyautogui.click(640, 360)"

    _obs, _reward, _done, info = wrapper.step(action, pause=2.0)

    executed_action, _pause = fake_env.calls[0]
    assert "pyautogui.click(960, 540)" in executed_action
    assert info["claude_coord_rewrite_enabled"] is True
    assert info["claude_action_rewritten"] is True
    assert info["kimi_coord_rewrite_enabled"] is False
    assert info["action_rewritten"] is True


def test_non_claude_model_does_not_rescale_pixel_coordinates():
    fake_env = _FakeEnv()
    wrapper = OSWorldEnvWrapper(
        fake_env,
        model_name="gpt-4o",
        screen_size=(1920, 1080),
    )
    action = "import pyautogui\npyautogui.click(640, 360)"

    _obs, _reward, _done, info = wrapper.step(action, pause=2.0)

    executed_action, _pause = fake_env.calls[0]
    assert executed_action == action
    assert info["claude_coord_rewrite_enabled"] is False
    assert info["claude_action_rewritten"] is False
