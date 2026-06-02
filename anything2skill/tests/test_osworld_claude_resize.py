from __future__ import annotations

import io

from PIL import Image

from anything2skill.benchmarks.osworld.claude_resize import (
    CLAUDE_BASE_SIZE,
    CLAUDE_MODEL_IDS,
    is_claude_model,
    resize_screenshot_for_claude,
    rewrite_claude_pixel_coordinates,
)


def _make_png(size: tuple[int, int], color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# is_claude_model
# ---------------------------------------------------------------------------


def test_is_claude_model_substring_match():
    assert is_claude_model("claude-sonnet-4-6")
    assert is_claude_model("Claude-Opus-4-7")
    assert is_claude_model("anthropic/claude-haiku-4-5")


def test_is_claude_model_rejects_others():
    assert not is_claude_model("gpt-4o")
    assert not is_claude_model("kimi2.6")
    assert not is_claude_model("")
    assert not is_claude_model(None)  # type: ignore[arg-type]


def test_is_claude_model_id_set_overrides_substring():
    # Patch the module-level frozenset to simulate an encrypted alias being onboarded.
    from anything2skill.benchmarks.osworld import claude_resize as cr

    original = cr.CLAUDE_MODEL_IDS
    try:
        cr.CLAUDE_MODEL_IDS = frozenset({"x-relay-3"})
        assert is_claude_model("x-relay-3")
        assert is_claude_model("X-Relay-3")
        assert not is_claude_model("x-relay-9")
    finally:
        cr.CLAUDE_MODEL_IDS = original


def test_claude_model_ids_is_frozenset_for_immutability():
    assert isinstance(CLAUDE_MODEL_IDS, frozenset)


# ---------------------------------------------------------------------------
# resize_screenshot_for_claude
# ---------------------------------------------------------------------------


def test_resize_downscales_to_base_size():
    src = _make_png((1920, 1080))
    out = resize_screenshot_for_claude(src)
    img = Image.open(io.BytesIO(out))
    assert img.size == CLAUDE_BASE_SIZE
    assert img.format == "PNG"


def test_resize_upscales_smaller_input():
    src = _make_png((640, 360))
    out = resize_screenshot_for_claude(src)
    assert Image.open(io.BytesIO(out)).size == CLAUDE_BASE_SIZE


def test_resize_noop_when_already_base_size():
    src = _make_png(CLAUDE_BASE_SIZE)
    out = resize_screenshot_for_claude(src)
    assert Image.open(io.BytesIO(out)).size == CLAUDE_BASE_SIZE


# ---------------------------------------------------------------------------
# rewrite_claude_pixel_coordinates
# ---------------------------------------------------------------------------


def test_rewrite_scales_click_to_actual_screen():
    code = "import pyautogui\npyautogui.click(640, 360)"
    rewritten = rewrite_claude_pixel_coordinates(code, (1920, 1080))
    assert "pyautogui.click(960, 540)" in rewritten


def test_rewrite_scales_drag_and_move():
    code = (
        "import pyautogui\n"
        "pyautogui.moveTo(100, 200)\n"
        "pyautogui.dragTo(300, 400)\n"
    )
    rewritten = rewrite_claude_pixel_coordinates(code, (1920, 1080))
    # 100 * 1920/1280 = 150;  200 * 1080/720 = 300
    # 300 * 1920/1280 = 450;  400 * 1080/720 = 600
    assert "pyautogui.moveTo(150, 300)" in rewritten
    assert "pyautogui.dragTo(450, 600)" in rewritten


def test_rewrite_keyword_arguments():
    code = "import pyautogui\npyautogui.click(x=640, y=360)"
    rewritten = rewrite_claude_pixel_coordinates(code, (1920, 1080))
    assert "x=960" in rewritten and "y=540" in rewritten


def test_rewrite_skips_when_screen_size_matches_base():
    code = "import pyautogui\npyautogui.click(640, 360)"
    assert rewrite_claude_pixel_coordinates(code, CLAUDE_BASE_SIZE) == code


def test_rewrite_does_not_touch_variables():
    code = (
        "import pyautogui\n"
        "x, y = 10, 20\n"
        "pyautogui.click(x, y)\n"
    )
    rewritten = rewrite_claude_pixel_coordinates(code, (1920, 1080))
    assert "pyautogui.click(x, y)" in rewritten


def test_rewrite_scales_float_coordinates():
    # 640.5 * 1920/1280 = 960.75 → int trunc = 960
    # 360.5 * 1080/720  = 540.75 → int trunc = 540
    code = "import pyautogui\npyautogui.click(640.5, 360.5)"
    rewritten = rewrite_claude_pixel_coordinates(code, (1920, 1080))
    assert "pyautogui.click(960, 540)" in rewritten


def test_rewrite_passthrough_on_syntax_error():
    code = "this is not python"
    assert rewrite_claude_pixel_coordinates(code, (1920, 1080)) == code


def test_rewrite_passthrough_on_invalid_screen_size():
    code = "import pyautogui\npyautogui.click(640, 360)"
    assert rewrite_claude_pixel_coordinates(code, "invalid") == code  # type: ignore[arg-type]


def test_rewrite_ignores_non_pyautogui_calls():
    code = "click(640, 360)"
    assert rewrite_claude_pixel_coordinates(code, (1920, 1080)) == code


def test_rewrite_handles_zero_coordinates():
    code = "import pyautogui\npyautogui.click(0, 0)"
    # Multiplying by anything keeps zero — rewriter returns input unchanged.
    rewritten = rewrite_claude_pixel_coordinates(code, (1920, 1080))
    assert "pyautogui.click(0, 0)" in rewritten


def test_rewrite_truncates_like_official():
    # 1 * 1920/1280 = 1.5; OSWorld official truncates → 1, rounding would give 2.
    code = "import pyautogui\npyautogui.click(1, 1)"
    rewritten = rewrite_claude_pixel_coordinates(code, (1920, 1080))
    assert "pyautogui.click(1, 1)" in rewritten


def test_rewrite_scales_scroll_with_coordinates():
    # pyautogui.scroll(clicks, x, y) — clicks is wheel amount, x/y at indices 1/2.
    code = "import pyautogui\npyautogui.scroll(-3, 640, 360)"
    rewritten = rewrite_claude_pixel_coordinates(code, (1920, 1080))
    assert "pyautogui.scroll(-3, 960, 540)" in rewritten


def test_rewrite_scales_hscroll_with_coordinates():
    code = "import pyautogui\npyautogui.hscroll(5, 100, 200)"
    rewritten = rewrite_claude_pixel_coordinates(code, (1920, 1080))
    assert "pyautogui.hscroll(5, 150, 300)" in rewritten


def test_rewrite_leaves_scroll_clicks_amount_alone():
    # Without coords the scroll() call is left untouched — clicks must not be scaled.
    code = "import pyautogui\npyautogui.scroll(-3)"
    rewritten = rewrite_claude_pixel_coordinates(code, (1920, 1080))
    assert "pyautogui.scroll(-3)" in rewritten


# ---------------------------------------------------------------------------
# OSWorldKit.encode_observation seam
# ---------------------------------------------------------------------------


def _decode_data_url_to_size(blocks: list[dict]) -> tuple[int, int]:
    import base64

    url = blocks[0]["image_url"]["url"]
    b64 = url.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    return img.size


def test_kit_resizes_observation_when_model_is_claude():
    from anything2skill.benchmarks.osworld.kit import OSWorldKit

    kit = OSWorldKit(env_cfg={"agent_model": "claude-sonnet-4-6"})
    obs = {"screenshot": _make_png((1920, 1080))}

    blocks = kit.encode_observation(obs)
    assert _decode_data_url_to_size(blocks) == CLAUDE_BASE_SIZE


def test_kit_passes_observation_through_for_non_claude_model():
    from anything2skill.benchmarks.osworld.kit import OSWorldKit

    kit = OSWorldKit(env_cfg={"agent_model": "gpt-4o"})
    obs = {"screenshot": _make_png((1920, 1080))}

    blocks = kit.encode_observation(obs)
    assert _decode_data_url_to_size(blocks) == (1920, 1080)
