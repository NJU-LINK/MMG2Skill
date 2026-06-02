from __future__ import annotations

import ast

from anything2skill.benchmarks.osworld.pyautogui_sanitizer import (
    rewrite_kimi_normalized_coordinates,
    rewrite_pyautogui_text_inputs,
)


def _extract_press_args(code: str) -> list[str]:
    tree = ast.parse(code)
    args: list[str] = []
    for node in tree.body:
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "pyautogui"
            and node.value.func.attr == "press"
        ):
            continue
        arg = node.value.args[0]
        assert isinstance(arg, ast.Constant)
        assert isinstance(arg.value, str)
        args.append(arg.value)
    return args


def test_rewrite_simple_typewrite_to_presses():
    code = "import pyautogui\npyautogui.typewrite('abc')"

    rewritten = rewrite_pyautogui_text_inputs(code)

    assert _extract_press_args(rewritten) == ["a", "b", "c"]
    assert "pyautogui.typewrite" not in rewritten


def test_rewrite_newline_to_enter_press():
    code = "import pyautogui\npyautogui.typewrite('a\\nb')"

    rewritten = rewrite_pyautogui_text_inputs(code)

    assert _extract_press_args(rewritten) == ["a", "enter", "b"]


def test_rewrite_quote_double_quote_and_backslash():
    code = "import pyautogui\npyautogui.write(\"'\\\"\\\\\")"

    rewritten = rewrite_pyautogui_text_inputs(code)

    assert _extract_press_args(rewritten) == ["'", '"', "\\"]


def test_fallback_rewrites_malformed_multiline_literal():
    code = "import pyautogui\npyautogui.typewrite('ab\ncd')"

    rewritten = rewrite_pyautogui_text_inputs(code)

    assert _extract_press_args(rewritten) == ["a", "b", "enter", "c", "d"]
    assert "pyautogui.typewrite" not in rewritten


def test_non_text_actions_remain_unchanged():
    code = "import pyautogui\npyautogui.click(1, 2)\npyautogui.hotkey('ctrl', 'c')"

    rewritten = rewrite_pyautogui_text_inputs(code)

    assert rewritten == code


def test_kimi_rewrite_projects_normalized_click_coordinates():
    code = "import pyautogui\npyautogui.click(0.5, 0.5)"

    rewritten = rewrite_kimi_normalized_coordinates(code, (1920, 1080))

    assert "pyautogui.click(960, 540)" in rewritten


def test_kimi_rewrite_leaves_absolute_coordinates_unchanged():
    code = "import pyautogui\npyautogui.click(960, 540)"

    rewritten = rewrite_kimi_normalized_coordinates(code, (1920, 1080))

    assert rewritten == code


def test_kimi_rewrite_projects_keyword_coordinates_and_preserves_args():
    code = "import pyautogui\npyautogui.moveTo(x=0.25, y=0.75, duration=0.2)"

    rewritten = rewrite_kimi_normalized_coordinates(code, (1920, 1080))

    assert "pyautogui.moveTo(x=480, y=810, duration=0.2)" in rewritten


def test_kimi_rewrite_ignores_non_target_or_dynamic_coordinates():
    code = (
        "import pyautogui\n"
        "pyautogui.scroll(-5)\n"
        "pyautogui.click(x, 0.5)\n"
        "pyautogui.click(0.5)"
    )

    rewritten = rewrite_kimi_normalized_coordinates(code, (1920, 1080))

    assert rewritten == code
