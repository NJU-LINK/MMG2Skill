from __future__ import annotations

"""Tests for the Minecraft DSL action expander."""

from anything2skill.benchmarks.minecraft.env_wrapper import _expand_dsl_action


def test_expand_single_atom():
    assert _expand_dsl_action("click(left)", 100) == ["click(left)"]


def test_expand_atom_repeat():
    assert _expand_dsl_action("click(left) * 5", 100) == ["click(left)"] * 5


def test_expand_compound_repeat():
    assert _expand_dsl_action("move(1, 2) and click(left) * 3", 100) == [
        "move(1, 2) and click(left)"
    ] * 3


def test_expand_repeat_clamped(caplog):
    result = _expand_dsl_action("press(w) * 200", 100)
    assert len(result) == 100
    assert all(r == "press(w)" for r in result)


def test_expand_zero_repeat():
    assert _expand_dsl_action("press(w) * 0", 100) == ["no_op"]


def test_expand_negative_repeat():
    assert _expand_dsl_action("press(w) * -3", 100) == ["no_op"]


def test_expand_empty():
    assert _expand_dsl_action("", 100) == ["no_op"]


def test_expand_whitespace():
    assert _expand_dsl_action("   ", 100) == ["no_op"]


def test_expand_no_repeat_falls_through():
    assert _expand_dsl_action("press(w, space)", 100) == ["press(w, space)"]


def test_expand_respects_custom_cap():
    result = _expand_dsl_action("click(left) * 50", 10)
    assert len(result) == 10
