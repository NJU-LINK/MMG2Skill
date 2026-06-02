"""Tests for card game benchmarks (3 RLCard games kept for paper)."""

from __future__ import annotations

import pytest


# ══════════════════════════════════════════════════════════════════════
# Mahjong
# ══════════════════════════════════════════════════════════════════════


def test_mahjong_kit_registers():
    from anything2skill.benchmarks.registry import get_kit, list_kits

    kit = get_kit("mahjong")
    assert kit is not None
    assert "mahjong" in list_kits()


def test_mahjong_kit_properties():
    from anything2skill.benchmarks.registry import get_kit

    kit = get_kit("mahjong")
    assert len(kit.system_prompt) > 0
    assert kit.supports_skill_images is False


def test_mahjong_parse_actions():
    from anything2skill.benchmarks.registry import get_kit

    kit = get_kit("mahjong")
    assert kit.parse_actions("**Action**: bamboo-1") == ["bamboo-1"]
    assert kit.parse_actions("**Action**: pong") == ["pong"]
    assert kit.parse_actions("**Action**: stand") == ["stand"]
    assert kit.parse_actions("DONE") == ["DONE"]


def test_mahjong_collect_tasks():
    from anything2skill.benchmarks.registry import get_kit

    kit = get_kit("mahjong")
    tasks = kit.collect_tasks({"tasks": {}})
    assert len(tasks) == 1
    assert tasks[0].task_id == "mahjong"


def test_mahjong_env_wrapper():
    pytest.importorskip("rlcard", reason="rlcard not installed")

    from anything2skill.benchmarks.mahjong.env_wrapper import MahjongEnvWrapper
    from anything2skill.benchmarks.mahjong.kit import MahjongTask

    env = MahjongEnvWrapper(seed=42, opponent_type="random")
    task = MahjongTask(task_id="test", instruction="play mahjong")
    obs = env.reset(task)

    assert obs["game"] == "mahjong"
    assert "hand" in obs
    assert "legal_actions" in obs
    assert obs["num_players"] == 4

    done = False
    steps = 0
    while not done and steps < 100:
        legal = obs.get("legal_actions", [])
        if not legal:
            break
        obs, reward, done, info = env.step(legal[0])
        steps += 1

    score = env.evaluate()
    assert 0.0 <= score <= 1.0
    env.close()


# ══════════════════════════════════════════════════════════════════════
# Doudizhu
# ══════════════════════════════════════════════════════════════════════


def test_doudizhu_kit_registers():
    from anything2skill.benchmarks.registry import get_kit, list_kits

    kit = get_kit("doudizhu")
    assert kit is not None
    assert "doudizhu" in list_kits()


def test_doudizhu_kit_properties():
    from anything2skill.benchmarks.registry import get_kit

    kit = get_kit("doudizhu")
    assert len(kit.system_prompt) > 0
    assert kit.supports_skill_images is False


def test_doudizhu_parse_actions():
    from anything2skill.benchmarks.registry import get_kit

    kit = get_kit("doudizhu")
    assert kit.parse_actions("**Action**: pass") == ["pass"]
    assert kit.parse_actions("**Action**: 3") == ["3"]
    assert kit.parse_actions("DONE") == ["DONE"]


def test_doudizhu_collect_tasks():
    from anything2skill.benchmarks.registry import get_kit

    kit = get_kit("doudizhu")
    tasks = kit.collect_tasks({"tasks": {}})
    assert len(tasks) == 1
    assert tasks[0].task_id == "doudizhu"


def test_doudizhu_env_wrapper():
    pytest.importorskip("rlcard", reason="rlcard not installed")

    from anything2skill.benchmarks.doudizhu.env_wrapper import DoudizhuEnvWrapper
    from anything2skill.benchmarks.doudizhu.kit import DoudizhuTask

    env = DoudizhuEnvWrapper(seed=42, opponent_type="random")
    task = DoudizhuTask(task_id="test", instruction="play doudizhu")
    obs = env.reset(task)

    assert obs["game"] == "doudizhu"
    assert "hand" in obs
    assert "legal_actions" in obs
    assert "role" in obs

    done = False
    steps = 0
    while not done and steps < 50:
        legal = obs.get("legal_actions", [])
        if not legal:
            break
        obs, reward, done, info = env.step(legal[0])
        steps += 1

    score = env.evaluate()
    assert 0.0 <= score <= 1.0
    env.close()


# ══════════════════════════════════════════════════════════════════════
# No-Limit Hold'em
# ══════════════════════════════════════════════════════════════════════


def test_nolimit_holdem_kit_registers():
    from anything2skill.benchmarks.registry import get_kit, list_kits

    kit = get_kit("nolimit_holdem")
    assert kit is not None
    assert "nolimit_holdem" in list_kits()


def test_nolimit_holdem_kit_properties():
    from anything2skill.benchmarks.registry import get_kit

    kit = get_kit("nolimit_holdem")
    assert len(kit.system_prompt) > 0
    assert kit.supports_skill_images is False


def test_nolimit_holdem_parse_actions():
    from anything2skill.benchmarks.registry import get_kit

    kit = get_kit("nolimit_holdem")
    assert kit.parse_actions("**Action**: fold") == ["fold"]
    assert kit.parse_actions("**Action**: check_call") == ["check_call"]
    assert kit.parse_actions("**Action**: all_in") == ["all_in"]
    assert kit.parse_actions("DONE") == ["DONE"]


def test_nolimit_holdem_collect_tasks():
    from anything2skill.benchmarks.registry import get_kit

    kit = get_kit("nolimit_holdem")
    tasks = kit.collect_tasks({"tasks": {}})
    assert len(tasks) == 1
    assert tasks[0].task_id == "nolimit_holdem"


def test_nolimit_holdem_env_wrapper():
    pytest.importorskip("rlcard", reason="rlcard not installed")

    from anything2skill.benchmarks.nolimit_holdem.env_wrapper import NolimitHoldemEnvWrapper
    from anything2skill.benchmarks.nolimit_holdem.kit import NolimitHoldemTask

    env = NolimitHoldemEnvWrapper(seed=42, opponent_type="random")
    task = NolimitHoldemTask(task_id="test", instruction="play no-limit holdem")
    obs = env.reset(task)

    assert obs["game"] == "nolimit_holdem"
    assert "hand" in obs
    assert "legal_actions" in obs

    done = False
    steps = 0
    while not done and steps < 30:
        legal = obs.get("legal_actions", [])
        if not legal:
            break
        obs, reward, done, info = env.step(legal[0])
        steps += 1

    score = env.evaluate()
    assert 0.0 <= score <= 1.0
    env.close()
