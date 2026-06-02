"""Shared agent factory for all benchmarks.

Extracts the common _create_agent() logic that was duplicated across
OSWorld's runner.py and runner_multienv.py.
"""

from __future__ import annotations

from anything2skill.agent.base import BaseAgent
from anything2skill.benchmark_kit import BenchmarkKit
from anything2skill.parser.data_types import Skills, TutorialMaterial
from anything2skill.vlm.client import VLMClient


def create_agent(
    agent_mode: str,
    vlm: VLMClient,
    skills: Skills,
    kit: BenchmarkKit,
    agent_cfg: dict,
    result_dir: str | None = None,
    *,
    tutorial: TutorialMaterial | None = None,
    max_images: int | None = None,
) -> BaseAgent:
    """Create a SimpleAgent, PhasedAgent, VanillaAgent or VanillaTutorialAgent.

    Args:
        agent_mode: "simple", "phased", "vanilla", or "vanilla_tutorial".
        vlm: VLM client instance.
        skills: Extracted skills for the current task (empty Skills for
            vanilla / vanilla_tutorial).
        kit: BenchmarkKit instance.
        agent_cfg: Agent config dict (history_window, max_tokens, etc.).
        result_dir: Optional result directory for agent-level trajectory.
        tutorial: Raw tutorial material; only consumed by ``vanilla_tutorial``.
        max_images: Cap on tutorial images for ``vanilla_tutorial``; ``None``
            means no cap. Reuses ``skills.max_images`` from config.

    Returns:
        A BaseAgent instance.
    """
    llm_params = {
        "max_tokens": agent_cfg.get("max_tokens", 1500),
    }
    if agent_cfg.get("temperature") is not None:
        llm_params["temperature"] = agent_cfg["temperature"]

    if agent_mode == "phased":
        from anything2skill.agent.phased_agent import PhasedAgent
        from anything2skill.agent.planner import SoftPlanner

        planner = SoftPlanner(vlm, kit, llm_params=llm_params)
        return PhasedAgent(
            vlm=vlm,
            skills=skills,
            kit=kit,
            planner=planner,
            history_window=agent_cfg.get("history_window", 3),
            result_dir=result_dir,
            llm_params=llm_params,
        )
    elif agent_mode == "vanilla":
        from anything2skill.agent.vanilla_agent import VanillaAgent

        return VanillaAgent(
            vlm=vlm,
            skills=skills,
            kit=kit,
            history_window=agent_cfg.get("history_window", 3),
            result_dir=result_dir,
            llm_params=llm_params,
        )
    elif agent_mode == "vanilla_tutorial":
        from anything2skill.agent.vanilla_tutorial_agent import VanillaTutorialAgent

        return VanillaTutorialAgent(
            vlm=vlm,
            skills=skills,
            kit=kit,
            tutorial=tutorial,
            max_images=max_images,
            history_window=agent_cfg.get("history_window", 3),
            result_dir=result_dir,
            llm_params=llm_params,
        )
    else:
        from anything2skill.agent.simple_agent import SimpleAgent

        return SimpleAgent(
            vlm=vlm,
            skills=skills,
            kit=kit,
            history_window=agent_cfg.get("history_window", 3),
            result_dir=result_dir,
            llm_params=llm_params,
        )
