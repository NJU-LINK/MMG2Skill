"""Doudizhu BenchmarkKit."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from anything2skill.benchmark_kit import BenchmarkKit, TaskDescriptor
from anything2skill.benchmarks.doudizhu import prompts
from anything2skill.benchmarks.registry import register_kit
from anything2skill.env_base import EnvironmentInterface

logger = logging.getLogger("anything2skill.benchmarks.doudizhu")


@dataclass
class DoudizhuTask(TaskDescriptor):
    """Doudizhu-specific task descriptor."""
    game_index: int = 0
    seed: int | None = None


@register_kit("doudizhu")
class DoudizhuKit(BenchmarkKit):
    """BenchmarkKit for Doudizhu (斗地主)."""

    def encode_observation(self, obs: dict) -> list[dict]:
        hand = obs.get("hand", "")
        role = obs.get("role", "unknown")
        recent_actions = obs.get("recent_actions", [])
        num_cards_left = obs.get("num_cards_left", [])
        legal_actions = obs.get("legal_actions", [])

        action_lines = []
        for entry in recent_actions[-10:]:
            action_lines.append(f"  {entry['player']}: {entry['action']}")
        actions_text = "\n".join(action_lines) if action_lines else "  (none yet)"

        cards_left_text = ", ".join(
            f"player {i}: {n}" for i, n in enumerate(num_cards_left)
        ) if num_cards_left else "unknown"

        text = (
            f"**Your Role**: {role}\n"
            f"**Your Hand**: {hand}\n"
            f"**Cards Remaining**: {cards_left_text}\n"
            f"**Recent Actions**:\n{actions_text}\n"
            f"**Legal Actions**: {', '.join(str(a) for a in legal_actions)}"
        )
        return [{"type": "text", "text": text}]

    @property
    def system_prompt(self) -> str:
        return prompts.SYSTEM_PROMPT.strip()

    @property
    def bridging_text(self) -> str:
        return prompts.BRIDGING_TEXT.strip()

    @property
    def reflection_guidance(self) -> str:
        return prompts.REFLECTION_GUIDANCE.strip()

    @property
    def planner_guidance(self) -> str:
        return prompts.PLANNER_GUIDANCE.strip()

    @property
    def skill_extraction_guidance(self) -> str:
        return prompts.SKILL_EXTRACTION_GUIDANCE.strip()

    @property
    def supports_skill_images(self) -> bool:
        return False

    def parse_actions(self, response: str) -> list[str]:
        stripped = response.strip().upper()
        if stripped in ("DONE", "FAIL"):
            return [stripped]

        # Try **Action**: first, then plain Action:
        for pat in (r"<action>(\S+)</action>", r"\*\*Action\*\*:\s*(\S+)", r"(?<!\w)Action:\s*(\S+)"):
            match = re.search(pat, response, re.IGNORECASE)
            if match:
                action = match.group(1).strip()
                if action.upper() in ("DONE", "FAIL"):
                    return [action.upper()]
                return [action]

        # Fallback: try to find "pass" in the response
        if re.search(r"\bpass\b", response, re.IGNORECASE):
            return ["pass"]

        logger.warning("Could not parse action from response: %s", response[:120])
        return ["__NOPARSE__"]

    def save_observation(self, obs: dict, path: Path) -> None:
        import json
        with open(path / "observation.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in obs.items() if k != "raw_state"}, f, indent=2, default=str)

    def load_saved_observation(self, step_dir: Path) -> list[dict]:
        import json
        obs_path = step_dir / "observation.json"
        if not obs_path.is_file():
            return []
        with open(obs_path, "r", encoding="utf-8") as f:
            obs = json.load(f)
        return self.encode_observation(obs)

    @property
    def reviser_guidance(self) -> str:
        return prompts.REVISER_GUIDANCE.strip()

    def create_env(self, env_cfg: dict) -> EnvironmentInterface:
        from anything2skill.benchmarks.doudizhu.env_wrapper import DoudizhuEnvWrapper
        return DoudizhuEnvWrapper(
            seed=env_cfg.get("seed"),
            opponent_type=env_cfg.get("opponent_type", "auto"),
        )

    def collect_tasks(self, cfg: dict) -> list[TaskDescriptor]:
        tasks_cfg = cfg.get("tasks", {})
        task_file = tasks_cfg.get("task_file")
        base_id = "doudizhu"
        instruction = (
            "Play Doudizhu (斗地主) as the Landlord against two Peasant opponents. "
            "Use strategic card play to empty your hand first and win the game."
        )

        if task_file:
            import json as _json
            with open(task_file, "r", encoding="utf-8") as f:
                entries = _json.load(f)
            from anything2skill.benchmarks.rlcard_common import expand_seeds
            tasks = []
            idx = 0
            for entry in entries:
                if entry.get("task_name") != base_id:
                    continue
                for seed in expand_seeds(entry.get("seed_start", 0), entry.get("num_seeds", 0)):
                    idx += 1
                    tasks.append(DoudizhuTask(
                        task_id=base_id,
                        instruction=instruction,
                        game_index=idx,
                        seed=seed,
                    ))
            return tasks

        seed = tasks_cfg.get("seed")
        num_games = tasks_cfg.get("num_games", 1)
        tasks = []
        for i in range(1, num_games + 1):
            tasks.append(DoudizhuTask(
                task_id=base_id,
                instruction=instruction,
                game_index=i,
                seed=(seed + i) if seed is not None else None,
            ))
        return tasks

    def get_result_subdir(self, task: TaskDescriptor) -> str:
        if isinstance(task, DoudizhuTask):
            return f"{task.task_id}/{task.game_index:03d}"
        return task.task_id
