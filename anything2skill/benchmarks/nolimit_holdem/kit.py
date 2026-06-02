"""No-Limit Hold'em BenchmarkKit."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from anything2skill.benchmark_kit import BenchmarkKit, TaskDescriptor
from anything2skill.benchmarks.nolimit_holdem import prompts
from anything2skill.benchmarks.registry import register_kit
from anything2skill.env_base import EnvironmentInterface

logger = logging.getLogger("anything2skill.benchmarks.nolimit_holdem")


@dataclass
class NolimitHoldemTask(TaskDescriptor):
    """No-Limit Hold'em-specific task descriptor."""
    game_index: int = 0
    seed: int | None = None


@register_kit("nolimit_holdem")
class NolimitHoldemKit(BenchmarkKit):
    """BenchmarkKit for No-Limit Hold'em."""

    def encode_observation(self, obs: dict) -> list[dict]:
        hand = obs.get("hand", [])
        public_cards = obs.get("public_cards", [])
        round_name = obs.get("round", "Preflop")
        my_chips = obs.get("my_chips", 0)
        all_chips = obs.get("all_chips", [])
        pot = obs.get("pot", 0)
        stakes = obs.get("stakes", [])
        legal_actions = obs.get("legal_actions", [])

        hand_str = ", ".join(hand) if hand else "None"
        public_str = ", ".join(public_cards) if public_cards else "None (not yet dealt)"

        text = (
            f"**Your Hole Cards**: {hand_str}\n"
            f"**Community Cards**: {public_str}\n"
            f"**Current Round**: {round_name}\n"
            f"**Pot**: {pot}\n"
            f"**Your Chips Bet**: {my_chips}\n"
            f"**All Players' Chips Bet**: {all_chips}\n"
            f"**Remaining Stacks**: {stakes}\n"
            f"**Legal Actions**: {', '.join(legal_actions)}"
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

        for pat in (r"<action>([\w\-]+)</action>", r"\*\*Action\*\*:\s*([\w\-]+)", r"(?<!\w)Action:\s*([\w\-]+)"):
            match = re.search(pat, response, re.IGNORECASE)
            if match:
                action = match.group(1).strip()
                if action.upper() in ("DONE", "FAIL"):
                    return [action.upper()]
                return [action.lower()]

        lower = response.lower()
        for kw in ("all_in", "raise_pot", "raise_half_pot", "check_call", "fold"):
            if kw in lower:
                return [kw]

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
        from anything2skill.benchmarks.nolimit_holdem.env_wrapper import NolimitHoldemEnvWrapper
        return NolimitHoldemEnvWrapper(
            seed=env_cfg.get("seed"),
            opponent_type=env_cfg.get("opponent_type", "auto"),
        )

    def collect_tasks(self, cfg: dict) -> list[TaskDescriptor]:
        tasks_cfg = cfg.get("tasks", {})
        task_file = tasks_cfg.get("task_file")
        base_id = "nolimit_holdem"
        instruction = (
            "Play No-Limit Hold'em poker against an opponent. "
            "Use hand selection, bet sizing, position awareness, "
            "and bluffing to maximize your expected payoff (net chips, range -100 to +100)."
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
                    tasks.append(NolimitHoldemTask(
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
            tasks.append(NolimitHoldemTask(
                task_id=base_id,
                instruction=instruction,
                game_index=i,
                seed=(seed + i) if seed is not None else None,
            ))
        return tasks

    def get_result_subdir(self, task: TaskDescriptor) -> str:
        if isinstance(task, NolimitHoldemTask):
            return f"{task.task_id}/{task.game_index:03d}"
        return task.task_id
