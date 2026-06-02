"""Mahjong BenchmarkKit."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from anything2skill.benchmark_kit import BenchmarkKit, TaskDescriptor
from anything2skill.benchmarks.mahjong import prompts
from anything2skill.benchmarks.registry import register_kit
from anything2skill.env_base import EnvironmentInterface

logger = logging.getLogger("anything2skill.benchmarks.mahjong")


@dataclass
class MahjongTask(TaskDescriptor):
    """Mahjong-specific task descriptor."""
    game_index: int = 0
    seed: int | None = None


@register_kit("mahjong")
class MahjongKit(BenchmarkKit):
    """BenchmarkKit for Mahjong (4-player)."""

    def encode_observation(self, obs: dict) -> list[dict]:
        hand = obs.get("hand", [])
        table = obs.get("table", [])
        piles = obs.get("player_piles", {})
        legal_actions = obs.get("legal_actions", [])

        lines = [
            f"**Your Hand** ({len(hand)} tiles): {', '.join(hand)}",
            f"**Table** (discarded): {', '.join(table)}",
        ]
        for pid, pile_cards in piles.items():
            lines.append(f"**Player {pid} Claimed**: {', '.join(pile_cards) if pile_cards else '(none)'}")
        lines.append(f"**Legal Actions**: {', '.join(str(a) for a in legal_actions)}")

        return [{"type": "text", "text": "\n".join(lines)}]

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
                return [action]

        # Keyword fallbacks for claim actions
        lower = response.lower()
        for kw in ("pong", "chow", "gong", "stand"):
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
        from anything2skill.benchmarks.mahjong.env_wrapper import MahjongEnvWrapper
        return MahjongEnvWrapper(
            seed=env_cfg.get("seed"),
            opponent_type=env_cfg.get("opponent_type", "auto"),
        )

    def collect_tasks(self, cfg: dict) -> list[TaskDescriptor]:
        tasks_cfg = cfg.get("tasks", {})
        task_file = tasks_cfg.get("task_file")
        base_id = "mahjong"
        instruction = (
            "Play Mahjong (4 players). Maximize your expected payoff "
            "(+1 for your win, 0 for a draw, -1 when an opponent wins): "
            "complete a winning hand (4 sets + 1 pair) before opponents while "
            "avoiding discards that let them complete theirs."
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
                    tasks.append(MahjongTask(
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
            tasks.append(MahjongTask(
                task_id=base_id,
                instruction=instruction,
                game_index=i,
                seed=(seed + i) if seed is not None else None,
            ))
        return tasks

    def get_result_subdir(self, task: TaskDescriptor) -> str:
        if isinstance(task, MahjongTask):
            return f"{task.task_id}/{task.game_index:03d}"
        return task.task_id
