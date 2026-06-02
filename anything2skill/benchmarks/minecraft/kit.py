from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from anything2skill.benchmark_kit import BenchmarkKit, TaskDescriptor
from anything2skill.benchmarks.registry import register_kit
from anything2skill.env_base import EnvironmentInterface

from .prompts import (
    ACTION_SPACE,
    BRIDGING_TEXT,
    PLANNER_GUIDANCE,
    REFLECTION_GUIDANCE,
    REVISER_GUIDANCE,
    SKILL_EXTRACTION_GUIDANCE,
    SYSTEM_PROMPT,
)

_BEGINNERS_GUIDE_TUTORIAL_ID = "_beginners_guide"

logger = logging.getLogger("anything2skill.benchmarks.minecraft")

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_OPENHA_ROOT = os.path.join(_PROJECT_ROOT, "OpenHA")

_FRAMEWORK_SIGNALS = {"DONE", "FAIL", "WAIT"}


@dataclass
class MinecraftTask(TaskDescriptor):
    """Minecraft-specific task descriptor."""

    task_type: str = ""
    difficulty: str = "zero"
    task_config: dict = field(default_factory=dict)


@register_kit("minecraft")
class MinecraftKit(BenchmarkKit):
    """BenchmarkKit for the Minecraft benchmark."""

    def encode_observation(self, obs: dict) -> list[dict]:
        screenshot_b64 = base64.b64encode(obs["screenshot_png"]).decode("utf-8")
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_b64}",
                    "detail": "high",
                },
            },
        ]

    def parse_actions(self, response: str) -> list[str]:
        stripped = response.strip()
        if stripped.upper() in _FRAMEWORK_SIGNALS:
            return [stripped.upper()]

        action_matches = re.findall(
            r"<action>\s*(.*?)\s*</action>",
            response,
            re.DOTALL | re.IGNORECASE,
        )
        if not action_matches:
            logger.warning("Response missing <action> tag; returning WAIT")
            return ["WAIT"]

        actions: list[str] = []
        for match in action_matches:
            action = match.strip()
            if not action:
                continue
            upper = action.upper()
            if upper in _FRAMEWORK_SIGNALS:
                actions.append(upper)
            else:
                actions.append(action)

        return actions or ["WAIT"]

    def _action_space_text(self) -> str:
        max_repeat = int(self.env_cfg.get("max_repeat", 200))
        return ACTION_SPACE.format(max_repeat=max_repeat).strip()

    @property
    def system_prompt(self) -> str:
        render_size = self.env_cfg.get("render_size", [640, 360])
        w, h = int(render_size[0]), int(render_size[1])
        return SYSTEM_PROMPT.format(
            action_space=self._action_space_text(),
            width=w,
            height=h,
            cx=w // 2,
            cy=h // 2,
        ).strip()

    @property
    def bridging_text(self) -> str:
        return BRIDGING_TEXT.strip()

    @property
    def reflection_guidance(self) -> str:
        return REFLECTION_GUIDANCE.strip()

    @property
    def planner_guidance(self) -> str:
        return PLANNER_GUIDANCE.strip()

    @property
    def skill_extraction_guidance(self) -> str:
        return SKILL_EXTRACTION_GUIDANCE.format(
            action_space=self._action_space_text(),
        ).strip()

    @property
    def reviser_guidance(self) -> str:
        render_size = self.env_cfg.get("render_size", [640, 360])
        w, h = int(render_size[0]), int(render_size[1])
        return REVISER_GUIDANCE.format(width=w, height=h).strip()

    def tutorial_ids_for(self, task: TaskDescriptor) -> list[str]:
        # Per-task wiki page first (recipe / block specifics), shared
        # beginner's guide second (general GUI / control mechanics).
        return [task.task_id, _BEGINNERS_GUIDE_TUTORIAL_ID]

    def create_env(self, env_cfg: dict) -> EnvironmentInterface:
        if _OPENHA_ROOT not in sys.path:
            sys.path.insert(0, _OPENHA_ROOT)

        from .env_wrapper import MinecraftEnvWrapper

        return MinecraftEnvWrapper(env_cfg)

    def collect_tasks(self, cfg: dict) -> list[TaskDescriptor]:
        tasks_cfg = cfg.get("tasks", {})
        task_file = tasks_cfg.get("task_file")
        if not task_file:
            raise ValueError("Minecraft tasks.task_file must be provided")

        if _OPENHA_ROOT not in sys.path:
            sys.path.insert(0, _OPENHA_ROOT)

        # NOTE: upstream `gen_task_config` routes by task-name prefix, but the
        # smelt_item spawn JSON reuses the `craft_item:` prefix, so craft-side
        # dispatch wins for keys present in both (e.g. iron_ingot, gold_ingot).
        # We dispatch by the explicit `task_type` field instead.
        from openagents.assets import EVENT_DESCRIPTION
        from openagents.envs.tasks.craft_item import gen_craft_item_task_config
        from openagents.envs.tasks.mine_block import gen_mine_block_task_config
        from openagents.envs.tasks.smelt_item import gen_smelt_item_task_config

        task_type_dispatch = {
            "craft_item": gen_craft_item_task_config,
            "smelt_item": gen_smelt_item_task_config,
            "mine_block": gen_mine_block_task_config,
        }

        with open(task_file, "r", encoding="utf-8") as f:
            task_entries = json.load(f)

        task_id_filter = tasks_cfg.get("task_id")
        task_type_filter = tasks_cfg.get("task_type")
        tasks: list[TaskDescriptor] = []

        for entry in task_entries:
            task_name = entry["task_name"]
            difficulty = entry.get("difficulty", "zero")
            task_id = f"{task_name}_{difficulty}"

            if task_id_filter and task_id != task_id_filter:
                continue
            if task_type_filter and entry.get("task_type") != task_type_filter:
                continue

            task_type = entry.get("task_type", "")
            gen_fn = task_type_dispatch.get(task_type)
            if gen_fn is None:
                raise ValueError(
                    f"Unknown minecraft task_type={task_type!r} for task={task_name!r}; "
                    f"expected one of {sorted(task_type_dispatch)}"
                )
            task_config = gen_fn(task_name, difficulty)
            # Upstream gen_fn picks task_description via random.choice; pin to
            # the first entry so repeated runs use a stable instruction.
            if task_name in EVENT_DESCRIPTION:
                task_config["task_description"] = EVENT_DESCRIPTION[task_name][0]
            tasks.append(
                MinecraftTask(
                    task_id=task_id,
                    instruction=task_config["task_description"],
                    task_type=entry.get("task_type", ""),
                    difficulty=difficulty,
                    task_config=task_config,
                )
            )

        logger.info("Collected %d Minecraft tasks from %s", len(tasks), task_file)
        return tasks

    def save_observation(self, obs: dict, path: Path) -> None:
        screenshot_path = path / "screenshot.png"
        with open(screenshot_path, "wb") as f:
            f.write(obs["screenshot_png"])

        status_path = path / "status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(obs.get("status", {}), f, indent=2, ensure_ascii=False)

    def load_saved_observation(self, step_dir: Path) -> list[dict]:
        # status.json on disk is for developer debug only — the model must
        # read game state from the screenshot, never from a structured status
        # crutch. Missing screenshot degrades to empty list (analyzer falls
        # back to pure-text reasoning instead of crashing).
        from anything2skill.vlm.client import encode_image_file

        screenshot_path = step_dir / "screenshot.png"
        if not screenshot_path.is_file():
            return []
        try:
            data_url = encode_image_file(str(screenshot_path))
        except Exception as e:
            logger.warning("Failed to encode screenshot %s: %s", screenshot_path, e)
            return []
        return [
            {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "high"},
            }
        ]

    def get_result_subdir(self, task: TaskDescriptor) -> str:
        if isinstance(task, MinecraftTask):
            return f"{task.task_type}/{task.task_id}"
        return task.task_id
