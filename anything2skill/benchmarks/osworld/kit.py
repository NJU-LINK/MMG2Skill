"""OSWorld BenchmarkKit: GUI automation on Ubuntu desktop via pyautogui."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from anything2skill.benchmarks.osworld.claude_resize import (
    is_claude_model,
    resize_screenshot_for_claude,
)
from anything2skill.benchmarks.osworld.prompts import (
    BRIDGING_TEXT,
    PLANNER_GUIDANCE,
    REFLECTION_GUIDANCE,
    REVISER_GUIDANCE,
    SKILL_EXTRACTION_GUIDANCE,
    SYSTEM_PROMPT,
)
from anything2skill.benchmarks.registry import register_kit
from anything2skill.benchmark_kit import BenchmarkKit, TaskDescriptor
from anything2skill.env_base import EnvironmentInterface

logger = logging.getLogger("anything2skill.benchmarks.osworld")

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_OSWORLD_ROOT = os.path.join(_PROJECT_ROOT, "OSWorld")


@dataclass
class OSWorldTask(TaskDescriptor):
    """OSWorld-specific task descriptor."""

    domain: str = ""
    task_config: dict = field(default_factory=dict)


@register_kit("osworld")
class OSWorldKit(BenchmarkKit):
    """BenchmarkKit for OSWorld GUI automation benchmark.

    Observations are screenshots (bytes). Actions are pyautogui Python code.
    """

    def __init__(self, env_cfg: dict | None = None):
        super().__init__(env_cfg)
        self.client_password = self.env_cfg.get("client_password", "password")
        self._model_name = self.env_cfg.get("agent_model", "")

    # ------------------------------------------------------------------
    # Observation encoding (env feedback)
    # ------------------------------------------------------------------

    def encode_observation(self, obs: dict) -> list[dict]:
        screenshot_bytes = obs["screenshot"]
        if is_claude_model(self._model_name):
            screenshot_bytes = resize_screenshot_for_claude(screenshot_bytes)
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_b64}",
                    "detail": "high",
                },
            }
        ]

    # ------------------------------------------------------------------
    # Domain prompts (text lives in prompts.py)
    # ------------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.strip().format(client_password=self.client_password)

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
        return SKILL_EXTRACTION_GUIDANCE.strip()

    @property
    def reviser_guidance(self) -> str:
        return REVISER_GUIDANCE.strip()

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def parse_actions(self, response: str) -> list[str]:
        """Extract executable pyautogui actions from LLM response."""
        cleaned = "\n".join(
            line.strip() for line in response.split(";") if line.strip()
        )

        special_commands = {"WAIT", "DONE", "FAIL"}

        if cleaned.strip() in special_commands:
            return [cleaned.strip()]

        # Alternation walked in source order: the signal-fence arm is tried
        # first at each position so multiline forms like ```DONE\n``` aren't
        # swallowed by the code-fence arm's language-tag group.
        pattern = re.compile(
            r"```\s*(?P<sig>WAIT|DONE|FAIL)\s*```"
            r"|"
            r"```(?:(?!(?:WAIT|DONE|FAIL)\s)\w+\s+)?(?P<code>.*?)```",
            re.DOTALL,
        )

        codes: list[str] = []
        for m in pattern.finditer(cleaned):
            sig = m.group("sig")
            if sig is not None:
                codes.append(sig)
                continue
            match = (m.group("code") or "").strip()
            lines = match.split("\n")
            first_line = lines[0] if lines else ""
            if first_line in special_commands:
                codes.append(first_line)
            elif match in special_commands:
                codes.append(match)
            elif lines[-1] in special_commands:
                if len(lines) > 1:
                    codes.append("\n".join(lines[:-1]))
                codes.append(lines[-1])
            else:
                codes.append(match)

        if not codes:
            stripped = response.strip()
            if stripped in special_commands:
                return [stripped]

        return codes

    # ------------------------------------------------------------------
    # Observation saving
    # ------------------------------------------------------------------

    def save_observation(self, obs: dict, path: Path) -> None:
        screenshot_path = path / "screenshot.png"
        with open(screenshot_path, "wb") as f:
            f.write(obs["screenshot"])

    def load_saved_observation(self, step_dir: Path) -> list[dict]:
        """Load a saved screenshot from disk as VLM content blocks."""
        screenshot_path = step_dir / "screenshot.png"
        if not screenshot_path.exists():
            return []

        if is_claude_model(self._model_name):
            with open(screenshot_path, "rb") as f:
                resized = resize_screenshot_for_claude(f.read())
            data_url = (
                f"data:image/png;base64,{base64.b64encode(resized).decode('utf-8')}"
            )
        else:
            from anything2skill.vlm.client import encode_image_file

            data_url = encode_image_file(str(screenshot_path))
        return [{"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}]

    # ------------------------------------------------------------------
    # Environment creation
    # ------------------------------------------------------------------

    def create_env(self, env_cfg: dict) -> EnvironmentInterface:
        """Create DesktopEnv + OSWorldEnvWrapper."""
        from anything2skill.benchmarks.osworld.env_wrapper import OSWorldEnvWrapper

        if _OSWORLD_ROOT not in sys.path:
            sys.path.insert(0, _OSWORLD_ROOT)

        # DesktopEnv has relative path deps in cache_dir
        os.chdir(_OSWORLD_ROOT)

        from desktop_env.desktop_env import DesktopEnv

        screen_size = tuple(env_cfg.get("screen_size", [1920, 1080]))

        desktop_env = DesktopEnv(
            provider_name=env_cfg.get("provider_name", "docker"),
            path_to_vm=env_cfg.get("path_to_vm"),
            action_space=env_cfg.get("action_space", "pyautogui"),
            screen_size=screen_size,
            headless=env_cfg.get("headless", True),
            os_type="Ubuntu",
            require_a11y_tree=env_cfg.get("observation_type", "screenshot")
            in ("a11y_tree", "screenshot_a11y_tree", "som"),
            client_password=env_cfg.get("client_password", "password"),
            cache_dir=os.path.join(_OSWORLD_ROOT, "cache"),
        )

        return OSWorldEnvWrapper(
            desktop_env,
            stabilization_wait=env_cfg.get("stabilization_wait", 60.0),
            pre_evaluate_wait=env_cfg.get("pre_evaluate_wait", 20.0),
            wait_action_duration=env_cfg.get("wait_action_duration", 20.0),
            recording=env_cfg.get("recording", True),
            model_name=env_cfg.get("agent_model", ""),
            screen_size=screen_size,
        )

    # ------------------------------------------------------------------
    # Task collection
    # ------------------------------------------------------------------

    def collect_tasks(self, cfg: dict) -> list[TaskDescriptor]:
        """Collect OSWorld tasks from test_all.json."""
        tasks_cfg = cfg.get("tasks", {})
        data_cfg = cfg.get("data", {})

        test_config_base_dir = tasks_cfg.get("configs_base_dir") or os.path.join(
            _OSWORLD_ROOT, "evaluation_examples"
        )
        test_all_path = os.path.join(test_config_base_dir, "test_all.json")
        with open(test_all_path, "r", encoding="utf-8") as f:
            test_all_meta = json.load(f)

        domain = tasks_cfg.get("domain", "os")
        if domain == "all":
            task_ids_by_domain = test_all_meta
        else:
            task_ids_by_domain = {domain: test_all_meta.get(domain, [])}

        task_id_filter = tasks_cfg.get("task_id")
        task_file = tasks_cfg.get("task_file")
        task_id_set: set[str] | None = None
        if task_file:
            with open(task_file, "r", encoding="utf-8") as f:
                task_id_set = set(json.load(f))
            logger.info("Loaded %d task IDs from %s", len(task_id_set), task_file)

        tasks: list[TaskDescriptor] = []

        for task_domain, task_ids in task_ids_by_domain.items():
            for example_id in task_ids:
                if task_id_filter and example_id != task_id_filter:
                    continue
                if task_id_set is not None and example_id not in task_id_set:
                    continue

                config_file = os.path.join(
                    test_config_base_dir, f"examples/{task_domain}/{example_id}.json"
                )
                if not os.path.exists(config_file):
                    logger.warning("Task config not found: %s", config_file)
                    continue

                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)

                tasks.append(
                    OSWorldTask(
                        task_id=example_id,
                        instruction=example["instruction"],
                        domain=task_domain,
                        task_config=example,
                    )
                )

        logger.info("Collected %d tasks", len(tasks))
        return tasks

    def get_result_subdir(self, task: TaskDescriptor) -> str:
        """OSWorld organizes results by domain/task_id."""
        if isinstance(task, OSWorldTask):
            return f"{task.domain}/{task.task_id}"
        return task.task_id
