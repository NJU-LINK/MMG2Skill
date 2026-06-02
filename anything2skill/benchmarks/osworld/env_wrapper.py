"""OSWorld environment wrapper with GUI-specific behavior (waits, recording)."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from anything2skill.benchmarks.osworld.claude_resize import (
    is_claude_model,
    rewrite_claude_pixel_coordinates,
)
from anything2skill.benchmarks.osworld.pyautogui_sanitizer import (
    rewrite_kimi_normalized_coordinates,
    rewrite_pyautogui_text_inputs,
)
from anything2skill.env_base import EnvironmentInterface

if TYPE_CHECKING:
    from anything2skill.benchmark_kit import TaskDescriptor

logger = logging.getLogger("anything2skill.benchmarks.osworld.env")


class OSWorldEnvWrapper(EnvironmentInterface):
    """Wraps OSWorld's DesktopEnv to implement EnvironmentInterface.

    Absorbs GUI-specific behavior that was previously in lib_run_single:
    - Post-reset stabilization wait (VM boot time)
    - Pre-evaluate wait (GUI animation settle)
    - Screen recording start/stop

    Usage:
        from desktop_env.desktop_env import DesktopEnv
        desktop_env = DesktopEnv(provider_name="docker", ...)
        env = OSWorldEnvWrapper(desktop_env, stabilization_wait=60.0)
    """

    def __init__(
        self,
        desktop_env,
        *,
        stabilization_wait: float = 60.0,
        pre_evaluate_wait: float = 20.0,
        wait_action_duration: float = 20.0,
        recording: bool = True,
        model_name: str = "",
        screen_size: tuple[int, int] | list[int] = (1920, 1080),
    ):
        self.env = desktop_env
        self.stabilization_wait = stabilization_wait
        self.pre_evaluate_wait = pre_evaluate_wait
        self.wait_action_duration = wait_action_duration
        self.recording = recording
        self.kimi_coord_rewrite_enabled = "kimi" in model_name.lower()
        self.claude_coord_rewrite_enabled = is_claude_model(model_name)
        self.screen_size = tuple(screen_size)
        self._recording_active = False

    def reset(self, task: TaskDescriptor) -> dict:
        from anything2skill.benchmarks.osworld.kit import OSWorldTask

        task_config = task.task_config if isinstance(task, OSWorldTask) else {}
        self.env.reset(task_config=task_config)
        logger.info(
            "Waiting %.0fs for VM stabilization...", self.stabilization_wait,
        )
        time.sleep(self.stabilization_wait)
        obs = self.env._get_obs()
        if self.recording:
            self.env.controller.start_recording()
            self._recording_active = True
        return obs

    def step(self, action: str, pause: float = 2.0) -> tuple[dict, float, bool, dict]:
        requested_action = action
        executed_action = action
        kimi_action_rewritten = False
        claude_action_rewritten = False
        if action == "WAIT":
            pause = self.wait_action_duration / 2
        elif isinstance(action, str) and action not in {"DONE", "FAIL"}:
            executed_action = rewrite_pyautogui_text_inputs(action)
            if self.kimi_coord_rewrite_enabled:
                text_rewritten_action = executed_action
                executed_action = rewrite_kimi_normalized_coordinates(
                    executed_action,
                    self.screen_size,
                )
                kimi_action_rewritten = executed_action != text_rewritten_action
            if self.claude_coord_rewrite_enabled:
                pre_claude_action = executed_action
                executed_action = rewrite_claude_pixel_coordinates(
                    executed_action,
                    self.screen_size,
                )
                claude_action_rewritten = executed_action != pre_claude_action

        obs, reward, done, info = self.env.step(executed_action, pause)

        metadata = {
            "requested_action": requested_action,
            "executed_action": executed_action,
            "action_rewritten": requested_action != executed_action,
            "kimi_coord_rewrite_enabled": self.kimi_coord_rewrite_enabled,
            "kimi_action_rewritten": kimi_action_rewritten,
            "claude_coord_rewrite_enabled": self.claude_coord_rewrite_enabled,
            "claude_action_rewritten": claude_action_rewritten,
        }
        if isinstance(info, dict):
            info = {**info, **metadata}
        else:
            info = metadata
        return obs, reward, done, info

    def evaluate(self) -> float:
        logger.info(
            "Waiting %.0fs before evaluation...", self.pre_evaluate_wait,
        )
        time.sleep(self.pre_evaluate_wait)
        return self.env.evaluate()

    def stop_recording(self, result_dir: str) -> None:
        """Stop recording and save MP4. Called by runner via duck-typing."""
        if self._recording_active and self.env.controller is not None:
            mp4_path = os.path.join(result_dir, "recording.mp4")
            self.env.controller.end_recording(mp4_path)
            self._recording_active = False
            logger.info("Recording saved to %s", mp4_path)

    def close(self):
        if self._recording_active:
            logger.warning("Recording still active at close(), stopping without save")
            try:
                self.env.controller.end_recording("")
            except Exception:
                pass
            self._recording_active = False
        self.env.close()

    @property
    def vm_ip(self) -> str:
        return getattr(self.env, "vm_ip", "")
