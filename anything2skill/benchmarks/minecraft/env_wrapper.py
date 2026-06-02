from __future__ import annotations

import io
import logging
import os
import re
import time
from typing import TYPE_CHECKING

from anything2skill.env_base import EnvironmentInterface

if TYPE_CHECKING:
    from anything2skill.benchmark_kit import TaskDescriptor


logger = logging.getLogger("anything2skill.benchmarks.minecraft.env")

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_OPENHA_ROOT = os.path.join(_PROJECT_ROOT, "OpenHA")

_REPEAT_RE = re.compile(r"^(.*?)\s*\*\s*(-?\d+)\s*$", re.DOTALL)


def _expand_dsl_action(action: str, max_repeat: int) -> list[str]:
    """Expand a DSL action string into a list of per-frame compound strings.

    Each returned element becomes ``Action: <compound>`` fed to the tokenizer
    and is executed as one ``sim.step`` frame. Always returns at least one
    element; empty or invalid input collapses to ``["no_op"]``.
    """
    action = action.strip()
    if not action:
        return ["no_op"]

    m = _REPEAT_RE.match(action)
    if not m:
        return [action]

    compound = m.group(1).strip()
    count = int(m.group(2))
    if count <= 0 or not compound:
        return ["no_op"]
    if count > max_repeat:
        logger.warning(
            "Repeat count %d exceeds cap %d, clamping.", count, max_repeat
        )
        count = max_repeat
    return [compound] * count


_TICKS_PER_SECOND = 20


def _seconds_to_frames(seconds: float) -> int:
    return max(1, int(round(seconds * _TICKS_PER_SECOND)))


class MinecraftEnvWrapper(EnvironmentInterface):
    def __init__(self, env_cfg: dict):
        self._render_size = env_cfg.get("render_size", [640, 360])
        self._wait_action_duration = float(env_cfg.get("wait_action_duration", 1.0))
        self._max_repeat = int(env_cfg.get("max_repeat", 200))
        # None -> defer to task_config["maximum_steps"] (upstream default 600).
        # A concrete value overrides the upstream per-task budget.
        cfg_budget = env_cfg.get("max_env_steps")
        self._cfg_max_env_steps = int(cfg_budget) if cfg_budget is not None else None
        self._sim = None
        self._ever_rewarded = False
        self._env_step_count = 0
        self._max_env_steps = 600
        self._tokenizer = None

        self._TextActionTokenizer = None
        self._post_info = None
        self._np = None
        self._Image = None

        self._last_obs = {"screenshot_png": b"", "status": {}}

    def _ensure_imports(self):
        import os as _os
        import sys as _sys

        openha_root = _os.path.abspath(_OPENHA_ROOT)
        if openha_root not in _sys.path:
            _sys.path.insert(0, openha_root)

        if self._TextActionTokenizer is None or self._post_info is None:
            from openagents.agents.utils.action_mapping import TextActionTokenizer
            from openagents.envs.callbacks.recording import post_info

            self._TextActionTokenizer = TextActionTokenizer
            self._post_info = post_info

        if self._np is None:
            import numpy as np

            self._np = np

        if self._Image is None:
            from PIL import Image

            self._Image = Image

        if self._tokenizer is None:
            self._tokenizer = self._TextActionTokenizer(action_chunk_len=1)

    def reset(self, task: TaskDescriptor) -> dict:
        self._ensure_imports()

        if self._sim is not None:
            try:
                self._sim.close()
            finally:
                self._sim = None

        self._ever_rewarded = False
        self._env_step_count = 0
        self._last_obs = {"screenshot_png": b"", "status": {}}

        task_config = task.task_config
        seed = task_config.get("seed")
        init_actions = task_config.get("init_actions", [])
        callback_cfg = task_config.get("callback", {})
        init_inventory_cfg = callback_cfg.get("init_inventory", {})
        init_inventory = init_inventory_cfg.get("init_inventory", [])
        commands = callback_cfg.get("commands", [])
        reward_cfg = task_config.get("rewards", [])
        mobs = callback_cfg.get("mobs")

        from minestudio.simulator import MinecraftSim
        from minestudio.simulator.callbacks import RewardsCallback
        from openagents.envs.callbacks import (
            CommandsCallback,
            InitInventoryCallback,
            SummonMobsCallback,
        )

        callbacks = [
            InitInventoryCallback(
                init_inventory,
                inventory_distraction_level=init_inventory_cfg.get(
                    "inventory_distraction_level", [0]
                ),
                equip_distraction_level=init_inventory_cfg.get(
                    "equip_distraction_level", [0]
                ),
                forbidden_slots=init_inventory_cfg.get("forbidden_slots", []),
            ),
            RewardsCallback(reward_cfg),
            CommandsCallback(commands),
        ]
        if mobs:
            callbacks.append(SummonMobsCallback(mobs))

        self._sim = MinecraftSim(
            action_type="env",
            render_size=tuple(self._render_size),
            seed=seed,
            callbacks=callbacks,
        )

        obs, info = self._sim.reset()

        for init_action in init_actions:
            time.sleep(0.1)
            obs, reward, terminated, truncated, info = self._sim.step(init_action)
            if reward > 0:
                self._ever_rewarded = True
            if terminated or truncated:
                logger.warning("Minecraft env terminated during reset init_actions.")
                break

        self._max_env_steps = (
            self._cfg_max_env_steps
            if self._cfg_max_env_steps is not None
            else task_config.get("maximum_steps", 600)
        )

        time.sleep(1.0)  # wait for chunk loading and lighting to stabilize

        obs, reward, terminated, truncated, info = self._sim.step(self._make_noop_action())
        if reward > 0:
            self._ever_rewarded = True
        if terminated or truncated:
            logger.warning("Minecraft env terminated on initial noop after reset.")

        return self._build_obs(obs, info)

    def _build_obs(self, obs, info) -> dict:
        del obs

        self._ensure_imports()
        processed_info = self._post_info(info)

        image = self._Image.fromarray(self._np.uint8(info["pov"]))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        obs_dict = {
            "screenshot_png": buffer.getvalue(),
            "status": processed_info,
        }
        self._last_obs = obs_dict
        return obs_dict

    def step(self, action: str, pause: float = 0.0) -> tuple[dict, float, bool, dict]:
        if self._sim is None:
            raise RuntimeError("Minecraft environment has not been reset.")

        self._ensure_imports()

        if action in {"DONE", "FAIL"}:
            return self._last_obs, 0.0, True, {"signal": action}

        if action == "WAIT":
            wait_frames = _seconds_to_frames(self._wait_action_duration)
            env_actions = [self._make_noop_action() for _ in range(wait_frames)]
        else:
            frame_atoms = _expand_dsl_action(action, self._max_repeat)
            env_actions = []
            for atom in frame_atoms:
                env_actions.extend(self._tokenizer.decode(f"Action: {atom}"))

        # Minecraft ticks only advance when we call sim.step(); wall-clock sleep
        # would freeze the game. Convert `pause` seconds to noop frames and
        # append them so animations / GUI transitions settle before we snap the
        # observation. WAIT already pads itself, so skip extra settling there.
        if action != "WAIT" and pause > 0:
            settle_frames = _seconds_to_frames(pause)
            env_actions.extend(self._make_noop_action() for _ in range(settle_frames))

        total_reward = 0.0
        done = False
        obs_dict = self._last_obs
        info_dict = {"status": self._last_obs.get("status", {})}

        for env_action in env_actions:
            obs, reward, terminated, truncated, info = self._sim.step(env_action)
            self._env_step_count += 1

            reward_value = float(reward)
            total_reward += reward_value
            if reward_value > 0:
                self._ever_rewarded = True

            done = bool(terminated or truncated)
            if self._env_step_count >= self._max_env_steps:
                done = True

            obs_dict = self._build_obs(obs, info)
            info_dict = {
                "status": obs_dict["status"],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "env_step_count": self._env_step_count,
            }

            if done:
                break

        return obs_dict, total_reward, done, info_dict

    def evaluate(self) -> float:
        return 1.0 if self._ever_rewarded else 0.0

    def close(self):
        if self._sim is not None:
            try:
                self._sim.close()
            except Exception:
                pass
        self._sim = None

    @property
    def vm_ip(self) -> str:
        return ""

    def _make_noop_action(self) -> dict:
        self._ensure_imports()
        return {
            "hotbar.1": 0,
            "hotbar.2": 0,
            "hotbar.3": 0,
            "hotbar.4": 0,
            "hotbar.5": 0,
            "hotbar.6": 0,
            "hotbar.7": 0,
            "hotbar.8": 0,
            "hotbar.9": 0,
            "forward": 0,
            "back": 0,
            "left": 0,
            "right": 0,
            "sprint": 0,
            "sneak": 0,
            "jump": 0,
            "inventory": 0,
            "drop": 0,
            "attack": 0,
            "use": 0,
            "camera": self._np.array([0.0, 0.0]),
        }
