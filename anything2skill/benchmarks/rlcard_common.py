"""Shared RLCard utilities for card game benchmarks.

Provides:
- Game-opponent compatibility table and opponent factory
- BaseRLCardEnvWrapper: common base for Mahjong / Doudizhu / No-Limit Hold'em wrappers
- expand_seeds: deterministic seed generation from (seed_start, num_seeds)
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import struct
import sys
from typing import TYPE_CHECKING, Any

from anything2skill.env_base import EnvironmentInterface

if TYPE_CHECKING:
    from anything2skill.benchmark_kit import TaskDescriptor

logger = logging.getLogger("anything2skill.benchmarks.rlcard_common")

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_RLCARD_ROOT = os.path.join(_PROJECT_ROOT, "RLCard")
_MODELS_ROOT = os.path.join(_PROJECT_ROOT, "rlcard_models")


def expand_seeds(seed_start: int, num_seeds: int) -> list[int]:
    """Deterministically expand (seed_start, num_seeds) into well-spread seeds.

    Uses SHA-256 to hash (seed_start, index) pairs, producing seeds in
    [0, 2^31) that are uncorrelated even for adjacent seed_start values.
    """
    seeds = []
    for i in range(num_seeds):
        h = hashlib.sha256(struct.pack(">qq", seed_start, i)).digest()
        seeds.append(int.from_bytes(h[:4], "big") % (2**31))
    return seeds


def ensure_rlcard_path() -> None:
    """Ensure the vendored RLCard is on sys.path and loaded.

    If a system-installed rlcard was already imported, replace it with
    the local fork to pick up fixes (Python 3.12+ compat, etc.).
    """
    if _RLCARD_ROOT not in sys.path:
        sys.path.insert(0, _RLCARD_ROOT)
    import importlib
    if "rlcard" in sys.modules:
        mod = sys.modules["rlcard"]
        mod_path = getattr(mod, "__file__", "") or ""
        if _RLCARD_ROOT not in mod_path:
            to_remove = [k for k in sys.modules if k == "rlcard" or k.startswith("rlcard.")]
            for k in to_remove:
                del sys.modules[k]
            importlib.import_module("rlcard")


# ---------------------------------------------------------------------------
# Game-opponent compatibility table
# ---------------------------------------------------------------------------

GAME_OPPONENTS: dict[str, list[dict[str, Any]]] = {
    "mahjong": [
        {"type": "dqn",  "requires_checkpoint": True, "checkpoint_file": "checkpoint_dqn.pt"},
        {"type": "nfsp", "requires_checkpoint": True, "checkpoint_file": "checkpoint_nfsp.pt"},
        {"type": "dmc",  "requires_checkpoint": True, "checkpoint_file": "checkpoint_dmc.pt"},
        {"type": "random"},
    ],
    "doudizhu": [
        {"type": "dqn",  "requires_checkpoint": True, "checkpoint_file": "checkpoint_dqn.pt"},
        {"type": "nfsp", "requires_checkpoint": True, "checkpoint_file": "checkpoint_nfsp.pt"},
        {"type": "dmc",  "requires_checkpoint": True, "checkpoint_file": "checkpoint_dmc.pt"},
        {"type": "random"},
    ],
    "no-limit-holdem": [
        {"type": "dqn",  "requires_checkpoint": True, "checkpoint_file": "checkpoint_dqn.pt"},
        {"type": "nfsp", "requires_checkpoint": True, "checkpoint_file": "checkpoint_nfsp.pt"},
        {"type": "dmc",  "requires_checkpoint": True, "checkpoint_file": "checkpoint_dmc.pt"},
        {"type": "random"},
    ],
}


# ---------------------------------------------------------------------------
# Opponent loading helpers
# ---------------------------------------------------------------------------

def _try_load_checkpoint(agent_type: str, checkpoint_path: str, player_id: int = 0):
    """Load a trained agent from checkpoint. Returns agent or None.

    Tries per-player checkpoint first (``checkpoint_{type}_p{player_id}.pt``),
    then falls back to the generic file. This handles asymmetric games (e.g.
    doudizhu) where different seats have different state shapes.
    """
    import os as _os

    if player_id > 0:
        base, ext = _os.path.splitext(checkpoint_path)
        per_player_path = f"{base}_p{player_id}{ext}"
        if _os.path.exists(per_player_path):
            checkpoint_path = per_player_path

    if not os.path.exists(checkpoint_path):
        return None
    try:
        ensure_rlcard_path()
        import torch  # noqa: F401

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if agent_type == "nfsp":
            from rlcard.agents.nfsp_agent import NFSPAgent
            agent = NFSPAgent.from_checkpoint(checkpoint=checkpoint)
        elif agent_type == "dmc":
            from rlcard.agents.dmc_agent.model import DMCAgent
            agent = DMCAgent(
                state_shape=checkpoint["state_shape"],
                action_shape=checkpoint["action_shape"],
                device="cpu",
            )
            agent.load_state_dict(checkpoint["model_state_dict"])
            agent.eval()
        elif agent_type == "dqn":
            from rlcard.agents.dqn_agent import DQNAgent
            agent = DQNAgent.from_checkpoint(checkpoint=checkpoint)
        else:
            return None

        logger.info("Loaded %s opponent from %s", agent_type, checkpoint_path)
        return agent
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s checkpoint (%s)", agent_type, exc)
        return None


def create_random_agent(num_actions: int):
    """Create a RandomAgent without triggering rlcard.agents.__init__."""
    spec = importlib.util.spec_from_file_location(
        "rlcard.agents.random_agent",
        os.path.join(_RLCARD_ROOT, "rlcard", "agents", "random_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.RandomAgent(num_actions=num_actions)


def create_opponent(
    game: str,
    opponent_type: str,
    num_actions: int,
    player_id: int = 0,
) -> Any:
    """Create a single opponent agent instance.

    Args:
        game: RLCard game name.
        opponent_type: "auto" | "random" | "nfsp" | "dqn" | "dmc".
        num_actions: Number of actions in the game.
        player_id: Seat index for this opponent (used for per-player checkpoints).

    Returns:
        An agent with ``eval_step(state)`` method.
    """
    compatible = GAME_OPPONENTS.get(game, [])
    compatible_types = [e["type"] for e in compatible]

    if opponent_type == "auto":
        for entry in compatible:
            if entry.get("requires_checkpoint"):
                ckpt_dir = os.path.join(_MODELS_ROOT, f"{game}_{entry['type']}")
                ckpt_path = os.path.join(ckpt_dir, entry["checkpoint_file"])
                agent = _try_load_checkpoint(entry["type"], ckpt_path, player_id)
                if agent is not None:
                    return agent
            elif entry["type"] == "random":
                return create_random_agent(num_actions)
        return create_random_agent(num_actions)

    if opponent_type not in compatible_types:
        raise ValueError(
            f"Opponent type '{opponent_type}' is not compatible with {game}. "
            f"Compatible types: {compatible_types}"
        )

    if opponent_type == "random":
        return create_random_agent(num_actions)

    entry = next(e for e in compatible if e["type"] == opponent_type)
    ckpt_dir = os.path.join(_MODELS_ROOT, f"{game}_{opponent_type}")
    ckpt_path = os.path.join(ckpt_dir, entry["checkpoint_file"])
    agent = _try_load_checkpoint(opponent_type, ckpt_path, player_id)
    if agent is None:
        raise FileNotFoundError(
            f"{opponent_type.upper()} checkpoint not found at {ckpt_path}. "
            f"Train first: bash scripts/rlcard/setup.sh --game {game}"
        )
    return agent


# ---------------------------------------------------------------------------
# Base environment wrapper
# ---------------------------------------------------------------------------

class BaseRLCardEnvWrapper(EnvironmentInterface):
    """Base class for RLCard game wrappers.

    Handles: rlcard env creation, opponent management, advance_opponents,
    DONE/FAIL signals, evaluate(). Subclasses implement _build_obs() and
    _action_str_to_id().
    """

    game_name: str = ""  # override in subclass

    def __init__(
        self,
        seed: int | None = None,
        num_players: int | None = None,
        opponent_type: str = "auto",
    ):
        ensure_rlcard_path()
        import rlcard

        config: dict = {}
        if seed is not None:
            config["seed"] = seed
        if num_players is not None:
            config["game_num_players"] = num_players

        self._env = rlcard.make(self.game_name, config=config)
        self._num_players = self._env.num_players
        self._done = False
        self._payoff: float = 0.0
        self._last_opponent_actions: list[dict[str, Any]] = []
        self._illegal_action_count: int = 0

        # Create opponents for seats 1..N-1 (per-seat for asymmetric games)
        self._opponent_agents: dict = {}
        if self._num_players > 1:
            for pid in range(1, self._num_players):
                agent = create_opponent(
                    self.game_name, opponent_type, self._env.num_actions,
                    player_id=pid,
                )
                self._opponent_agents[pid] = agent
                logger.info(
                    "%s opponent seat %d: %s (type=%s)",
                    self.game_name, pid, type(agent).__name__, opponent_type,
                )

    # ------------------------------------------------------------------
    # Opponent advancement
    # ------------------------------------------------------------------

    def _advance_opponents(self) -> list[dict[str, Any]]:
        """Auto-play opponent turns until player 0's turn or game end."""
        actions: list[dict[str, Any]] = []
        while not self._env.is_over() and self._env.get_player_id() != 0:
            player_id = self._env.get_player_id()
            agent = self._opponent_agents.get(player_id)
            if agent is None:
                logger.warning("No opponent agent for player %d; breaking", player_id)
                break
            state = self._env.get_state(player_id)
            action, _ = agent.eval_step(state)
            logger.debug("opponent (player %d): action=%s", player_id, action)

            use_raw = isinstance(action, str)
            action_name = action if use_raw else self._decode_action_id(action)

            self._env.step(action, raw_action=use_raw)
            actions.append({"player": player_id, "action": action_name})
        return actions

    def _decode_action_id(self, action_id: int) -> str:
        """Convert an integer action id back to a human-readable string.
        Override in subclass for game-specific decoding.
        """
        return str(action_id)

    # ------------------------------------------------------------------
    # EnvironmentInterface
    # ------------------------------------------------------------------

    def reset(self, task: TaskDescriptor) -> dict:
        self._done = False
        self._payoff = 0.0
        self._last_opponent_actions = []
        self._illegal_action_count = 0

        # Per-task seed for reproducible deals
        task_seed = getattr(task, "seed", None)
        if task_seed is not None:
            self._env.seed(task_seed)

        state, player_id = self._env.reset()
        logger.debug("reset: player_id=%s", player_id)

        if player_id != 0:
            self._last_opponent_actions = self._advance_opponents()
            if not self._env.is_over():
                state = self._env.get_state(0)

        return self._build_obs(state)

    def step(self, action: str, pause: float = 0.0) -> tuple[dict, float, bool, dict]:  # noqa: ARG002
        if action in ("DONE", "FAIL"):
            self._done = True
            return {"terminal": True, "action": action}, self._payoff, True, {}

        action_id = self._action_str_to_id(action)
        legal = list(self._env.get_state(0)["legal_actions"].keys())

        if action_id is None or action_id not in legal:
            self._illegal_action_count += 1
            logger.warning(
                "Illegal action '%s' (id=%s, legal=%s), count=%d — requesting retry",
                action, action_id, legal, self._illegal_action_count,
            )
            obs = self._build_obs(self._env.get_state(0))
            return obs, 0.0, False, {"illegal_action": True}

        next_state, next_player_id = self._env.step(action_id)
        logger.debug("step: action=%s(%s) next_player=%s", action, action_id, next_player_id)

        self._last_opponent_actions = []
        if not self._env.is_over() and next_player_id != 0:
            self._last_opponent_actions = self._advance_opponents()

        done = self._env.is_over()
        reward = 0.0
        if done:
            payoffs = self._env.get_payoffs()
            self._payoff = float(payoffs[0])
            reward = self._payoff
            self._done = True

        obs = self._build_obs(self._env.get_state(0))
        return obs, reward, done, {}

    def evaluate(self) -> float:
        """Return the raw payoff from the game (e.g. chips won/lost)."""
        if not self._done:
            payoffs = self._env.get_payoffs()
            self._payoff = float(payoffs[0])
        return self._payoff

    @property
    def illegal_action_count(self) -> int:
        """Return the number of illegal actions attempted by the agent."""
        return self._illegal_action_count

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Abstract (subclass must implement)
    # ------------------------------------------------------------------

    def _build_obs(self, state: dict) -> dict:
        raise NotImplementedError

    def _action_str_to_id(self, action: str) -> int | None:
        raise NotImplementedError
