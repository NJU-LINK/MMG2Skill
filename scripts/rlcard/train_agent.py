"""Train an RL agent for RLCard games.

Supports NFSP/DQN/DMC agent types on the 3 benchmark games. The trained
checkpoint is saved under ``rlcard_models/{game}_{agent_type}/`` and will be
auto-loaded by ``RLCardEnvWrapper`` when ``opponent_type=auto``.

Usage::

    # NFSP on Mahjong
    uv run python scripts/rlcard/train_agent.py --game mahjong --agent nfsp

    # DMC on Doudizhu
    uv run python scripts/rlcard/train_agent.py --game doudizhu --agent dmc

Notes
-----
* Avoids importing ``rlcard.models`` (which triggers a ``distutils`` import
  chain broken on Python 3.12+).
* DMC uses its own multiprocess trainer (``DMCTrainer``); NFSP/DQN use the
  standard ``env.run()`` + ``reorganize`` loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RLCARD_ROOT = _PROJECT_ROOT / "RLCard"
if str(_RLCARD_ROOT) not in sys.path:
    sys.path.insert(0, str(_RLCARD_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

try:
    import torch  # noqa: E402  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Training requires PyTorch. Install it with:\n"
        "    uv pip install 'anything2skill[train]'\n"
        "or directly:\n"
        "    uv pip install torch"
    ) from exc

import rlcard  # noqa: E402
from rlcard.agents.random_agent import RandomAgent  # noqa: E402
from rlcard.utils.utils import reorganize, tournament  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_agent")

# Game -> compatible trainable agents (paper keeps 3 games)
TRAINABLE_AGENTS: dict[str, list[str]] = {
    "mahjong": ["nfsp", "dqn", "dmc"],
    "doudizhu": ["nfsp", "dqn", "dmc"],
    "no-limit-holdem": ["nfsp", "dqn", "dmc"],
}

CHECKPOINT_FILENAMES: dict[str, str] = {
    "nfsp": "checkpoint_nfsp.pt",
    "dmc": "checkpoint_dmc.pt",
    "dqn": "checkpoint_dqn.pt",
}


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RL agent for RLCard games.")
    parser.add_argument("--game", default="doudizhu", help="RLCard game name.")
    parser.add_argument("--agent", default="nfsp", choices=["nfsp", "dmc", "dqn"], help="Agent type to train.")
    parser.add_argument("--num_episodes", type=int, default=50000, help="Training episodes.")
    parser.add_argument("--total_frames", type=int, default=100000000, help="Total frames (DMC).")
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_num", type=int, default=2000)
    parser.add_argument("--save_dir", default=None, help="Override save dir (default: rlcard_models/{game}_{agent}).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--replay_memory_size", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--mlp_layers", type=int, nargs="+", default=[128, 128, 128],
        help="MLP hidden layer sizes.",
    )
    # DMC-specific
    parser.add_argument("--num_actors", type=int, default=5, help="DMC actor count.")
    parser.add_argument("--save_interval", type=int, default=30, help="DMC save interval (minutes).")
    parser.add_argument("--device", default=None, help="Force device (cpu, mps, cuda). Default: auto-detect.")
    parser.add_argument(
        "--player_id", type=int, default=0,
        help="Seat index to train for. Use >0 for asymmetric games (e.g. doudizhu seat 1).",
    )
    return parser.parse_args()


def _get_device(args) -> torch.device:
    """Resolve training device from --device flag or auto-detect."""
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps:0")
    return torch.device("cpu")


def _make_eval_opponent(game: str, num_actions: int):
    """Create an evaluation opponent (RandomAgent for all games)."""
    del game  # unused — kept for signature compatibility
    return RandomAgent(num_actions=num_actions)


def _create_nfsp(env, args, player_id: int = 0):
    from rlcard.agents.nfsp_agent import NFSPAgent
    device = _get_device(args)
    return NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[player_id],
        hidden_layers_sizes=list(args.mlp_layers),
        q_mlp_layers=list(args.mlp_layers),
        device=device,
    )


def _create_dqn(env, args, player_id: int = 0):
    from rlcard.agents.dqn_agent import DQNAgent
    device = _get_device(args)
    return DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[player_id],
        mlp_layers=list(args.mlp_layers),
        learning_rate=args.learning_rate,
        replay_memory_size=args.replay_memory_size,
        batch_size=args.batch_size,
        device=device,
    )


# ---------------------------------------------------------------------------
# NFSP training (standard env.run loop)
# ---------------------------------------------------------------------------

def _append_log(save_dir: Path, record: dict) -> None:
    """Append a JSON record to training_log.jsonl."""
    with open(save_dir / "training_log.jsonl", "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _train_standard(agent, agent_type: str, env, eval_env, num_players: int, args, save_dir: Path, opponents=None, player_id: int = 0):
    """Train NFSP/DQN with the standard episode loop.

    Args:
        opponents: Optional list of opponent agents. If None, uses RandomAgent.
        player_id: Seat index the learning agent occupies.
    """
    ckpt_filename = CHECKPOINT_FILENAMES[agent_type]
    if player_id > 0:
        base, ext = ckpt_filename.rsplit(".", 1)
        ckpt_filename = f"{base}_p{player_id}.{ext}"

    if opponents is None:
        opponents = [RandomAgent(num_actions=env.num_actions) for _ in range(num_players - 1)]

    # Place the learning agent at the requested seat.
    agents = list(opponents)
    agents.insert(player_id, agent)
    env.set_agents(agents)

    eval_opponents = [_make_eval_opponent(args.game, env.num_actions) for _ in range(num_players - 1)]
    eval_agents = list(eval_opponents)
    eval_agents.insert(player_id, agent)
    eval_env.set_agents(eval_agents)

    logger.info(
        "Training %s on %s (seat %d): %d episodes, %d players",
        agent_type.upper(), args.game, player_id, args.num_episodes, num_players,
    )
    best_reward = -np.inf

    for episode in range(1, args.num_episodes + 1):
        if agent_type == "nfsp":
            agent.sample_episode_policy()
        trajectories, payoffs = env.run(is_training=True)
        trajectories = reorganize(trajectories, payoffs)
        for ts in trajectories[player_id]:
            agent.feed(ts)

        if episode % args.eval_every == 0:
            reward = tournament(eval_env, args.eval_num)[player_id]
            logger.info(
                "Episode %d / %d  |  avg reward: %.4f",
                episode, args.num_episodes, reward,
            )
            _append_log(save_dir, {
                "episode": episode,
                "reward": reward,
                "best_reward": float(best_reward) if best_reward != -np.inf else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if reward > best_reward:
                best_reward = reward
                agent.save_checkpoint(str(save_dir), filename=ckpt_filename)
                logger.info("New best %.4f — saved to %s", reward, save_dir)

    # Save final checkpoint
    agent.save_checkpoint(str(save_dir), filename=ckpt_filename.replace(".pt", "_last.pt"))
    logger.info("Training complete (seat %d). Best avg reward: %.4f. Dir: %s", player_id, best_reward, save_dir)


# ---------------------------------------------------------------------------
# DMC training (uses DMCTrainer with multiprocessing)
# ---------------------------------------------------------------------------

def _train_dmc(env, args, save_dir: Path):
    """Train DMC using RLCard's built-in DMCTrainer."""
    from rlcard.agents.dmc_agent.trainer import DMCTrainer

    cuda = ""
    training_device = "cpu"
    if torch.cuda.is_available():
        cuda = "0"
        training_device = "0"

    trainer = DMCTrainer(
        env,
        cuda=cuda,
        load_model=False,
        xpid=f"{args.game}_dmc",
        save_interval=args.save_interval,
        num_actors=args.num_actors,
        training_device=training_device,
        savedir=str(save_dir),
        total_frames=args.total_frames,
        exp_epsilon=0.01,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate or 0.0001,
    )

    logger.info(
        "Training DMC on %s: %d frames, %d actors, save_dir=%s",
        args.game, args.total_frames, args.num_actors, save_dir,
    )
    trainer.start()

    # DMCTrainer.start() spawns actor processes but never terminates them.
    # Clean up to prevent the script from hanging.
    import multiprocessing as mp
    for child in mp.active_children():
        child.terminate()
    for child in mp.active_children():
        child.join(timeout=5)

    logger.info("DMC training complete. Checkpoints saved under %s", save_dir)

    # Convert DMC checkpoint to our standard format for env_wrapper loading.
    _convert_dmc_checkpoint(save_dir, args.game, env)


def _convert_dmc_checkpoint(save_dir: Path, game: str, env):
    """Convert DMCTrainer's model.tar to per-player checkpoint files.

    Saves ``checkpoint_dmc_p{pid}.pt`` for every player and keeps
    ``checkpoint_dmc.pt`` as an alias for player 0 (backward compat).
    """
    dmc_model_path = save_dir / f"{game}_dmc" / "model.tar"

    if not dmc_model_path.exists():
        logger.warning("DMC model.tar not found at %s, skipping conversion", dmc_model_path)
        return

    checkpoint = torch.load(dmc_model_path, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model_state_dict", {})

    num_players = env.num_players

    if isinstance(model_state, list):
        per_player = {pid: model_state[pid] for pid in range(len(model_state))}
    elif isinstance(model_state, dict):
        # Older format: keys prefixed with "{pid}."
        per_player = {}
        for pid in range(num_players):
            prefix = f"{pid}."
            per_player[pid] = {k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)}
    else:
        logger.warning("Unexpected model_state_dict type: %s", type(model_state))
        return

    for pid, state_dict in per_player.items():
        action_shape = env.action_shape[pid] if env.action_shape[pid] is not None else [env.num_actions]
        converted = {
            "state_shape": env.state_shape[pid],
            "action_shape": action_shape,
            "model_state_dict": state_dict,
        }
        per_player_path = save_dir / f"checkpoint_dmc_p{pid}.pt"
        torch.save(converted, per_player_path)
        logger.info("Saved DMC player %d checkpoint to %s", pid, per_player_path)

    # Backward-compat alias: checkpoint_dmc.pt = player 0
    import shutil
    p0_path = save_dir / "checkpoint_dmc_p0.pt"
    alias_path = save_dir / CHECKPOINT_FILENAMES["dmc"]
    if p0_path.exists():
        shutil.copy2(p0_path, alias_path)
        logger.info("Aliased checkpoint_dmc.pt → player 0")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    game = args.game
    agent_type = args.agent
    player_id = args.player_id

    # Validate compatibility
    compatible = TRAINABLE_AGENTS.get(game, [])
    if agent_type not in compatible:
        raise SystemExit(
            f"Agent type '{agent_type}' is not compatible with game '{game}'. "
            f"Compatible types: {compatible}"
        )

    _set_seed(args.seed)

    save_dir = Path(args.save_dir) if args.save_dir else Path(f"rlcard_models/{game}_{agent_type}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build env (use default player counts for all games)
    env_config: dict = {"seed": args.seed}
    env = rlcard.make(game, config=env_config)
    num_players = env.num_players

    if player_id >= num_players:
        raise SystemExit(
            f"--player_id {player_id} out of range for {game} ({num_players} players)"
        )

    if agent_type == "dmc":
        if player_id > 0:
            logger.warning("DMC trains all seats simultaneously; --player_id is ignored")
        _train_dmc(env, args, save_dir)
    elif agent_type == "nfsp":
        eval_env = rlcard.make(game, config={**env_config, "seed": args.seed + 1})
        agent = _create_nfsp(env, args, player_id)
        logger.info("Using device: %s", agent.device)
        _train_standard(agent, agent_type, env, eval_env, num_players, args, save_dir, opponents=None, player_id=player_id)
    else:
        # DQN
        eval_env = rlcard.make(game, config={**env_config, "seed": args.seed + 1})
        agent = _create_dqn(env, args, player_id)
        logger.info("Using device: %s", agent.device)
        _train_standard(agent, agent_type, env, eval_env, num_players, args, save_dir, opponents=None, player_id=player_id)


if __name__ == "__main__":
    main()
