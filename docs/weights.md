# RLCard Opponent Weights (人机)

The RLCard benchmarks play the VLM agent (seat 0) against trained RL opponents
in the other seats. Each of the three benchmark games is pinned to one opponent
type:

| Game | Opponent type | Folder | Checkpoint file(s) |
|------|---------------|--------|--------------------|
| `doudizhu` (斗地主) | **DMC** | `rlcard_models/doudizhu_dmc/` | `checkpoint_dmc.pt`, `checkpoint_dmc_p0.pt`, `checkpoint_dmc_p1.pt`, `checkpoint_dmc_p2.pt` |
| `mahjong` (麻将) | **NFSP** | `rlcard_models/mahjong_nfsp/` | `checkpoint_nfsp.pt` |
| `no-limit-holdem` (无限注德州) | **DQN** | `rlcard_models/no-limit-holdem_dqn/` | `checkpoint_dqn.pt` |

Only **doudizhu** and **mahjong** count toward the success-inferable *Strategy*
evaluation (30 + 30 = 60 hands). **No-limit hold'em is a private-information
diagnostic**: its payoff depends on opponent hole cards that may never appear in
the agent-visible trajectory, so it is reported separately rather than as a core
score. The DQN weights are still shipped so the diagnostic can be reproduced.
Opponents are **fixed** during LLM evaluation (they receive no feedback).

A `random` opponent is always available and needs **no** weights. Folder and
file naming is the contract read by
[`anything2skill/benchmarks/rlcard_common.py`](../anything2skill/benchmarks/rlcard_common.py):
`create_opponent()` looks for `rlcard_models/{game}_{type}/{checkpoint_file}`,
so the layout above must be preserved exactly.

## Why weights live in a GitHub Release

`rlcard_models/` is in `.gitignore` — the `.pt` checkpoints are too large for
the git repo. They are distributed as assets on a **GitHub Release** instead.
After a fresh `git clone`, the `rlcard_models/` folders are empty (or missing),
so you must download the weights into them before running RLCard benchmarks.

Each game folder is packed into a single `<game>_<agent>.tar.gz` asset
(e.g. `doudizhu_dmc.tar.gz`) that, when extracted into `rlcard_models/`,
recreates the folder and its `.pt` files.

## Getting the weights (download)

```bash
# Pull every weight tarball from the default release tag (weights-v1)
bash scripts/rlcard/download_weights.sh

# A specific release, or another repo
bash scripts/rlcard/download_weights.sh --tag weights-v2
bash scripts/rlcard/download_weights.sh --repo NJU-LINK/MMG2Skill

# Only one game
bash scripts/rlcard/download_weights.sh --game doudizhu_dmc
```

The script extracts each tarball back into `rlcard_models/<game>_<agent>/`.
Requires the GitHub CLI (`gh auth login`; read access is enough for a public repo).

Manual alternative — download the assets from the release page and run:

```bash
mkdir -p rlcard_models
tar xzf doudizhu_dmc.tar.gz -C rlcard_models   # → rlcard_models/doudizhu_dmc/*.pt
```

## Publishing the weights (upload, maintainers)

Train (or place) the checkpoints under `rlcard_models/` first
(`bash scripts/rlcard/setup.sh`), then:

```bash
# Pack rlcard_models/*/*.pt and upload to the default release tag (weights-v1)
bash scripts/rlcard/upload_weights.sh

# Custom tag / repo / single folder
bash scripts/rlcard/upload_weights.sh --tag weights-v2
bash scripts/rlcard/upload_weights.sh --repo NJU-LINK/MMG2Skill
bash scripts/rlcard/upload_weights.sh --dir doudizhu_dmc
```

The script creates the release if it does not exist and re-uploads assets with
`--clobber`, so re-running after retraining overwrites the old tarballs.
Requires `gh` authenticated with **push** access to the target repo.

## How the weights were trained

Opponents were trained with RLCard's official implementations and then frozen;
`scripts/rlcard/setup.sh` reproduces these defaults:

| Game / opponent | Training budget | Setup |
|-----------------|-----------------|-------|
| `doudizhu` / DMC | 10⁸ frames | Seat-specific checkpoints for the landlord and the two peasant seats (hence `checkpoint_dmc_p{0,1,2}.pt`) |
| `mahjong` / NFSP | 5×10⁴ episodes | Seat 0 is the learner, `RandomAgent` in the other seats |
| `no-limit-holdem` / DQN | 5×10⁴ episodes | Two-player heads-up |

These map directly to `setup.sh` (`TOTAL_FRAMES=100000000` for DMC,
`NUM_EPISODES=DQN_EPISODES=50000` for NFSP/DQN) and to the pinned
`(game → opponent)` table in that script.

## How the weights are selected at runtime

The opponent type is chosen per game in `configs/benchmark/<game>.yaml` via
`env.opponent_type` (`dmc | nfsp | dqn | random | auto`). With `auto`, the
loader tries trained checkpoints in the `GAME_OPPONENTS` order and falls back to
`random` if none are present. A missing checkpoint for an explicitly requested
type raises a `FileNotFoundError` pointing you back to `setup.sh` — run the
download script to fix it.

For how the opponents are trained, see `scripts/rlcard/setup.sh` and
`scripts/rlcard/train_agent.py`.
