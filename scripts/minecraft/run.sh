#!/usr/bin/env bash
# Run Anything2Skill on Minecraft benchmark.
set -euo pipefail

cd "$(cd "$(dirname "$0")/../.." && pwd)"

# ---- Settings (edit here) ----
ENV_NAME="openha"
NUM_ENVS=10
# MODEL="qwen3.6-plus"
MODEL="claude-opus-4-6-20260205"
AGENT_MODE="simple"
MAX_STEPS=60
TASK_ID=""          # e.g. "mine_block:dirt_zero", empty = all tasks
TASK_TYPE=""        # e.g. "craft_item", empty = all types
FORCE_REGENERATE=false
MAX_ATTEMPTS=1      # 1 = bare run (no reviser); >=2 = enable reviser loop
REVISER_MODEL=""    # empty = reuse agent model; otherwise a distinct reviser LLM
# -------------------------------

# Auto-activate conda env so `python` resolves to ${ENV_NAME}'s interpreter
# even in non-interactive shells (cron, CI, nohup). Temporarily relax
# `set -u` because conda's activate.d hooks (e.g. openha's java_home.sh)
# dereference unset vars like JAVA_HOME without guarding.
# shellcheck disable=SC1091
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

EXTRA_ARGS=()
[ -n "$TASK_ID" ]       && EXTRA_ARGS+=(tasks.task_id="$TASK_ID")
[ -n "$TASK_TYPE" ]     && EXTRA_ARGS+=(tasks.task_type="$TASK_TYPE")
[ -n "$REVISER_MODEL" ] && EXTRA_ARGS+=(reviser.model="$REVISER_MODEL")

# shellcheck disable=SC1091
# Redirect Minecraft client stdout (mc_*.log) into logs/minecraft/
export MALMO_MINECRAFT_OUTPUT_LOGDIR="logs/minecraft"
mkdir -p "$MALMO_MINECRAFT_OUTPUT_LOGDIR"

xvfb-run -a python -m anything2skill \
    benchmark=minecraft \
    runner.num_envs="$NUM_ENVS" \
    agent.model="$MODEL" \
    agent.agent_mode="$AGENT_MODE" \
    agent.max_steps="$MAX_STEPS" \
    skills.force_regenerate="$FORCE_REGENERATE" \
    reviser.max_attempts="$MAX_ATTEMPTS" \
    "${EXTRA_ARGS[@]}" \
    "$@"
