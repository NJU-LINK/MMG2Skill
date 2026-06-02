#!/usr/bin/env bash
# Run Anything2Skill on OSWorld "all" domain.
set -euo pipefail

cd "$(cd "$(dirname "$0")/../.." && pwd)"

# ---- Settings (edit here) ----
ENV_NAME="osworld"
NUM_ENVS=5
DOMAIN=all
# MODEL="qwen3.6-plus"
MODEL="claude-opus-4-6-20260205"
AGENT_MODE="simple"
MAX_STEPS=15
FORCE_REGENERATE=false
MAX_ATTEMPTS=1      # 1 = bare run (no reviser); >=2 = enable reviser loop
REVISER_MODEL=""    # empty = reuse agent model; otherwise a distinct reviser LLM
# -------------------------------

# Auto-activate conda env so `python` resolves to ${ENV_NAME}'s interpreter
# even in non-interactive shells (cron, CI, nohup).
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

EXTRA_ARGS=()
[ -n "$REVISER_MODEL" ] && EXTRA_ARGS+=(reviser.model="$REVISER_MODEL")

python -m anything2skill \
    benchmark=osworld \
    tasks.domain="$DOMAIN" \
    runner.num_envs="$NUM_ENVS" \
    agent.model="$MODEL" \
    agent.agent_mode="$AGENT_MODE" \
    agent.max_steps="$MAX_STEPS" \
    skills.force_regenerate="$FORCE_REGENERATE" \
    reviser.max_attempts="$MAX_ATTEMPTS" \
    "${EXTRA_ARGS[@]}"
