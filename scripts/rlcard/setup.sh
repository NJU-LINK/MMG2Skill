#!/usr/bin/env bash
# Setup everything needed for RLCard benchmarks:
#   1. Download tutorials
#   2. Train pinned RL opponents (doudizhu→dmc, mahjong→nfsp, nolimit_holdem→dqn)
#   3. Generate task JSON (random seeds for evaluation)
#
# Usage:
#     bash scripts/rlcard/setup.sh                         # all 3 games, full setup
#     bash scripts/rlcard/setup.sh --game doudizhu         # single game
#     bash scripts/rlcard/setup.sh --skip-train            # skip training
set -euo pipefail

cd "$(cd "$(dirname "$0")/../.." && pwd)"

# ---- Settings ----
GAME=""
PYTHON="${PYTHON:-uv run python}"
NUM_EPISODES=50000       # NFSP
DQN_EPISODES=50000       # DQN
EVAL_EVERY=100
TOTAL_FRAMES=100000000   # DMC
SEED=42
NUM_TASKS=""
DEVICE=""
TUTORIAL_TYPE="both"
SKIP_DOWNLOAD=false
SKIP_TRAIN=false
SKIP_GENERATE=false
# -------------------

ALL_GAMES="nolimit_holdem doudizhu mahjong"

# Pinned (game → opponent) combos used by the paper.
_pinned_opponent() {
    case "$1" in
        doudizhu)       echo "dmc" ;;
        mahjong)        echo "nfsp" ;;
        nolimit_holdem) echo "dqn" ;;
        *)              echo "" ;;
    esac
}

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Setup RLCard benchmarks: download tutorials, train pinned opponents, generate tasks.

Step 1: Download tutorials (data_collection)
Step 2: Train pinned opponents only
          doudizhu       → dmc  (handles all 3 seats internally)
          mahjong        → nfsp (pid=0)
          nolimit_holdem → dqn  (pid=0)
Step 3: Generate task JSON with random seeds

Options:
  --game GAME           Setup only this game (default: all 3 games)
  --skip-download       Skip tutorial download
  --skip-train          Skip opponent training
  --skip-generate       Skip task JSON generation
  --num-tasks N         Tasks per game for generation (overrides per-game
                        defaults in generate_tasks.py:DEFAULT_NUMS
                        [noho=150, dd=30, mj=30])
  --num-episodes N      NFSP training episodes (default: $NUM_EPISODES)
  --dqn-episodes N      DQN episodes (default: $DQN_EPISODES)
  --eval-every N        Evaluate every N episodes (default: $EVAL_EVERY)
  --total-frames N      DMC training frames (default: $TOTAL_FRAMES)
  --seed N              Training seed (default: $SEED)
  --device DEVICE       Training device: cpu, cuda, mps (default: auto-detect)
  --tutorial-type TYPE  Tutorial modality: html|screenshot|both (default: $TUTORIAL_TYPE)
                        html       → download_tutorials.py
                        screenshot → download_via_playwright.py
                        both       → run both sequentially
  -h, --help            Show this help

Examples:
  $(basename "$0")                          # full setup, all 3 games
  $(basename "$0") --game doudizhu          # single game
  $(basename "$0") --skip-train             # only download + generate
  $(basename "$0") --skip-download --skip-generate  # only train
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --game)          GAME="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --skip-train)    SKIP_TRAIN=true; shift ;;
        --skip-generate) SKIP_GENERATE=true; shift ;;
        --num-tasks)     NUM_TASKS="$2"; shift 2 ;;
        --num-episodes)  NUM_EPISODES="$2"; shift 2 ;;
        --dqn-episodes)  DQN_EPISODES="$2"; shift 2 ;;
        --eval-every)    EVAL_EVERY="$2"; shift 2 ;;
        --total-frames)  TOTAL_FRAMES="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --device)        DEVICE="$2"; shift 2 ;;
        --tutorial-type) TUTORIAL_TYPE="$2"; shift 2 ;;
        -h|--help)       usage ;;
        *)               echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ -n "$GAME" ]; then
    if ! echo "$ALL_GAMES" | grep -qw "$GAME"; then
        echo "Error: unknown game '$GAME'." >&2
        echo "Available: $ALL_GAMES" >&2
        exit 1
    fi
    GAMES="$GAME"
else
    GAMES="$ALL_GAMES"
fi

case "$TUTORIAL_TYPE" in
    html|screenshot|both) ;;
    *)
        echo "Error: --tutorial-type must be html|screenshot|both (got '$TUTORIAL_TYPE')." >&2
        exit 1
        ;;
esac

echo "╔═══════════════════════════════════════════════╗"
echo "║       RLCard Setup                            ║"
echo "╠═══════════════════════════════════════════════╣"
echo "║  Games: $(echo $GAMES | wc -w | tr -d ' ') game(s)"
echo "║  Download: $( [ "$SKIP_DOWNLOAD" = true ] && echo 'skip' || echo "yes ($TUTORIAL_TYPE)" )"
echo "║  Train:    $( [ "$SKIP_TRAIN" = true ] && echo 'skip' || echo 'pinned opponents' )"
echo "║  Generate: $( [ "$SKIP_GENERATE" = true ] && echo 'skip' || { [ -n "$NUM_TASKS" ] && echo "$NUM_TASKS tasks/game (override)" || echo 'per-game defaults (noho=150 dd=30 mj=30)'; } )"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# ═══════════════════════════════════════════════════
# Lookup helpers (bash 3.2 compatible)
# ═══════════════════════════════════════════════════
_rlcard_id() {
    case "$1" in
        nolimit_holdem) echo "no-limit-holdem" ;;
        *)              echo "$1" ;;
    esac
}

# ═══════════════════════════════════════════════════
# Step 1: Download tutorials
# ═══════════════════════════════════════════════════
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "━━━ Step 1/3: Downloading tutorials ($TUTORIAL_TYPE) ━━━"
    for G in $GAMES; do
        if [ "$TUTORIAL_TYPE" = "html" ] || [ "$TUTORIAL_TYPE" = "both" ]; then
            echo "  [html] Downloading tutorials for $G..."
            $PYTHON data_collection/rlcard/download_tutorials.py --game "$G"
        fi
        if [ "$TUTORIAL_TYPE" = "screenshot" ] || [ "$TUTORIAL_TYPE" = "both" ]; then
            echo "  [screenshot] Capturing tutorials for $G..."
            $PYTHON data_collection/rlcard/download_via_playwright.py --game "$G"
        fi
    done
    echo ""
else
    echo "━━━ Step 1/3: Skipped (--skip-download) ━━━"
    echo ""
fi

# ═══════════════════════════════════════════════════
# Step 2: Train pinned opponents
# ═══════════════════════════════════════════════════
if [ "$SKIP_TRAIN" = false ]; then
    echo "━━━ Step 2/3: Training pinned opponents ━━━"
    echo "  NFSP: $NUM_EPISODES eps   DQN: $DQN_EPISODES eps   DMC: $TOTAL_FRAMES frames"
    echo "  eval_every: $EVAL_EVERY"
    echo ""

    TOTAL_FAILED=0
    TRAIN_PIDS=()
    TRAIN_LABELS=()

    for G in $GAMES; do
        AGENT="$(_pinned_opponent "$G")"
        RLCARD_ID="$(_rlcard_id "$G")"

        SAVE_DIR="rlcard_models/${RLCARD_ID}_${AGENT}"

        DEVICE_ARGS=""
        if [ -n "$DEVICE" ]; then
            DEVICE_ARGS="--device $DEVICE"
        fi

        case "$AGENT" in
            dmc)
                CKPT_FILE="checkpoint_dmc.pt"
                if [ -f "$SAVE_DIR/$CKPT_FILE" ]; then
                    echo "  $G/dmc: checkpoint exists, skipping"
                    continue
                fi
                mkdir -p "$SAVE_DIR"
                LOG_FILE="$SAVE_DIR/train.log"
                echo "  Launching $G/dmc ($TOTAL_FRAMES frames) → $LOG_FILE"
                $PYTHON scripts/rlcard/train_agent.py \
                    --game "$RLCARD_ID" \
                    --agent dmc \
                    --save_dir "$SAVE_DIR" \
                    --seed "$SEED" \
                    --total_frames "$TOTAL_FRAMES" \
                    $DEVICE_ARGS \
                    > "$LOG_FILE" 2>&1 &
                TRAIN_PIDS+=($!)
                TRAIN_LABELS+=("$G/dmc")
                ;;
            nfsp|dqn)
                CKPT_FILE="checkpoint_${AGENT}.pt"
                if [ -f "$SAVE_DIR/$CKPT_FILE" ]; then
                    echo "  $G/$AGENT: checkpoint exists, skipping"
                    continue
                fi
                case "$AGENT" in
                    dqn) EP=$DQN_EPISODES ;;
                    *)   EP=$NUM_EPISODES ;;
                esac
                mkdir -p "$SAVE_DIR"
                LOG_FILE="$SAVE_DIR/train_p0.log"
                echo "  Launching $G/$AGENT ($EP eps) → $LOG_FILE"
                $PYTHON scripts/rlcard/train_agent.py \
                    --game "$RLCARD_ID" \
                    --agent "$AGENT" \
                    --save_dir "$SAVE_DIR" \
                    --seed "$SEED" \
                    --num_episodes "$EP" \
                    --eval_every "$EVAL_EVERY" \
                    --player_id 0 \
                    $DEVICE_ARGS \
                    > "$LOG_FILE" 2>&1 &
                TRAIN_PIDS+=($!)
                TRAIN_LABELS+=("$G/$AGENT")
                ;;
            *)
                echo "  Error: no pinned opponent for $G" >&2
                TOTAL_FAILED=$((TOTAL_FAILED + 1))
                ;;
        esac
    done

    if [ "${#TRAIN_PIDS[@]}" -gt 0 ]; then
        echo "  ${#TRAIN_PIDS[@]} training jobs launched, waiting..."
        for i in "${!TRAIN_PIDS[@]}"; do
            if ! wait "${TRAIN_PIDS[$i]}"; then
                echo "    FAILED: ${TRAIN_LABELS[$i]}"
                TOTAL_FAILED=$((TOTAL_FAILED + 1))
            else
                echo "    done: ${TRAIN_LABELS[$i]}"
            fi
        done
    fi

    echo "  All training complete ($TOTAL_FAILED failures)."
    echo ""
else
    echo "━━━ Step 2/3: Skipped (--skip-train) ━━━"
    echo ""
fi

# ═══════════════════════════════════════════════════
# Step 3: Generate task JSON
# ═══════════════════════════════════════════════════
if [ "$SKIP_GENERATE" = false ]; then
    echo "━━━ Step 3/3: Generating task JSON ━━━"
    GENERATE_ARGS=""
    if [ -n "$NUM_TASKS" ]; then
        GENERATE_ARGS="--num $NUM_TASKS"
    fi
    if [ -n "$GAME" ]; then
        GENERATE_ARGS="$GENERATE_ARGS --game $GAME"
    fi
    $PYTHON scripts/rlcard/generate_tasks.py $GENERATE_ARGS
    echo ""
else
    echo "━━━ Step 3/3: Skipped (--skip-generate) ━━━"
    echo ""
fi

echo "╔═══════════════════════════════════════════════╗"
echo "║  Setup complete!                              ║"
echo "║  Next: bash scripts/rlcard/run_eval.sh        ║"
echo "╚═══════════════════════════════════════════════╝"
