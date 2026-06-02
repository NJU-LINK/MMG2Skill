#!/usr/bin/env bash
# Run Anything2Skill benchmark evaluation on RLCard games using the Qwen /
# dashscope endpoint (configs/api/qwen.yaml). Independent of run_eval.sh so the
# main script stays clean.
#
# Multi-game cascade (default when no --game is given):
#   Launches all 3 games concurrently with 5:5:10 initial worker split.
#   As each game finishes, a sibling instance of the next game is launched
#   with the freed workers. These siblings piggy-back on the per-task
#   .in_progress.lock mechanism to avoid duplicate work.
#
# Usage:
#     bash scripts/rlcard/run_eval_qwen.sh
#     bash scripts/rlcard/run_eval_qwen.sh --game doudizhu
#     bash scripts/rlcard/run_eval_qwen.sh --mode vanilla_tutorial
set -euo pipefail

cd "$(cd "$(dirname "$0")/../.." && pwd)"

# ---- Defaults ----
GAME=""
PYTHON="${PYTHON:-uv run python}"
MODEL="kimi-k2.6"
AGENT_MODE="simple"
TUTORIAL_TYPE="html"
MAX_ATTEMPTS=5
NUM_ENVS=20
PARALLEL=true
FORCE_REGENERATE=false
TASK_FILE=""
# -------------------

# Step ratios for parallel allocation (from empirical avg_steps)
# nolimit_holdem: ~2.35, doudizhu: ~14, mahjong: ~23.05
# Cascade order = ALL_GAMES order; doudizhu first so its workers migrate earliest.
_game_ratio() {
    case "$1" in
        nolimit_holdem) echo 10 ;;
        doudizhu)       echo 5 ;;
        mahjong)        echo 5 ;;
        *)              echo 1 ;;
    esac
}
ALL_GAMES="doudizhu nolimit_holdem mahjong"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [GAME]

Run evaluation via Qwen / dashscope endpoint (api=qwen).

Options:
  --game GAME           Game name (default: all 3 games)
  --model MODEL         VLM model (default: $MODEL)
  --mode MODE           Agent mode: simple|phased|vanilla|vanilla_tutorial (default: $AGENT_MODE)
  --tutorial-type TYPE  Tutorial modality: html|screenshot (default: $TUTORIAL_TYPE)
  --max-attempts N      Reviser attempts per task (default: $MAX_ATTEMPTS)
  --num-envs N          Parallel evaluation workers (default: $NUM_ENVS)
  --parallel            No-op: cascade is default for multi-game
  --sequential          Disable cascade, run games sequentially
  --task-file PATH      Use specific task file (default: auto-detect)
  --force-regenerate    Force re-extract skills
  -h, --help            Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --game)             GAME="$2"; shift 2 ;;
        --model)            MODEL="$2"; shift 2 ;;
        --mode)             AGENT_MODE="$2"; shift 2 ;;
        --tutorial-type)    TUTORIAL_TYPE="$2"; shift 2 ;;
        --max-attempts)     MAX_ATTEMPTS="$2"; shift 2 ;;
        --num-envs)         NUM_ENVS="$2"; shift 2 ;;
        --parallel)         :; shift ;;
        --sequential)       PARALLEL=false; shift ;;
        --task-file)        TASK_FILE="$2"; shift 2 ;;
        --force-regenerate) FORCE_REGENERATE=true; shift ;;
        -h|--help)          usage ;;
        -*)                 echo "Unknown option: $1" >&2; exit 1 ;;
        *)                  GAME="$1"; shift ;;
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
    html|screenshot) ;;
    *)
        echo "Error: --tutorial-type must be html or screenshot (got '$TUTORIAL_TYPE')." >&2
        exit 1
        ;;
esac

if { [ "$AGENT_MODE" = "vanilla" ] || [ "$AGENT_MODE" = "vanilla_tutorial" ]; } \
   && [ "$MAX_ATTEMPTS" -gt 1 ]; then
    echo "Warning: $AGENT_MODE mode does not support reviser (no skills); clamping --max-attempts to 1." >&2
    MAX_ATTEMPTS=1
fi

if [ -z "$TASK_FILE" ]; then
    TASK_FILE="configs/tasks/rlcard.json"
    if [ ! -f "$TASK_FILE" ]; then
        echo "Error: task file not found: $TASK_FILE" >&2
        echo "Run 'bash scripts/rlcard/setup.sh' first to generate it." >&2
        exit 1
    fi
fi

echo "╔═══════════════════════════════════════════════╗"
echo "║       RLCard Evaluation (Qwen endpoint)       ║"
echo "╠═══════════════════════════════════════════════╣"
echo "║  API:         qwen (configs/api/qwen.yaml)"
echo "║  Games:       $(echo $GAMES | wc -w | tr -d ' ') game(s)"
echo "║  Model:       $MODEL"
echo "║  Mode:        $AGENT_MODE"
echo "║  Tutorial:    $TUTORIAL_TYPE"
echo "║  Max attempts: $MAX_ATTEMPTS"
echo "║  Task file:   $TASK_FILE"
if [ "$PARALLEL" = "true" ] && [ "$(echo $GAMES | wc -w)" -gt 1 ]; then
    echo "║  Parallel:    cascade (next game inherits finished game's workers)"
fi
echo "╚═══════════════════════════════════════════════╝"
echo ""

allocate_workers() {
    local total_envs="$1"; shift
    local games="$*"
    local ratio_sum=0
    for g in $games; do ratio_sum=$(( ratio_sum + $(_game_ratio "$g") )); done
    local assigned=0
    local max_ratio=0
    local max_game=""
    local tmp=""
    for g in $games; do
        local r=$(_game_ratio "$g")
        local w=$(( total_envs * r / ratio_sum ))
        [ "$w" -lt 1 ] && w=1
        tmp="$tmp$g:$w "
        assigned=$(( assigned + w ))
        if [ "$r" -gt "$max_ratio" ]; then max_ratio=$r; max_game=$g; fi
    done
    local remainder=$(( total_envs - assigned ))
    for pair in $tmp; do
        local g="${pair%:*}"
        local w="${pair#*:}"
        if [ "$g" = "$max_game" ] && [ "$remainder" -gt 0 ]; then
            w=$(( w + remainder ))
        fi
        echo "$g $w"
    done
}

run_one_game() {
    local game="$1"
    local workers="$2"
    $PYTHON -m anything2skill \
        benchmark="$game" \
        api=qwen \
        agent.model="$MODEL" \
        agent.agent_mode="$AGENT_MODE" \
        agent.history_window=60 \
        data.tutorial_type="$TUTORIAL_TYPE" \
        runner.num_envs="$workers" \
        reviser.max_attempts="$MAX_ATTEMPTS" \
        tasks.task_file="$TASK_FILE" \
        skills.force_regenerate="$FORCE_REGENERATE"
}

if [ "$PARALLEL" = "true" ] && [ "$(echo $GAMES | wc -w)" -gt 1 ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_DIR="logs/_parallel_${MODEL}_${TIMESTAMP}"
    mkdir -p "$LOG_DIR"

    GAMES_ARR=($GAMES)
    NUM_GAMES=${#GAMES_ARR[@]}

    GAME_WORKERS=()
    MAIN_PID=()
    ASSIGNED=()
    ALLOC=$(allocate_workers "$NUM_ENVS" $GAMES)
    while read -r g w; do
        for i in $(seq 0 $(( NUM_GAMES - 1 ))); do
            if [ "${GAMES_ARR[$i]}" = "$g" ]; then
                GAME_WORKERS[$i]=$w
                ASSIGNED[$i]=$w
                break
            fi
        done
    done <<< "$ALLOC"

    echo "--- Initial allocation (total=$NUM_ENVS, cascade on completion) ---"
    echo "--- Logs: $LOG_DIR ; output also mirrored to this terminal ---"
    for i in $(seq 0 $(( NUM_GAMES - 1 ))); do
        g=${GAMES_ARR[$i]}
        w=${GAME_WORKERS[$i]}
        LOG="$LOG_DIR/${g}.log"
        echo "→ launching $g ($w workers)"
        ( run_one_game "$g" "$w" 2>&1 | tee -a "$LOG" ) &
        MAIN_PID[$i]=$!
    done
    echo ""

    FAIL=0
    SIBLING_PIDS=()

    for i in $(seq 0 $(( NUM_GAMES - 1 ))); do
        g=${GAMES_ARR[$i]}
        wait "${MAIN_PID[$i]}" || FAIL=$(( FAIL + 1 ))
        next_idx=$(( i + 1 ))
        if [ "$next_idx" -ge "$NUM_GAMES" ]; then
            continue
        fi
        next_g=${GAMES_ARR[$next_idx]}
        boost=${ASSIGNED[$i]}
        ASSIGNED[$next_idx]=$(( ASSIGNED[$next_idx] + boost ))
        LOG="$LOG_DIR/${next_g}_boost_from_${g}.log"
        echo "✓ $g finished → boosting $next_g with +$boost workers (total=${ASSIGNED[$next_idx]})"
        ( run_one_game "$next_g" "$boost" 2>&1 | tee -a "$LOG" ) &
        SIBLING_PIDS+=($!)
    done

    for pid in "${SIBLING_PIDS[@]}"; do
        wait "$pid" || FAIL=$(( FAIL + 1 ))
    done

    echo ""
    if [ "$FAIL" -gt 0 ]; then
        echo "⚠ $FAIL process(es) failed. Check logs in $LOG_DIR"
        exit 1
    fi
else
    for G in $GAMES; do
        echo "========================================="
        echo "  Evaluating (qwen): $G"
        echo "========================================="
        run_one_game "$G" "$NUM_ENVS"
    done
fi

echo ""
echo "╔═══════════════════════════════════════════════╗"
echo "║  Evaluation complete! (qwen endpoint)         ║"
echo "╚═══════════════════════════════════════════════╝"
