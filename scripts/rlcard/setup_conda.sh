#!/bin/bash
# Setup conda environment for RLCard benchmarks.
# Uses Tsinghua mirror for faster downloads in China.
#
# Usage:
#     bash scripts/rlcard/setup_conda.sh
#     bash scripts/rlcard/setup_conda.sh --python 3.11

set -e

ENV_NAME="rlcard"
PYTHON_VERSION="3.12"
TUNA_CONDA="https://mirrors.tuna.tsinghua.edu.cn/anaconda"
TUNA_PIP="https://pypi.tuna.tsinghua.edu.cn/simple"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RLCARD_DIR="${REPO_ROOT}/RLCard"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python) PYTHON_VERSION="$2"; shift 2 ;;
        *)        echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ ! -d "$RLCARD_DIR" ]; then
    echo "Error: RLCard submodule not found at $RLCARD_DIR" >&2
    echo "Run: git submodule update --init RLCard" >&2
    exit 1
fi

echo "=== Removing old environment (if exists) ==="
conda env remove -n ${ENV_NAME} -y 2>/dev/null || true

echo "=== Creating conda environment (Python ${PYTHON_VERSION}) ==="
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y \
  -c ${TUNA_CONDA}/pkgs/main

echo "=== Installing PyTorch ==="
conda run -n ${ENV_NAME} pip install -i ${TUNA_PIP} torch

echo "=== Installing RLCard from submodule ==="
conda run -n ${ENV_NAME} pip install -i ${TUNA_PIP} -e "${RLCARD_DIR}[torch]"

echo "=== Installing anything2skill requirements ==="
conda run -n ${ENV_NAME} pip install -i ${TUNA_PIP} \
  -r "${REPO_ROOT}/requirements.txt"

echo "=== Verifying installation ==="
conda run -n ${ENV_NAME} bash -c "
cd '${REPO_ROOT}' && python -c '
import rlcard
print(f\"rlcard {rlcard.__version__} OK\")
import torch
print(f\"torch {torch.__version__} OK\")
import anything2skill
print(\"anything2skill OK\")
'"

echo ""
echo "=== Done! ==="
echo "Activate with:  conda activate ${ENV_NAME}"
echo "Then run:"
echo "  export PYTHON=python"
echo "  bash scripts/rlcard/setup.sh --skip-download --skip-generate"
echo "  bash scripts/rlcard/run_eval.sh --game doudizhu"
