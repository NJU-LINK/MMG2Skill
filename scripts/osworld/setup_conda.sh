#!/bin/bash
# Setup conda environment for OSWorld benchmark
# Uses Tsinghua mirror for faster downloads in China

set -e

ENV_NAME="osworld"
PYTHON_VERSION="3.10"
TUNA_CONDA="https://mirrors.tuna.tsinghua.edu.cn/anaconda"
TUNA_PIP="https://pypi.tuna.tsinghua.edu.cn/simple"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OSWORLD_DIR="${REPO_ROOT}/OSWorld"

echo "=== Removing old environment (if exists) ==="
conda env remove -n ${ENV_NAME} -y 2>/dev/null || true

echo "=== Creating conda environment ==="
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y \
  -c ${TUNA_CONDA}/pkgs/main

echo "=== Installing OSWorld requirements ==="
conda run -n ${ENV_NAME} pip install -i ${TUNA_PIP} \
  -r "${OSWORLD_DIR}/requirements.txt"

echo "=== Installing anything2skill requirements ==="
conda run -n ${ENV_NAME} pip install -i ${TUNA_PIP} \
  -r "${REPO_ROOT}/requirements.txt"

echo "=== Verifying installation ==="
conda run -n ${ENV_NAME} bash -c "
cd '${REPO_ROOT}' && python -c '
import sys
sys.path.insert(0, \"${OSWORLD_DIR}\")
from desktop_env.desktop_env import DesktopEnv
print(\"DesktopEnv import OK\")
import anything2skill
print(\"anything2skill import OK\")
'"

echo "=== Done! ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Run with:      conda run -n ${ENV_NAME} python -m anything2skill benchmark=osworld"
