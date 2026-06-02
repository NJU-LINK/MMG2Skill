#!/bin/bash
# Setup conda environment for Minecraft (OpenHA) benchmark
# Uses Tsinghua mirror for faster downloads in China

set -e

ENV_NAME="openha"
PYTHON_VERSION="3.10"
TUNA_CONDA="https://mirrors.tuna.tsinghua.edu.cn/anaconda"
TUNA_PIP="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "=== Removing old environment (if exists) ==="
conda env remove -n ${ENV_NAME} -y 2>/dev/null || true

echo "=== Creating conda environment ==="
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y \
  -c ${TUNA_CONDA}/pkgs/main

echo "=== Installing OpenJDK 8 ==="
conda install -n ${ENV_NAME} --channel=conda-forge openjdk=8 -y



echo "=== Installing pip dependencies ==="
conda run -n ${ENV_NAME} pip install -i ${TUNA_PIP} \
  minestudio \
  numpy==1.26.4 \
  opencv-python \
  av \
  imageio \
  gymnasium \
  minecraft_data \
  rich \
  tqdm \
  einops \
  Pillow

echo "=== Installing anything2skill requirements ==="
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
conda run -n ${ENV_NAME} pip install -i ${TUNA_PIP} -r "${REPO_ROOT}/requirements.txt"

echo "=== Verifying installation ==="
conda run -n ${ENV_NAME} python -c "
from minestudio.simulator import MinecraftSim
print('MinecraftSim import OK')
import anything2skill
print('anything2skill import OK')
"

echo "=== Done! ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Run with: xvfb-run -a conda run -n ${ENV_NAME} python -m anything2skill benchmark=minecraft"
