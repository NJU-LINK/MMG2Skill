"""Unified entry point for all benchmarks.

Usage:
    python -m anything2skill benchmark=osworld
    python -m anything2skill benchmark=osworld tasks.task_id=UUID
    python -m anything2skill benchmark=osworld runner.num_envs=5
"""

from __future__ import annotations

import os
import sys

# macOS: Homebrew libraries (cairo etc.) are not on the default dyld search path.
if sys.platform == "darwin":
    _brew_lib = "/opt/homebrew/lib"
    if os.path.isdir(_brew_lib):
        os.environ["DYLD_LIBRARY_PATH"] = (
            _brew_lib + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
        )

import hydra
from omegaconf import DictConfig

from anything2skill.runner import run_parallel


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_parallel(cfg)


if __name__ == "__main__":
    main()
