#!/usr/bin/env python3
"""Generate a single task JSON file for all RLCard benchmarks.

Each entry gets a random seed_start (via system entropy), paired with num_seeds.
The actual per-task seeds are derived deterministically by expand_seeds() at runtime.

Per-game defaults (DEFAULT_NUMS) come from rlcard-3game-report.md §4.2 detection
budget: noho needs N=150 to tighten BB SE, doudizhu and mahjong both use N=30.

Usage:
    python scripts/rlcard/generate_tasks.py                          # all games, per-game defaults
    python scripts/rlcard/generate_tasks.py --num 20                 # override all games uniformly
    python scripts/rlcard/generate_tasks.py --game doudizhu --num 50 # single-game override
"""

from __future__ import annotations

import argparse
import json
import os
import random

ALL_GAMES = [
    "nolimit_holdem",
    "doudizhu",
    "mahjong",
]

DEFAULT_NUMS = {
    "nolimit_holdem": 150,
    "doudizhu": 30,
    "mahjong": 30,
}


def generate_entry(game: str, num: int, rng: random.Random) -> dict:
    return {
        "task_name": game,
        "task_type": "rlcard",
        "seed_start": rng.randint(0, 2**31 - 1),
        "num_seeds": num,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate RLCard task JSON file")
    parser.add_argument("--game", choices=ALL_GAMES, help="Single game (default: all)")
    parser.add_argument("--num", type=int, default=None,
                        help="Tasks per game (overrides per-game defaults when set)")
    parser.add_argument("--output", default=None, help="Output file (default: configs/tasks/rlcard.json)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(project_root, "configs", "tasks", "rlcard.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rng = random.SystemRandom()
    games = [args.game] if args.game else ALL_GAMES

    def _num_for(g: str) -> int:
        return args.num if args.num is not None else DEFAULT_NUMS[g]

    all_entries: list[dict] = [generate_entry(g, _num_for(g), rng) for g in games]

    for game, entry in zip(games, all_entries):
        print(f"  {game}: {entry['num_seeds']} seeds")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2)
    print(f"Generated {len(all_entries)} entries -> {output_path}")


if __name__ == "__main__":
    main()
