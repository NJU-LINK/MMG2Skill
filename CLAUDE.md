# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

每个 benchmark 使用独立 conda 环境（`openha` for Minecraft, `osworld` for OSWorld, `rlcard` for RLCard 卡牌游戏），先 `conda activate <env>` 再运行以下命令。首次准备环境：`bash scripts/<bench>/setup_conda.sh`。

```bash
# Run tests
python -m pytest anything2skill/tests/ -v

# Run a single test file
python -m pytest anything2skill/tests/test_agent.py -v

# OSWorld — single task / N-worker parallel
python -m anything2skill benchmark=osworld tasks.task_id=<UUID>
python -m anything2skill benchmark=osworld runner.num_envs=5

# Minecraft — single task / N-worker parallel
python -m anything2skill benchmark=minecraft tasks.task_id=<TASK_ID>
python -m anything2skill benchmark=minecraft runner.num_envs=4

# RLCard — pick a game (doudizhu/mahjong/nolimit_holdem)
python -m anything2skill benchmark=doudizhu runner.num_envs=4
python -m anything2skill benchmark=uno tasks.num_games=20

# Enable reviser refinement loop (≥2 attempts; optionally use a different reviser model)
python -m anything2skill benchmark=osworld reviser.max_attempts=3
python -m anything2skill benchmark=osworld reviser.max_attempts=3 reviser.model=claude-sonnet-4

# Or use the shell scripts (edit settings at top)
bash scripts/osworld/run.sh
bash scripts/minecraft/run.sh
bash scripts/rlcard/run_eval.sh

# Download tutorials per benchmark
python data_collection/osworld/download_tutorials.py
python data_collection/minecraft/download_tutorials.py
python data_collection/rlcard/download_tutorials.py
```

Dependencies are listed in `requirements.txt` (anything2skill) and per-submodule requirements (`OSWorld/`, `OpenHA/`, `RLCard/`).

## Architecture (see also `.claude/rules/project.md` for conventions)

### Data Flow: Tutorial → Skills → Agent Execution

1. **Tutorial materials** live under `data_tutorial/{tutorial_type}/{bench}/{task_id}/tutorial/` where `tutorial_type ∈ {html, screenshot}` (video reserved). `html`: `page*.html` + `images/` + `metadata.json`. `screenshot`: `images/frame_*.png` + `metadata.json` (must include `"content_type": "screenshot"`).
2. **Skill extraction** — `parser/skill_store.py:get_or_extract_skills()` calls the VLM to convert the tutorial into `SKILL.md` files cached at `skills_cache/{tutorial_type}/{model}/{bench}/{task_id}/{skill_name}/SKILL.md`. Subsequent runs hit the cache. Caches built from different `tutorial_type`s never collide.
3. **Agent loop** — `env.reset()` → `agent.predict(instruction, obs)` → `env.step(action)`. Inside `predict()`: `kit.encode_observation` → (PhasedAgent only) `SoftPlanner.plan` → `MessageBuilder.build_*` → `vlm.chat` → `kit.parse_actions`. Loops until `DONE` / `FAIL` / `max_steps`.
4. **(Optional) Reviser** — when `reviser.max_attempts > 1`, every completed attempt feeds analyzer + refiner to produce refined skills for the next attempt. See [`docs/reviser-loop.md`](docs/reviser-loop.md).

### Key Dependency Rule

`agent/` and `reviser/` must NEVER import from `benchmarks.*`. All domain-specific behavior enters via `BenchmarkKit` properties:

- `agent/message_builder.py` reads `kit.system_prompt`, `kit.bridging_text`, `kit.reflection_guidance`, …
- `agent/planner.py` reads `kit.planner_guidance` indirectly through `MessageBuilder`.
- `agent/prompts.py` exposes `{domain_system_prompt}` / `{domain_planner_guidance}` / `{domain_reflection_guidance}` / `{domain_guidance}` placeholders that `MessageBuilder` fills.
- `reviser/refiner.py` reads `kit.skill_extraction_guidance` and `kit.system_prompt` via `reviser/prompts.py`.

### Four Agent Modes

- **SimpleAgent** — Single VLM call per step. All skills in first user turn. VLM autonomously decides what to do (observe → reason → act/DONE). No planner.
- **PhasedAgent** — SoftPlanner evaluates state first, dispatches to Executor (act), Reflector (error recovery), or DONE. Executor receives instruction + planner guidance (no skills, no history). Reflector receives all skills + recent trajectory for diagnosis.
- **VanillaAgent** — No-skills baseline (like OSWorld PromptAgent). Same as SimpleAgent but without skills. For ablation experiments. Reviser is **disabled** in this mode (`max_attempts` clamps to 1).
- **VanillaTutorialAgent** — Skill-extraction ablation. Injects the **raw tutorial body + images** directly into the first user turn instead of going through VLM-based skill extraction. Measures the contribution of the extraction step itself. Tutorial image count is capped by top-level `skills.max_images` (`null` = no cap). Reviser is **disabled** here as well (`max_attempts` clamps to 1).

Config toggle: `agent.agent_mode: simple | phased | vanilla | vanilla_tutorial` in `configs/benchmark/<name>.yaml`.
Mode dispatch lives in `anything2skill/agent_factory.py:create_agent()`.

### Config (Hydra)

Uses Hydra with `@hydra.main()`. Config structure:

```
configs/
  config.yaml              # shared agent + reviser + skills defaults
  api/default.yaml         # API credentials (gitignored)
  benchmark/osworld.yaml   # benchmark-specific (@package _global_)
  benchmark/minecraft.yaml
  benchmark/<game>.yaml    # one per RLCard game
  tasks/                   # optional task-filter snippets
```

Top-level config sections:
- `agent.*` — model / max_tokens / temperature / history_window / agent_mode
- `reviser.*` — max_attempts, model (null = same as agent), analysis/refine temperature & token budgets, history_in_refine, tutorial_image_cap, rolling_summary_char_limit
- `skills.*` — cache_dir, max_images, force_regenerate
- `data.tutorial_dir`, `data.tutorial_type` (`html` / `screenshot` / `video`-reserved), `env.*`, `runner.*`, `tasks.*`

Merge priority: `api/default.yaml` < `config.yaml` < `benchmark/*.yaml` < CLI overrides.
CLI overrides use Hydra syntax: `agent.model=gpt-4o-mini agent.max_steps=20 benchmark=osworld reviser.max_attempts=3`.

### Skill Cache

`get_or_extract_skills()` in `parser/skill_store.py` is the main entry point. It checks `skills_cache/{model}/{bench}/{task_id}/` for cached SKILL.md files before calling VLM extraction. Use `skills.force_regenerate=true` CLI override to re-extract.

### Key Files

- `anything2skill/benchmark_kit.py` — BenchmarkKit ABC (the contract for new benchmarks)
- `anything2skill/env_base.py` — EnvironmentInterface ABC (reset/step/evaluate/close)
- `anything2skill/runner.py` — `run_parallel()` entry + `run_single_task()` per-task leaf + `_compute_run_names()` / `_effective_max_attempts()`
- `anything2skill/agent_factory.py` — `create_agent()` mode dispatch
- `anything2skill/reviser/reviser_runner.py` — attempt 1..N orchestration, dual-bucket layout, early-stop tagging
- `anything2skill/reviser/analyzer.py` — `ReviserAnalyzer` (trajectory → `<root_cause>` XML)
- `anything2skill/reviser/refiner.py` — `ReviserRefiner` (XML + tutorial → refined `Skills`)
- `anything2skill/metrics/tracker.py` — `ExperimentTracker` / `AttemptResult` / `TaskResult`
- `anything2skill/benchmarks/osworld/env_wrapper.py` — `OSWorldEnvWrapper`
- `anything2skill/benchmarks/minecraft/kit.py` — `MinecraftKit` (`<action>...</action>` tags)
- `anything2skill/benchmarks/rlcard_common.py` — `BaseRLCardEnvWrapper` + `GAME_OPPONENTS`, shared by all 9 card games

### Benchmark Integrations

- **OSWorld** — `OSWorld/` is a read-only git submodule. `OSWorldEnvWrapper` wraps OSWorld's `DesktopEnv` for VM management, absorbing GUI-specific behavior (post-reset stabilization wait, pre-evaluate wait, screen recording). Action format: ```python``` blocks (pyautogui).
- **Minecraft** — `OpenHA/` submodule provides the env backend. `MinecraftKit` uses `<action>...</action>` XML tags (thinking-model friendly) and exposes `WAIT` as a domain-specific signal alongside framework `DONE` / `FAIL`.
- **RLCard** — `RLCard/` submodule. The 3 game kits (`doudizhu`, `mahjong`, `nolimit_holdem`) share `benchmarks/rlcard_common.py:BaseRLCardEnvWrapper` for env lifecycle and `GAME_OPPONENTS` for opponent checkpoint config. Action format: `<action>...</action>` tags.

All benchmarks go through `runner.run_parallel()` — `num_envs=1` still runs in a worker subprocess. Per-task execution leaf is `run_single_task()`. Agents implement `predict(instruction, obs) -> (response, actions)`.

### Result Directory Layout

```
results/{tutorial_type}/{run_name}/{benchmark}/{domain}/{task_id}/
  task.json                      # task descriptor snapshot
  attempt_1/
    traj.jsonl                   # unified trajectory (step, action, response, reward, phase, planner info)
    result.txt                   # final score
    runtime.log                  # per-task execution log
    recording.mp4                # screen recording (GUI benchmarks only)
    step_N_TIMESTAMP/            # per-step observations (screenshots, etc.)
    meta.json                    # {score, steps_taken, skills_count, early_stop_triggered}
    skills/                      # snapshot of skills used this attempt
    root_cause.xml               # (attempt_2+ only) analyzer output that led to this attempt
  attempt_2/  attempt_3/  ...    # only present when reviser.max_attempts > 1
```

`run_name` is `a2s-{agent_mode}-{agent_model}[-r_{reviser_model}]` — note that `tutorial_type` is **not** encoded in `run_name`; the leading `results/{tutorial_type}/` partition keeps modalities apart instead, so notebooks can group by `(tutorial_type, model, mode)` without parsing run_name suffixes.
- **Canonical bucket** `a2s-{mode}-{model}` — pure agent baseline; always holds `attempt_1`. Read-only once written, shared across reviser variants.
- **Reviser bucket** `a2s-{mode}-{model}-r_{reviser_model}` — created when `max_attempts > 1`, even if the reviser model equals the agent model. Holds `attempt_2+`.

`meta.json["early_stop_triggered"]` is set on the first attempt whose successor analyzer produces empty `<issues>`. The loop does **not** break — it force-continues to `max_attempts` so one run yields both an early-stop view (score of that single attempt) and a full-run view (all attempts). See [`docs/reviser-loop.md`](docs/reviser-loop.md) for details.
