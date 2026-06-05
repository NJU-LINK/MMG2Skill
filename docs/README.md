# MMG2Skill — Architecture & Developer Guide

Internal architecture, data flow, and developer reference for MMG2Skill. For installation and usage, see the [top-level README](../README.md). The source package name is `anything2skill`.

It currently covers **OSWorld** (GUI automation), **OpenHA Minecraft** (open-world physical interaction), and **RLCard** doudizhu / mahjong (success-inferable strategy tasks), while retaining **RLCard nolimit_holdem** as a private-information boundary diagnostic.

---

## Documentation Index

| Document | Contents | Audience |
|------|------|--------|
| [Execution Flow](execution-flow.md) | Full execution flow from startup to completion | All users |
| [Prompt Assembly Flow](prompt-assembly.md) | How Kit prompts are assembled into VLM messages | Benchmark adapters, Agent developers |
| [Benchmark Adapter Guide](benchmark-adapter-guide.md) | How to write a new `BenchmarkKit` | Benchmark adapters |
| [Reviser Revision Loop](reviser-loop.md) | Dual-bucket attempt layout, early-stop semantics, analyze→refine pipeline | Researchers running ablations or analyzing experiment data |
| [RLCard Opponent Weights](weights.md) | Pinned opponent (人机) checkpoints, GitHub Release layout, upload/download scripts | Anyone running RLCard benchmarks |

---

## Architecture

Four-layer architecture:

- **Runner** (`runner.py`) — configuration loading / task collection / Skill loading / Agent creation / step loop / dual-bucket attempt scheduling
  - **BenchmarkKit** (domain layer) — domain prompts, observation encoding, action parsing
  - **MessageBuilder** (orchestration layer) — template filling, history orchestration, message assembly
  - **Agent** (decision layer) — SimpleAgent / PhasedAgent / VanillaAgent / VanillaTutorialAgent, calls the VLM for decisions
  - **Reviser** (revision layer, optional) — analyze→refine loop between attempts, producing revised skills
  - **VLM Client** — OpenAI-compatible API calls

`agent/` and `reviser/` must not import `benchmarks.*`; all domain text is injected through `kit.*` properties.

## Data Flow

1. **Tutorial materials** live under `data_tutorial/{tutorial_type}/{benchmark}/{task_id}/tutorial/`, where `tutorial_type ∈ {html, screenshot}` (`video` is reserved): HTML tutorials are `page*.html` + `images/`; screenshot tutorials are Playwright scrolling screenshots (`images/frame_*.png`), selected via `data.tutorial_type`.
2. **Skill extraction**: the VLM reads the tutorial and outputs `SKILL.md`, cached under `skills_cache/{tutorial_type}/{model}/{benchmark}/{task_id}/`. Different modalities use independent buckets and never cross-hit; the next run with the same task and modality hits the cache directly.
3. **Agent execution**: `env.reset()` → `agent.predict(instruction, obs)` → `env.step(action)` loops until `DONE` or `max_steps`; the trajectory is written to `attempt_N/traj.jsonl`.
4. **Optional Reviser**: when `reviser.max_attempts > 1`, after each attempt the analyzer parses the trajectory and the refiner revises the skills, writes them to the next attempt directory, and continues running.

Pipeline at a glance:

```
Tutorial (HTML + images) → VLM extraction → Skills (SKILL.md) → Agent use → (optional) Reviser revision → next-round Agent use
env.reset() → obs → agent.predict() → action → env.step() → obs → loop
```

See [execution-flow.md](execution-flow.md) for the full flow.

## Experiment Artifacts

```
results/{tutorial_type}/{run_name}/{benchmark}/{domain}/{task_id}/
  task.json
  attempt_1/
    traj.jsonl, result.txt, runtime.log, recording.mp4, step_N_*/
    skills/, meta.json
  attempt_2/  attempt_3/  ...     # produced only in Reviser mode; includes root_cause.xml
  experiment_results.json         # bucket-level aggregation, rewritten after each attempt
```

`run_name` = `a2s-{agent_mode}-{agent_model}[-r_{reviser_model}]` (note: `tutorial_type` is **not** in `run_name`; the leading `results/{tutorial_type}/` partition keeps modalities apart):

- the **canonical bucket** (bare-agent baseline) stores only `attempt_1` and is reused across reviser experiments
- the **reviser bucket** stores `attempt_2+` and is isolated by reviser model

`meta.json["early_stop_triggered"]` marks the first attempt where the analyzer outputs `likely_success`; the loop does not actually break, so one run produces both early-stop and full-run views. See [reviser-loop.md](reviser-loop.md).

## Adding a New Benchmark

1. Add `anything2skill/benchmarks/<name>/{__init__.py, prompts.py, kit.py}` and add `@register_kit("<name>")` to the kit.
2. Add `configs/benchmark/<name>.yaml` (first line: `# @package _global_`).
3. Add `scripts/<name>/setup_conda.sh` to prepare an independent conda environment.
4. Validate prompt assembly with `python -m anything2skill.tests.mock_prompt_viewer`.
5. You do not need to modify any files under `agent/` or `reviser/`.

See [benchmark-adapter-guide.md](benchmark-adapter-guide.md) for the full guide.

## Core Concepts Quick Reference

| Concept | Description |
|------|------|
| **BenchmarkKit** | Domain adaptation layer. Provides `system_prompt`, `encode_observation`, `parse_actions`, and related hooks |
| **MessageBuilder** | Message orchestration layer. Assembles VLM messages; Kit developers do not interact with it |
| **SimpleAgent** | One VLM call per step; all skills are placed in the first user turn |
| **PhasedAgent** | Planner evaluation → Executor/Reflector/DONE/FAIL dispatch |
| **VanillaAgent** | No-skills baseline for ablation experiments. Reviser is disabled in this mode |
| **VanillaTutorialAgent** | Feeds the raw tutorial directly (body + images, controlled by `skills.max_images`) and skips skill extraction. Used to ablate the contribution of the extraction step itself; Reviser is also disabled |
| **SoftPlanner** | VLM-driven soft planner; does not select skills, only provides action guidance |
| **Skills** | SOPs extracted from tutorials and cached as `SKILL.md` files |
| **DONE/FAIL** | Framework-level termination signals; `env.step()` sets `done=True` when receiving them |
| **Reviser** | Inter-attempt analyze→refine pipeline that revises skills |
| **Analyzer** | Phase 1: trajectory → `<root_cause>` XML |
| **Refiner** | Phase 2: root_cause + old skills + tutorial → new skills |
| **Canonical bucket** | `a2s-{mode}-{model}` — bare-agent baseline, always stores `attempt_1` |
| **Reviser bucket** | `a2s-{mode}-{model}-r_{reviser_model}` — `attempt_2+` |
| **Attempt** | One complete task execution; multiple attempts are chained by the reviser |
| **early_stop_triggered** | `meta.json` field marking the first attempt where the analyzer outputs `likely_success`; it does not actually break the loop |
| **ExperimentTracker** | `metrics/tracker.py`, aggregates per-attempt and per-task results |
