# Execution Flow

This document explains the full execution flow from startup to completion in chronological order.

---

## 1. Initialization

```bash
python -m anything2skill benchmark=osworld ...
```

`__main__` uniformly calls `run_parallel(cfg)` regardless of whether `num_envs` is 1 or N. The main process performs the following steps in order:

1. Hydra loads configuration (merge priority: `api.yaml` < `config.yaml` < `benchmark/*.yaml` < CLI)
2. `kit = get_kit("osworld", env_cfg)` — look up the Kit in the registry (used only for task collection)
3. Collect tasks, filter completed tasks, and put them into `Manager().Queue()`
4. Fork `num_envs` worker subprocesses; each worker creates its own `VLMClient` + `Kit` + `Env`

`num_envs=1` also uses the worker subprocess path; there is no degraded branch that runs a single env directly in the main process.

---

## 2. Task Collection

```python
tasks = kit.collect_tasks(config)
# Returns list[TaskDescriptor], each containing task_id and instruction
# In OSWorld this is OSWorldTask, which adds domain and task_config fields
```

---

## 3. Skill Loading

For each task, the runner first loads the tutorial, then extracts or reads cached skills:

1. `load_tutorial(task_id, data_dir, tutorial_type)` — reads `data_tutorial/{tutorial_type}/{bench}/{task_id}/tutorial/` and validates artifacts by `tutorial_type`: HTML must include `page*.html`, while screenshot must include a non-empty `images/` directory (and no `page.html`).
2. `get_or_extract_skills(tutorial, vlm, ..., tutorial_type=...)` — cache first:
   - **Cache hit**: read SKILL.md files from `skills_cache/{tutorial_type}/{model}/{bench}/{task_id}/` (different modalities use independent buckets and never cross-hit)
   - **Cache miss**: call the VLM for extraction (`build_skill_extraction_messages()` dispatches templates by `content_type` → `vlm.chat()` → parse markdown → write SKILL.md)
3. Return `Skills(task_id, instruction, skills=[Skill, ...])`

SKILL.md format:

```yaml
---
name: open-settings
description: Open the system settings application
images:
  - screenshot-1.png
---

## Steps
1. Click Activities
2. Search 'Settings'
3. Click Settings icon
```

---

## 4. Agent Creation

```python
agent = create_agent(agent_mode, vlm, skills, kit, agent_cfg, result_dir)
```

`agent_factory.py` dispatches by `agent_mode`:

| Mode | Class | Characteristics |
|------|------|------|
| `simple` | `SimpleAgent` | Single VLM call; all skills are in the first user turn |
| `phased` | `PhasedAgent` | Planner evaluates first → dispatch to Executor/Reflector/DONE/FAIL |
| `vanilla` | `VanillaAgent` | No-skills baseline; skips tutorial/skill loading |
| `vanilla_tutorial` | `VanillaTutorialAgent` | Does not extract skills; inserts the raw tutorial body + images directly into the first user turn |

All four receive `vlm`, `skills`, `kit`, `history_window`, and `llm_params`.
`PhasedAgent` additionally creates `SoftPlanner(vlm, kit)`.
`VanillaAgent` has empty `skills` because the runner does not load the tutorial.
`VanillaTutorialAgent` also has empty `skills`, but the runner still loads the tutorial and injects it (together with `skills.max_images` images, where `null` = unlimited) into the first user turn. This is the ablation baseline that skips skill extraction.

---

## 5. Step Loop

`run_single_task()` is the benchmark-agnostic core loop (`runner.py`):

```python
obs = env.reset(task)  # Initial observation

while not done and step_idx < max_steps:
    response, actions = agent.predict(instruction, obs)
    for action in actions:
        obs, reward, done, info = env.step(action, sleep_after_execution)
        # Save step_data to traj.jsonl
        # kit.save_observation(obs, step_obs_dir)
        if done: break
    step_idx += 1

score = env.evaluate()
```

### Internal Flow of `agent.predict()`

**SimpleAgent**:

1. `obs_content = kit.encode_observation(obs)`
2. `history = state.get_recent_history(window)`
3. `messages = msg.build_action_messages(skills, obs, history, instruction)`
4. `response = vlm.chat(messages)`
5. `actions = kit.parse_actions(response)`
6. `state.record(obs_content, response, actions[0])`

**PhasedAgent**:

1. `obs_content = kit.encode_observation(obs)`
2. `decision = planner.plan(skills, state, obs_content, instruction)` — the first VLM call, returning `PlanDecision(action, reasoning, guidance)`
3. Dispatch by `decision.action`:
   - **DONE** → return `("DONE", ["DONE"])`
   - **FAIL** → return `("FAIL", ["FAIL"])`
   - **REFLECT** → `build_reflect_messages(skills, obs, history, instruction, decision)` → VLM → parse (all skills + recent trajectory, single turn)
   - **EXECUTE** → `build_executor_messages(obs, instruction, decision)` → VLM → parse (no skills, no history, single turn)

---

## 5.5 Reviser Revision Loop (Optional)

When `reviser.max_attempts > 1`, `run_single_task()` does not return directly. Instead, it enters the loop in `ReviserRunner.run_with_reviser()`:

1. attempt_j completes → write `attempt_j/{traj.jsonl, meta.json, skills/, ...}`
2. `ReviserAnalyzer` reads the trajectory and outputs `<summary>` by `chunk_size` segments plus `<root_cause>` in the final segment
3. By default, `outcome_assessment == "likely_success"` → write `early_stop_triggered=True` to attempt_j's meta (**does not break**)
4. `ReviserRefiner` uses root_cause + tutorial + old skills to produce new skills, writing them to `attempt_{j+1}/skills/`
5. attempt_{j+1} reruns with the new skills

When `agent_mode == vanilla` or `vanilla_tutorial`, the value is clamped back to 1 because there are no skills to modify. See [`reviser-loop.md`](reviser-loop.md).

## 6. Comparison of the Four Agent Modes

### SimpleAgent (1 VLM call/step)

`encode_observation` → `build_action_messages` (skills in the first user turn) → `vlm.chat` → `parse_actions` → `state.record`

The VLM autonomously handles state perception, skill reference, progress judgment, error handling, and completion detection.

### PhasedAgent (2 VLM calls/step)

`encode_observation` → **Phase 1**: `planner.plan()` (`build_planner_messages` → VLM → JSON → `PlanDecision`) → **Phase 2**: dispatch by decision to Executor / Reflector (→ VLM → parse)

The Planner outputs JSON: `{action, reasoning, guidance}`. It does not select skills; it only provides action guidance.
The Executor sees the instruction + planner guidance + current observation (no skills, no history).
The Reflector sees all skills + recent trajectory (aggregated by step as action+obs) for comparative diagnosis.

### VanillaAgent (1 VLM call/step, no skills)

`encode_observation` → `build_vanilla_messages` (no skills) → `vlm.chat` → `parse_actions` → `state.record`

It has the same structure as SimpleAgent, but does not inject skills. The Runner skips tutorial loading and skill extraction.

### VanillaTutorialAgent (1 VLM call/step, raw tutorial instead of skills)

`encode_observation` → `build_vanilla_tutorial_messages` (the first user turn injects `## Reference Tutorial` + tutorial body + images limited by `skills.max_images`) → `vlm.chat` → `parse_actions` → `state.record`

Difference from VanillaAgent: the runner **loads the tutorial but does not extract skills** (`_load_task_skills` directly returns empty `Skills` + `TutorialMaterial`). The first user turn injects the raw tutorial body and images. The system prompt comes from `VANILLA_TUTORIAL_ACTION_SYSTEM_TMPL`, explicitly telling the model that this is RAW source material (noisy and possibly inconsistent with the current environment) and that it must select and adapt steps by itself. Later user turns are the same as SimpleAgent / VanillaAgent (only bridging + obs).

---

## 7. Result Storage

```
results/{tutorial_type}/{run_name}/{benchmark}/{domain}/{task_id}/
  task.json                # Snapshot of the task description
  attempt_1/               # canonical bucket stores only attempt_1
    traj.jsonl             # Step-by-step trajectory (step, action, response, reward, phase, planner...)
    result.txt             # Final score
    runtime.log            # Task execution log
    recording.mp4          # Screen recording (GUI benchmark)
    step_N_TIMESTAMP/      # Observation snapshot for each step
    skills/                # Skills snapshot used by this attempt
    meta.json              # {score, steps_taken, skills_count, early_stop_triggered, completed}
  attempt_2/  ...          # Only present in the reviser bucket, and additionally contains root_cause.xml
  experiment_results.json  # Bucket-level aggregation, rewritten after each attempt finishes
```

`run_name` format: `a2s-{agent_mode}-{agent_model}[-r_{reviser_model}]`, for example:
- `a2s-simple-gpt-4o` — canonical bucket (bare-agent baseline)
- `a2s-simple-gpt-4o-r_claude-sonnet-4` — reviser bucket (`attempt_2+`)

See Section 3 of [`reviser-loop.md`](reviser-loop.md).

---

## 8. Parallel / Single-Worker Execution

`run_parallel(cfg)` is the only entry point and uses `multiprocessing`:

1. The main process collects all tasks and puts them into `Manager().Queue()`
2. It starts `num_envs` worker processes (`num_envs=1` is also a subprocess and does not degrade into running inside the main process)
3. Each worker independently creates Kit + Env + VLM and pulls tasks from the queue
4. `.in_progress.lock` under the canonical bucket records the PID; after an abnormal worker exit, the next startup automatically reclaims stale locks by checking whether the PID is alive
5. Workers are automatically restarted after crashes
6. Signal handling gracefully shuts down all environments on SIGINT/SIGTERM

---

## Source File Index

| File | Responsibility |
|------|------|
| `runner.py` | `run_parallel()` (unified entry point), `run_single_task()` (per-task leaf), `_compute_run_names()`, `_effective_max_attempts()` |
| `agent_factory.py` | `create_agent()` — mode dispatch |
| `reviser/reviser_runner.py` | `ReviserRunner.run_with_reviser()` — attempt loop, dual-bucket layout, `early_stop` marking |
| `reviser/analyzer.py` | `ReviserAnalyzer` — Phase 1 (trajectory → root_cause XML) |
| `reviser/refiner.py` | `ReviserRefiner` — Phase 2 (root_cause + tutorial → new skills) |
| `metrics/tracker.py` | `ExperimentTracker` / `AttemptResult` / `TaskResult` |
| `agent/simple_agent.py` | `SimpleAgent.predict()` |
| `agent/phased_agent.py` | `PhasedAgent.predict()` |
| `agent/vanilla_agent.py` | `VanillaAgent.predict()` |
| `agent/vanilla_tutorial_agent.py` | `VanillaTutorialAgent.predict()` |
| `agent/planner.py` | `SoftPlanner.plan()` |
| `agent/state.py` | `AgentState` — trajectory history |
| `agent/message_builder.py` | `MessageBuilder` — message assembly |
| `benchmark_kit.py` | `BenchmarkKit` ABC |
| `env_base.py` | `EnvironmentInterface` ABC |
| `parser/skill_store.py` | `get_or_extract_skills()` — Skill cache |
| `parser/tutorial_loader.py` | `load_tutorial()` |
| `vlm/client.py` | `VLMClient.chat()` |
