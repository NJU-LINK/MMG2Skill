# 运行流程

本文档按时间线讲解从启动到结束的完整执行流程。

---

## 1. 初始化

```bash
python -m anything2skill benchmark=osworld ...
```

`__main__` 统一调用 `run_parallel(cfg)`（无论 `num_envs` 是 1 还是 N），主进程依次完成：

1. Hydra 加载配置（合并优先级：`api.yaml` < `config.yaml` < `benchmark/*.yaml` < CLI）
2. `kit = get_kit("osworld", env_cfg)` — 注册表查找 Kit（仅用于任务收集）
3. 收集任务、过滤已完成、放入 `Manager().Queue()`
4. fork `num_envs` 个 worker 子进程；每个 worker 自行创建 `VLMClient` + `Kit` + `Env`

`num_envs=1` 也走 worker 子进程路径 —— 没有"单 env 直跑"的退化分支。

---

## 2. 任务收集

```python
tasks = kit.collect_tasks(config)
# 返回 list[TaskDescriptor]，每个包含 task_id 和 instruction
# OSWorld 中为 OSWorldTask（增加 domain、task_config 字段）
```

---

## 3. Skill 加载

对每个 task，先加载教程，再提取或读取缓存的 skills：

1. `load_tutorial(task_id, data_dir, tutorial_type)` — 读取 `data_tutorial/{tutorial_type}/{bench}/{task_id}/tutorial/`，按 `tutorial_type` 校验产物：html 必须有 `page*.html`，screenshot 必须有非空 `images/`（无 page.html）
2. `get_or_extract_skills(tutorial, vlm, ..., tutorial_type=...)` — 缓存优先：
   - **有缓存**：从 `skills_cache/{tutorial_type}/{model}/{bench}/{task_id}/` 读取 SKILL.md 文件（不同模态独立分桶，互不命中）
   - **无缓存**：调用 VLM 抽取（`build_skill_extraction_messages()` 按 `content_type` 分发模板 → `vlm.chat()` → 解析 markdown → 写入 SKILL.md）
3. 返回 `Skills(task_id, instruction, skills=[Skill, ...])`

SKILL.md 格式：

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

## 4. Agent 创建

```python
agent = create_agent(agent_mode, vlm, skills, kit, agent_cfg, result_dir)
```

`agent_factory.py` 根据 `agent_mode` 分发：

| 模式 | 类 | 特点 |
|------|------|------|
| `simple` | `SimpleAgent` | 单次 VLM 调用，所有 skills 在第一轮 user turn |
| `phased` | `PhasedAgent` | Planner 先评估 → Executor/Reflector/DONE/FAIL 分发 |
| `vanilla` | `VanillaAgent` | 无 skills baseline，跳过 tutorial/skill 加载 |
| `vanilla_tutorial` | `VanillaTutorialAgent` | 不抽取 skill，直接把原始教程 body + 图片塞进第一轮 user turn |

四者都接收 `vlm`、`skills`、`kit`、`history_window`、`llm_params`。
PhasedAgent 额外创建 `SoftPlanner(vlm, kit)`。
VanillaAgent 的 `skills` 为空（runner 不加载 tutorial）。
VanillaTutorialAgent 的 `skills` 也为空，但 runner 仍会加载 tutorial 并把它（连同 `skills.max_images` 张图片，`null` = 不限）注入第一轮 user turn —— 这是"跳过 skill 抽取"的消融基线。

---

## 5. Step 循环

`run_single_task()` 是 benchmark 无关的核心循环（`runner.py`）：

```python
obs = env.reset(task)  # 初始观测

while not done and step_idx < max_steps:
    response, actions = agent.predict(instruction, obs)
    for action in actions:
        obs, reward, done, info = env.step(action, sleep_after_execution)
        # 保存 step_data 到 traj.jsonl
        # kit.save_observation(obs, step_obs_dir)
        if done: break
    step_idx += 1

score = env.evaluate()
```

### `agent.predict()` 内部流程

**SimpleAgent**：

1. `obs_content = kit.encode_observation(obs)`
2. `history = state.get_recent_history(window)`
3. `messages = msg.build_action_messages(skills, obs, history, instruction)`
4. `response = vlm.chat(messages)`
5. `actions = kit.parse_actions(response)`
6. `state.record(obs_content, response, actions[0])`

**PhasedAgent**：

1. `obs_content = kit.encode_observation(obs)`
2. `decision = planner.plan(skills, state, obs_content, instruction)` — 第 1 次 VLM 调用，返回 `PlanDecision(action, reasoning, guidance)`
3. 根据 `decision.action` 分发：
   - **DONE** → 返回 `("DONE", ["DONE"])`
   - **FAIL** → 返回 `("FAIL", ["FAIL"])`
   - **REFLECT** → `build_reflect_messages(skills, obs, history, instruction, decision)` → VLM → parse（全部 skills + 近期轨迹，单轮）
   - **EXECUTE** → `build_executor_messages(obs, instruction, decision)` → VLM → parse（无 skills 无历史，单轮）

---

## 5.5 Reviser 修订循环（可选）

当 `reviser.max_attempts > 1` 时，`run_single_task()` 不直接返回，而是进入 `ReviserRunner.run_with_reviser()` 的循环：

1. attempt_j 跑完 → 写 `attempt_j/{traj.jsonl, meta.json, skills/, ...}`
2. `ReviserAnalyzer` 读 traj，按 `chunk_size` 分段输出 `<summary>` + 末段 `<root_cause>`
3. 默认 `outcome_assessment == "likely_success"` → 把 `early_stop_triggered=True` 写到 attempt_j 的 meta（**不 break**）
4. `ReviserRefiner` 用 root_cause + 教程 + 旧 skills 产新 skills，写到 `attempt_{j+1}/skills/`
5. attempt_{j+1} 用新 skills 重跑

`agent_mode == vanilla` 或 `vanilla_tutorial` 时被夹回 1（无 skills 可改）。详见 [`reviser-loop.md`](reviser-loop.md)。

## 6. 四种 Agent 模式对比

### SimpleAgent（1 次 VLM 调用/step）

`encode_observation` → `build_action_messages`（skills 在第一轮 user turn）→ `vlm.chat` → `parse_actions` → `state.record`

VLM 自主完成：状态感知 → skill 参考 → 进度判断 → 错误处理 → 完成检测。

### PhasedAgent（2 次 VLM 调用/step）

`encode_observation` → **Phase 1**: `planner.plan()`（`build_planner_messages` → VLM → JSON → `PlanDecision`）→ **Phase 2**: 根据 decision 分发到 Executor / Reflector（→ VLM → parse）

Planner 输出 JSON：`{action, reasoning, guidance}`。不筛选 skill，只提供行动指引。
Executor 看 instruction + planner guidance + 当前观测（无 skills、无历史）。
Reflector 看全部 skills + 近期轨迹（按 step 聚合 action+obs）用于对比诊断。

### VanillaAgent（1 次 VLM 调用/step，无 skills）

`encode_observation` → `build_vanilla_messages`（无 skills）→ `vlm.chat` → `parse_actions` → `state.record`

与 SimpleAgent 结构相同，但不注入 skills。Runner 跳过 tutorial 加载和 skill 抽取。

### VanillaTutorialAgent（1 次 VLM 调用/step，原始教程代替 skills）

`encode_observation` → `build_vanilla_tutorial_messages`（第一轮 user turn 注入 `## Reference Tutorial` + 教程 body + 受 `skills.max_images` 限制的图片）→ `vlm.chat` → `parse_actions` → `state.record`

与 VanillaAgent 的区别：runner **加载 tutorial 但不抽取 skill**（`_load_task_skills` 直接返回空 `Skills` + `TutorialMaterial`），第一轮 user turn 注入原始教程 body 与图片。system prompt 来自 `VANILLA_TUTORIAL_ACTION_SYSTEM_TMPL`，明确告诉模型这是 RAW 素材（含噪声、可能与当前环境不一致），需要自己挑步骤、自己适配。后续 user turn 与 SimpleAgent / VanillaAgent 一致（仅 bridging + obs）。

---

## 7. 结果存储

```
results/{tutorial_type}/{run_name}/{benchmark}/{domain}/{task_id}/
  task.json                # 任务描述快照
  attempt_1/               # canonical bucket 仅存 attempt_1
    traj.jsonl             # 逐步轨迹（step, action, response, reward, phase, planner...）
    result.txt             # 最终分数
    runtime.log            # 任务执行日志
    recording.mp4          # 屏幕录制（GUI benchmark）
    step_N_TIMESTAMP/      # 每步观测快照
    skills/                # 本 attempt 用的 skills 快照
    meta.json              # {score, steps_taken, skills_count, early_stop_triggered, completed}
  attempt_2/  ...          # 仅 reviser bucket 才有，且额外含 root_cause.xml
  experiment_results.json  # bucket 级别聚合，每个 attempt 完成后重写
```

`run_name` 格式：`a2s-{agent_mode}-{agent_model}[-r_{reviser_model}]`，如：
- `a2s-simple-gpt-4o` —— canonical bucket（裸 agent baseline）
- `a2s-simple-gpt-4o-r_claude-sonnet-4` —— reviser bucket（attempt_2+）

详见 [`reviser-loop.md`](reviser-loop.md) 第 3 节。

---

## 8. 并行 / 单 worker 运行

`run_parallel(cfg)` 是唯一的入口，使用 `multiprocessing`：

1. 主进程收集所有任务，放入 `Manager().Queue()`
2. 启动 `num_envs` 个 worker 进程（`num_envs=1` 也是子进程，不退化为主进程内跑）
3. 每个 worker 独立创建 Kit + Env + VLM，从队列拉取任务
4. canonical bucket 下 `.in_progress.lock` 写入 PID；worker 异常退出后下一轮启动会按 PID 存活检测自动回收 stale lock
5. worker 挂掉后自动重启
6. 信号处理：SIGINT/SIGTERM 优雅关闭所有环境

---

## 源文件索引

| 文件 | 职责 |
|------|------|
| `runner.py` | `run_parallel()`（统一入口）、`run_single_task()`（per-task leaf）、`_compute_run_names()`、`_effective_max_attempts()` |
| `agent_factory.py` | `create_agent()` — 模式分发 |
| `reviser/reviser_runner.py` | `ReviserRunner.run_with_reviser()` — attempt 循环、双桶布局、early_stop 标记 |
| `reviser/analyzer.py` | `ReviserAnalyzer` — Phase 1（trajectory → root_cause XML） |
| `reviser/refiner.py` | `ReviserRefiner` — Phase 2（root_cause + tutorial → 新 skills） |
| `metrics/tracker.py` | `ExperimentTracker` / `AttemptResult` / `TaskResult` |
| `agent/simple_agent.py` | `SimpleAgent.predict()` |
| `agent/phased_agent.py` | `PhasedAgent.predict()` |
| `agent/vanilla_agent.py` | `VanillaAgent.predict()` |
| `agent/vanilla_tutorial_agent.py` | `VanillaTutorialAgent.predict()` |
| `agent/planner.py` | `SoftPlanner.plan()` |
| `agent/state.py` | `AgentState` — 轨迹历史 |
| `agent/message_builder.py` | `MessageBuilder` — 消息拼装 |
| `benchmark_kit.py` | `BenchmarkKit` ABC |
| `env_base.py` | `EnvironmentInterface` ABC |
| `parser/skill_store.py` | `get_or_extract_skills()` — Skill 缓存 |
| `parser/tutorial_loader.py` | `load_tutorial()` |
| `vlm/client.py` | `VLMClient.chat()` |
