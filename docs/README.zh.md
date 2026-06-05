# MMG2Skill —— 架构与开发者指南

MMG2Skill 的内部架构、数据流与开发者参考。安装与使用见[根目录 README](../README.zh.md)。源码包名为 `anything2skill`。

当前覆盖 **OSWorld**（GUI 自动化）、**OpenHA Minecraft**（开放世界物理交互）、**RLCard** doudizhu / mahjong（success-inferable 策略任务），并保留 **RLCard nolimit_holdem** 作为 private-information boundary diagnostic。

---

## 文档索引

| 文档 | 内容 | 适合谁 |
|------|------|--------|
| [运行流程](execution-flow.zh.md) | 从启动到结束的完整执行流程 | 所有用户 |
| [Prompt 拼接流程](prompt-assembly.zh.md) | Kit prompt 如何拼装成 VLM 消息 | Benchmark 适配者、Agent 开发者 |
| [Benchmark 适配指南](benchmark-adapter-guide.zh.md) | 如何编写新的 `BenchmarkKit` | Benchmark 适配者 |
| [Reviser 修订循环](reviser-loop.zh.md) | 双桶 attempt 布局、early-stop 语义、analyze→refine 管线 | 运行消融或分析实验数据的研究者 |
| [RLCard 人机权重](weights.zh.md) | 固定人机权重、GitHub Release 布局、上传/下载脚本 | 运行 RLCard benchmark 的用户 |

---

## 架构

四层架构：

- **Runner** (`runner.py`) — 配置加载 / 任务收集 / Skill 加载 / Agent 创建 / Step 循环 / 双桶 attempt 调度
  - **BenchmarkKit**（领域层）— 领域 prompt、观测编码、动作解析
  - **MessageBuilder**（编排层）— 模板填充、历史编排、消息组装
  - **Agent**（决策层）— SimpleAgent / PhasedAgent / VanillaAgent / VanillaTutorialAgent，调用 VLM 决策
  - **Reviser**（修订层，可选）— attempt 之间的 analyze→refine 循环，产出修订后的 skills
  - **VLM Client** — OpenAI 兼容 API 调用

`agent/` 与 `reviser/` 都禁止 import `benchmarks.*`，所有领域文本经由 `kit.*` property 注入。

## 数据流

1. **教程素材** 放在 `data_tutorial/{tutorial_type}/{benchmark}/{task_id}/tutorial/`，`tutorial_type ∈ {html, screenshot}`（video 预留）：html 是 `page*.html` + `images/`；screenshot 是 Playwright 滚动截图（`images/frame_*.png`），由 `data.tutorial_type` 切换。
2. **Skill 抽取**：VLM 读取教程，输出 `SKILL.md` 缓存到 `skills_cache/{tutorial_type}/{model}/{benchmark}/{task_id}/`，不同模态互不命中，下次同任务+同模态直接命中。
3. **Agent 执行**：`env.reset()` → `agent.predict(instruction, obs)` → `env.step(action)` 循环到 `DONE` 或 `max_steps`，轨迹写入 `attempt_N/traj.jsonl`。
4. **（可选）Reviser**：`reviser.max_attempts > 1` 时，每 attempt 完成后跑 analyzer 解析轨迹 + refiner 修订 skills，写入下一个 attempt 目录后继续跑。

数据流一览：

```
Tutorial (HTML + 图片) → VLM 抽取 → Skills (SKILL.md) → Agent 使用 → （可选）Reviser 修订 → 下一轮 Agent 使用
env.reset() → obs → agent.predict() → action → env.step() → obs → 循环
```

完整流程见 [execution-flow.zh.md](execution-flow.zh.md)。

## 实验产物

```
results/{tutorial_type}/{run_name}/{benchmark}/{domain}/{task_id}/
  task.json
  attempt_1/
    traj.jsonl, result.txt, runtime.log, recording.mp4, step_N_*/
    skills/, meta.json
  attempt_2/  attempt_3/  ...     # 仅 Reviser 模式产生，含 root_cause.xml
  experiment_results.json         # 桶级聚合，每个 attempt 结束后重写
```

`run_name` = `a2s-{agent_mode}-{agent_model}[-r_{reviser_model}]`（注意：`tutorial_type` **不**进 `run_name`，由前缀 `results/{tutorial_type}/` 隔离不同模态）：

- **canonical bucket**（裸 agent baseline）只存 `attempt_1`，跨 reviser 实验复用
- **reviser bucket** 存 `attempt_2+`，按 reviser 模型隔离

`meta.json["early_stop_triggered"]` 标记 analyzer 首次输出 `likely_success` 的 attempt，但循环不真 break，便于一次跑同时产出 early-stop / full-run 两个视图。详见 [reviser-loop.zh.md](reviser-loop.zh.md)。

## 添加新 Benchmark

1. `anything2skill/benchmarks/<name>/{__init__.py, prompts.py, kit.py}`，在 kit 上加 `@register_kit("<name>")`。
2. `configs/benchmark/<name>.yaml`（首行 `# @package _global_`）。
3. `scripts/<name>/setup_conda.sh` 准备独立 conda 环境。
4. 用 `python -m anything2skill.tests.mock_prompt_viewer` 验证 prompt 拼装。
5. 不需要修改 `agent/` 或 `reviser/` 下任何文件。

详见 [benchmark-adapter-guide.zh.md](benchmark-adapter-guide.zh.md)。

## 核心概念速查

| 概念 | 说明 |
|------|------|
| **BenchmarkKit** | 领域适配层。提供 `system_prompt`、`encode_observation`、`parse_actions` 等 |
| **MessageBuilder** | 消息编排层。拼装 VLM 消息，Kit 开发者不接触 |
| **SimpleAgent** | 单次 VLM 调用/step，所有 skills 在第一轮 user turn |
| **PhasedAgent** | Planner 评估 → Executor/Reflector/DONE/FAIL 分发 |
| **VanillaAgent** | 无 skills baseline，用于消融实验。Reviser 在此模式下被禁用 |
| **VanillaTutorialAgent** | 直接喂原始教程（body + 图片，受 `skills.max_images` 控制），跳过 skill 抽取。用于消融"抽取步骤本身"的贡献，Reviser 同样禁用 |
| **SoftPlanner** | VLM 驱动的软规划器，不筛选 skill，只提供行动指引 |
| **Skills** | 从教程抽取的 SOP，缓存为 SKILL.md 文件 |
| **DONE/FAIL** | 框架级终止信号，env.step() 收到后设 `done=True` |
| **Reviser** | attempt 之间的 analyze→refine 管线，修订 skills |
| **Analyzer** | Phase 1：trajectory → `<root_cause>` XML |
| **Refiner** | Phase 2：root_cause + 旧 skills + 教程 → 新 skills |
| **Canonical bucket** | `a2s-{mode}-{model}` —— 裸 agent baseline，永远存 attempt_1 |
| **Reviser bucket** | `a2s-{mode}-{model}-r_{reviser_model}` —— attempt_2+ |
| **Attempt** | 一次完整的 task 执行；多 attempt 通过 reviser 串联 |
| **early_stop_triggered** | meta.json 字段，标记 analyzer 首次输出 `likely_success` 的 attempt，不真正 break |
| **ExperimentTracker** | `metrics/tracker.py`，per-attempt / per-task 结果聚合 |
