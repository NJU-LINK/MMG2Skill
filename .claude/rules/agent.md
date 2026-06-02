---
paths:
  - "anything2skill/agent/**/*.py"
---

# Agent 层开发规则

## 核心约束

- `agent/` 中 **禁止** 导入 `anything2skill.benchmarks.*`
  - 仅在 `TYPE_CHECKING` 块中可 import 类型用于 type hints
  - 所有领域行为通过 `BenchmarkKit` 的 property 注入（`kit.system_prompt`、`kit.bridging_text` 等）
- `reviser/` 同样 **禁止** 导入 `anything2skill.benchmarks.*`，领域文本经由 `kit.skill_extraction_guidance`、`kit.system_prompt` 等 property 注入；analyzer / refiner 的 prompt 模板放在 `reviser/prompts.py`，与 `agent/prompts.py` 平行

## 消息编排

- `MessageBuilder` 是唯一负责 VLM 消息拼装的类
- prompt 模板定义在 `prompts.py`，通过 `.format()` 填入占位符
- 占位符命名：`{domain_*}` = 来自 Kit 的领域文本
- 新增 prompt 模板必须定义在 `prompts.py`，不能散落在其他文件
- 消息格式统一为 OSWorld 风格：instruction 在第一轮 user turn，后续 user turn 只有 bridging_text + obs

### MessageBuilder 私有 helper

| helper | 职责 |
|--------|------|
| `_task_heading(instruction)` | 返回 `# Your Task\n{instruction}` 文本块（不含 bridging） |
| `_task_with_bridging(instruction)` | 返回 `# Your Task\n{instruction}\n\n{bridging_text}`（多轮 agent 用） |
| `_build_skills_block(skills)` | skills 文本 + skill 图片（不含 instruction/bridging） |
| `_build_multiturn(system, first_content, history, obs)` | 多轮消息循环（SimpleAgent / VanillaAgent / VanillaTutorialAgent 共用） |
| `_build_tutorial_block(tutorial, max_images)` | 把原始教程 body + 受 `max_images` 限制的图片渲染为 `## Reference Tutorial` 内容块（VanillaTutorialAgent 专用，`max_images <= 0` 不限） |
| `_build_history_steps(history, current_obs)` | 按 step 聚合历史：`Step N / Action / Observation`（Planner / Reflector 用） |

## 框架级信号

- `DONE`（任务完成）和 `FAIL`（任务不可行）是框架层面的终止信号
- `WAIT` 等其他信号是领域特定的，由各 Kit 自行定义
- prompt 模板中不硬编码 `DONE`/`FAIL`，使用 "domain-specified format" 表述，让 kit 的 `system_prompt` 定义具体格式

## Agent 模式

- `SimpleAgent`：单次 VLM 调用，所有 skills 在第一轮 user turn，多轮历史，无 planner
- `PhasedAgent`：SoftPlanner 先评估 → Executor / Reflector / DONE / FAIL 分发
- `VanillaAgent`：无 skills baseline，与 SimpleAgent 结构相同但不注入 skills
- `VanillaTutorialAgent`：跳过 skill 抽取的消融基线。第一轮 user turn 注入原始教程 body + 受 `skills.max_images` 限制的图片（`null` = 不限），后续轮与 SimpleAgent / VanillaAgent 一致。Reviser 与 vanilla 一样被夹回 1
- 四者都继承 `BaseAgent`，实现 `predict(instruction, obs) -> (response, actions)`
- 新 agent 模式必须在 `agent_factory.py` 注册；如不支持 reviser，须加入 `runner._NO_REVISER_MODES`

## Planner

- `PlanAction` 枚举：`EXECUTE` / `REFLECT` / `DONE` / `FAIL`
- SoftPlanner 是"软规划"：无 step index tracking，VLM 自行判断当前状态
- `PlanDecision` 包含 `action`、`reasoning`、`guidance`
- Planner 不筛选 skill，只提供行动指引

## PhasedAgent 子组件

- **Executor**：接收 instruction + planner guidance + 当前观测，无 skills 无历史，单轮，专注执行
- **Reflector**：接收全部 skills + 近期轨迹（按 step 聚合）+ planner guidance，单轮，用于对比诊断

## 状态管理

- `AgentState` 记录轨迹历史（`obs_content`, `response`, `action`）
- `state.record()` 在每次 predict 末尾调用
- `history_window` 控制多轮上下文窗口大小

## 工具函数

- `skill_utils.py` 提供通用格式化：`format_skills_for_prompt()` 返回交错的 text/image content blocks
- 内联 `![alt](filename)` 引用在注入时被解析为 `[Image: filename]` 标签 + image_url block
- 这些函数操作 `Skills` / `Skill` 数据类型，不依赖任何 benchmark
