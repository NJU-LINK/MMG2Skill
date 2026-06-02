# Prompt 拼接流程

本文档是最核心的参考：讲解 **BenchmarkKit 提供的领域 prompt 如何被框架拼装成发给 VLM 的完整消息**。

> 如果你是 benchmark 适配者，请先读本文档，再读 [benchmark-adapter-guide.md](benchmark-adapter-guide.md)。

---

## 1. 模板架构

所有 prompt 模板定义在 `anything2skill/agent/prompts.py`，以字符串常量形式存在（如 `SIMPLE_ACTION_SYSTEM_TMPL`）。

模板通过 Python 的 `.format()` 填入占位符。占位符命名规则：

| 前缀 | 来源 | 示例 |
|------|------|------|
| `{domain_*}` | BenchmarkKit 的 property | `{domain_system_prompt}` ← `kit.system_prompt` |

**关键原则**：Kit 只提供领域文本（`system_prompt`、`bridging_text` 等），`MessageBuilder` 负责将它们填入模板、组装消息。Kit 开发者不需要接触消息拼装逻辑。

---

## 2. 占位符映射表

| 占位符 | 数据来源 | 使用位置 |
|--------|---------|---------|
| `{domain_system_prompt}` | `kit.system_prompt` | SimpleAgent/Executor/Reflector/Vanilla/VanillaTutorial 的 system 消息 |
| `{domain_reflection_guidance}` | `kit.reflection_guidance` | Reflector 的 system 消息 |
| `{domain_planner_guidance}` | `kit.planner_guidance` | Planner 的 system 消息 |
| `{domain_guidance}` | `kit.skill_extraction_guidance` | Skill Extraction 的 system 消息 |

user 消息中的内容（instruction、skills、planner guidance 等）由 `MessageBuilder` 直接拼装，不通过模板占位符。

---

## 3. 七种消息拼装

`MessageBuilder` 提供 7 个构建方法，每个对应一种场景。

### 统一风格

所有 agent 遵循 OSWorld 风格：
- **system**：仅领域 prompt + 角色指令
- **第一轮 user**：skills（如有）+ `# Your Task` + instruction + bridging_text（多轮）或直接接 obs（单轮）
- **后续 user**：bridging_text + obs

### 3.1 SimpleAgent

**入口**：`build_action_messages(skills, obs, history, instruction)`

```
[system]: SIMPLE_ACTION_SYSTEM_TMPL（domain_prompt + skills 使用指引 + 四步职责）
[user]:   skills_blocks (text + inline images) + "# Your Task" + instruction + bridging_text + obs_1
[asst]:   response_1
[user]:   bridging_text + obs_current
```

**特点**：所有 skills 在第一轮 user turn，多轮历史。

### 3.2 VanillaAgent

**入口**：`build_vanilla_messages(obs, history, instruction)`

```
[system]: VANILLA_ACTION_SYSTEM_TMPL（domain_prompt + 回复格式）
[user]:   "# Your Task" + instruction + bridging_text + obs_1
[asst]:   response_1
[user]:   bridging_text + obs_current
```

**特点**：无 skills，结构与 SimpleAgent 相同。

### 3.3 VanillaTutorialAgent

**入口**：`build_vanilla_tutorial_messages(tutorial, max_images, obs, history, instruction)`

```
[system]: VANILLA_TUTORIAL_ACTION_SYSTEM_TMPL（domain_prompt + "RAW source material" 提示 + 五步职责）
[user]:   "## Reference Tutorial" + tutorial.body
          + "Tutorial images (N):" + 交错的 [Image: filename] + image_url（≤ max_images，0 = 不限）
          + "# Your Task" + instruction + bridging_text + obs_1
[asst]:   response_1
[user]:   bridging_text + obs_current
```

**特点**：跳过 skill 抽取，直接喂原始教程 body + 图片。`tutorial=None` 时 tutorial 块被整体省略，等价于 VanillaAgent。`max_images <= 0` 表示不裁剪。system prompt 显式提示模型：教程是 RAW 素材，可能含与当前环境不符的步骤，需要自行筛选与适配。

### 3.4 Executor（PhasedAgent EXECUTE）

**入口**：`build_executor_messages(obs, instruction, decision)`

```
[system]: EXECUTOR_SYSTEM_TMPL（domain_prompt + 跟随 planner 指引）
[user]:   "## Your Task" + instruction + planner_reasoning + planner_guidance + bridging_text + obs_current
```

**特点**：无 skills、无历史，单轮。Executor 专注执行 planner 的指令。

### 3.5 Reflector（PhasedAgent REFLECT）

**入口**：`build_reflect_messages(skills, obs, history, instruction, decision)`

```
[system]: REFLECTOR_SYSTEM_TMPL（domain_prompt + 诊断流程 + reflection_guidance）
[user]:   skills_blocks (text + inline images) + "# Your Task" + instruction
          + planner_reasoning + planner_guidance
          + "## Recent Steps"
          + Step 1: action + obs_screenshot
          + Step 2: action + obs_screenshot
          + "## Current Observation" + screenshot_current
```

**特点**：全部 skills + 近期轨迹（按 step 聚合 action+obs），单轮，用于对比诊断。

### 3.6 Planner（SoftPlanner 评估）

**入口**：`build_planner_messages(skills, state, obs, instruction)`

```
[system]: PLANNER_SYSTEM_TMPL（评估者身份 + planner_guidance + JSON 输出格式）
[user]:   skills_blocks (text + inline images) + "# Your Task" + instruction
          + "## Recent Steps"
          + Step 1: action + obs_screenshot
          + Step 2: action + obs_screenshot
          + "## Current Observation" + screenshot_current
```

**特点**：Planner 输出 JSON `{action, reasoning, guidance}`，不筛选 skill。

### 3.7 Skill Extraction（教程→技能）

**入口**：`build_skill_extraction_messages(tutorial_content, instruction, image_entries)`

```
[system]: SKILL_EXTRACTION_SYSTEM_TMPL（技能抽取专家 + 领域指引 + 输出格式）
[user]:   "TASK: {instruction}\n\nTUTORIAL:\n{tutorial_content}" + 图片列表
```

---

## 4. 历史嵌入

### 多轮历史（SimpleAgent / VanillaAgent / VanillaTutorialAgent）

`_build_multiturn()` 将轨迹历史嵌入为多轮对话，第一轮 user 由调用方定制，后续统一为 `bridging_text + obs`：

```
[user: first_content + obs₁]  [assistant: response₁]    ← 历史轮 1
[user: bridging + obs₂]       [assistant: response₂]    ← 历史轮 2
[user: bridging + obs_current]                           ← 当前轮
```

- `history_window` 配置项控制注入的历史条数（默认 3）
- `bridging_text` 是 Kit 提供的衔接文本（OSWorld 中是 `"Given the screenshot below. What's the next step?"`）

### Step 聚合历史（Planner / Reflector）

`_build_history_steps()` 将历史按 step 聚合为单轮内容：

```
## Recent Steps
### Step 1
Action: pyautogui.click(100, 200)
Observation:
[screenshot_1]

### Step 2
Action: pyautogui.hotkey('ctrl', 's')
Observation:
[screenshot_2]

## Current Observation
[screenshot_current]
```

---

## 5. Skill 图片嵌入

Skill content 中的 `![alt](filename)` 内联引用在注入 agent 时被解析为交错的 text + image blocks：

- 每张图片前保留原始 `![alt](filename)` 引用作为 text block（保留描述性 alt text）
- 图片以 `{"type": "image_url", "image_url": {"url": "data:image/...", "detail": "high"}}` 格式紧随其后
- 图片按 content 中引用的顺序内联插入，不再集中堆积在末尾

---

## 6. OSWorld 具体示例

以 OSWorld 的 SimpleAgent 首次调用为例，追踪完整拼装过程。

**Kit 提供的领域文本**（定义在 `benchmarks/osworld/prompts.py`）：

```python
# kit.system_prompt →
"You are an agent that performs desktop computer tasks on Ubuntu..."

# kit.bridging_text →
"Given the screenshot below. What's the next step?"

# kit.planner_guidance →
"Observations are screenshots of an Ubuntu desktop. Actions are pyautogui Python code."
```

**拼装后的最终消息**：

```
Message 0 [system]:
  "You are an agent that performs desktop computer tasks on Ubuntu.
   Return pyautogui code in ```python``` blocks.
   Special codes: ```WAIT```, ```DONE```, ```FAIL```.
   ...
   You have reference skills extracted from tutorials. These skills are
   for REFERENCE ONLY — they may not match the actual environment...
   Your responsibilities on each turn:
   1. OBSERVE the current state carefully...
   2. ASSESS whether the reference skills apply...
   3. DECIDE...
   4. HANDLE ERRORS..."

Message 1 [user]:
  {"type": "text", "text": "## Reference Skills"}
  {"type": "text", "text": "### open-settings\n> Open the system settings..."}
  {"type": "text", "text": "## Steps\n1. Click 'Activities' in the top-left corner"}
  {"type": "text", "text": "![Right-click context menu...](screenshot-3.png)"}
  {"type": "image_url", ...}
  {"type": "text", "text": "2. In the settings panel, toggle Bluetooth on"}
  {"type": "text", "text": "![Settings panel showing...](screenshot-5.png)"}
  {"type": "image_url", ...}
  {"type": "text", "text": "## Expected Result\nBluetooth is turned on."}
  // 无内联图片引用时，整个 skill content 为单个 text block
  {"type": "text", "text": "# Your Task\nOpen Settings and turn on Bluetooth\n\n
   Given the screenshot below. What's the next step?"}
  + [screenshot image block]                          ← 当前观测
```

> Minecraft 与 RLCard 的拼装结构与 OSWorld 相同，只是 `system_prompt` / `bridging_text` / 动作格式不同（前两者使用 `<action>...</action>` 标签）。可用 `view_prompts(MinecraftKit())` 或 `view_prompts(BlackjackKit())` 自行核对。

---

## 7. 可视化工具

`anything2skill/tests/mock_prompt_viewer.py` 提供 `view_prompts()` 函数，传入任意 Kit 即可查看拼装结果。

### 基本用法

```python
from anything2skill.tests.mock_prompt_viewer import view_prompts
from anything2skill.benchmarks.osworld.kit import OSWorldKit

# 传入 Kit，查看所有场景（obs 用通用占位符）
view_prompts(OSWorldKit())

# 只看某个场景
view_prompts(OSWorldKit(), scenarios=["planner"])
view_prompts(OSWorldKit(), scenarios=["simple", "executor"])
```

可选场景：`simple`、`vanilla`、`executor`、`reflect`、`planner`、`extraction`（VanillaTutorialAgent 暂未接入 viewer，可直接调用 `MessageBuilder.build_vanilla_tutorial_messages()` 自行核对）

### 传入真实观测

默认使用 `<Observation>` 文本占位符。如需验证真实 obs 编码后的效果：

```python
# 传入真实 obs dict，会调用 kit.encode_observation() 编码
view_prompts(OSWorldKit(), obs={"screenshot": png_bytes})
```

### 导出 JSON

```python
# 导出到 anything2skill/tests/prompt_snapshots/*.json
view_prompts(OSWorldKit(), export=True)
```

每个场景生成一个 JSON 文件（如 `simple_with_history.json`），内容就是发给 VLM 的完整 `messages` 数组，可离线检查，或用任意兼容 API client 验证。

### CLI 快速运行

```bash
# 默认使用 OSWorldKit，打印全部场景
python -m anything2skill.tests.mock_prompt_viewer
```

---

## 8. 源文件索引

| 文件 | 职责 |
|------|------|
| `agent/prompts.py` | 所有 prompt 模板常量 |
| `agent/message_builder.py` | `MessageBuilder` — 消息拼装逻辑 |
| `agent/skill_utils.py` | Skills 格式化工具函数 |
| `benchmark_kit.py` | `BenchmarkKit` ABC — Kit 需实现的 property |
| `benchmarks/osworld/prompts.py` | OSWorld 领域 prompt 文本 |
| `benchmarks/osworld/kit.py` | OSWorldKit — 参考实现 |
