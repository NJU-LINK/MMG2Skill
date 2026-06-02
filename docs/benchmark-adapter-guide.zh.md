# Benchmark 适配指南

本文档面向需要接入新 benchmark 的开发者，讲解如何编写 `BenchmarkKit` 子类，以及你提供的 prompt 如何流入 agent 层。

---

## 1. 职责边界

**你负责（BenchmarkKit）**：提供 system_prompt 文本、提供 bridging_text 等可选文本、编码观测为 VLM content blocks、解析 LLM 回复为 action 列表、创建环境实例、收集任务列表

**框架负责（MessageBuilder）**：将 system_prompt 填入模板、编排多轮历史、组装 skills/planner 指令、控制消息结构和角色顺序、管理 skill 图片嵌入

**核心原则**：Kit 只提供领域文本和行为，不拼装 `list[dict]` 格式的消息列表。

---

## 2. 必须实现

### `system_prompt` (@property)

领域身份 + 动作格式 + 特殊信号定义。

```python
@property
def system_prompt(self) -> str:
    return """
    You are an agent that operates in [your environment].
    
    Return actions in [your format].
    
    Special codes:
    - WAIT — wait for environment to settle
    - DONE — task completed
    - FAIL — task cannot be completed
    """
```

**应包含**：环境是什么、动作格式、特殊信号含义
**不应包含**：skills 使用说明、planner 格式、多轮历史管理（这些由框架模板处理）

### `encode_observation(obs: dict) -> list[dict]`

将环境观测转换为 VLM 可理解的 content blocks。

```python
def encode_observation(self, obs: dict) -> list[dict]:
    blocks = []
    if obs.get("error"):
        blocks.append({"type": "text", "text": f"Error: {obs['error']}"})
    if obs.get("screenshot"):
        data_url = encode_image_bytes(obs["screenshot"], "png")
        blocks.append({"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})
    return blocks
```

### `parse_actions(response: str) -> list[str]`

从 LLM 回复中提取可执行的 action 列表。必须识别框架信号 `DONE`、`FAIL`。

```python
def parse_actions(self, response: str) -> list[str]:
    # 识别 DONE/FAIL/WAIT
    for signal in ("DONE", "FAIL", "WAIT"):
        if signal in response:
            return [signal]
    # 提取代码块或其他格式
    ...
```

### `create_env(env_cfg: dict) -> EnvironmentInterface`

创建并返回一个环境实例。每个 worker 调用一次。

### `collect_tasks(cfg: dict) -> list[TaskDescriptor]`

从配置中收集任务列表。可以返回 `TaskDescriptor` 子类。

---

## 3. 可选 property（有默认值）

| Property | 默认值 | 用途 |
|----------|--------|------|
| `bridging_text` | `"Given the current observation. What's the next step?"` | 历史轮次间的衔接文本 |
| `reflection_guidance` | `""` | 反思模式的领域指引 |
| `planner_guidance` | `""` | Planner 的领域上下文 |
| `skill_extraction_guidance` | `""` | 技能抽取的领域指引 |

只需在默认值不满足需求时重写。

---

## 4. 你的 prompt 如何流入 agent

以 `system_prompt` 为例，完整流程：

```
1. 你在 benchmarks/<name>/prompts.py 中定义常量
   SYSTEM_PROMPT = "You are an agent that..."

2. kit.py 中引用常量作为 property
   @property
   def system_prompt(self) -> str:
       return SYSTEM_PROMPT

3. MessageBuilder 在构建消息时读取 property
   system_text = SIMPLE_ACTION_SYSTEM_TMPL.format(
       domain_system_prompt=self.kit.system_prompt
   )

4. 模板拼装后生成完整 system 消息
   → 你的领域文本 + 框架的 SOP/观察/推理指令
```

各 property 的注入位置：

| Kit Property | 注入位置 |
|---|---|
| `kit.system_prompt` | system 消息（Autonomous/Phased/Reflection） |
| `kit.bridging_text` | 历史轮 user 消息的前缀 |
| `kit.reflection_guidance` | Reflection system 消息的尾部 |
| `kit.planner_guidance` | Planner system 消息中间 |
| `kit.skill_extraction_guidance` | Extraction system 消息中间 |

详细结构图见 [prompt-assembly.md](prompt-assembly.md) 第 3 节。

---

## 5. 注册

```python
from anything2skill.benchmarks.registry import register_kit

@register_kit("myworld")
class MyWorldKit(BenchmarkKit):
    ...
```

注册后，配置中 `benchmark=myworld` 即可使用。

---

## 6. 配置

创建 `configs/benchmark/myworld.yaml`，首行必须是 `# @package _global_`：

```yaml
# @package _global_

benchmark: myworld

env:
  # 环境参数
  ...

agent:
  agent_mode: simple  # 或 phased
  max_steps: 15

data:
  dir: data/myworld

tasks:
  # 任务筛选参数
  ...
```

---

## 7. Checklist

1. [ ] `benchmarks/<name>/__init__.py` — 空文件
2. [ ] `benchmarks/<name>/prompts.py` — 领域 prompt 常量（大写蛇形命名）
3. [ ] `benchmarks/<name>/kit.py` — BenchmarkKit 子类 + `@register_kit`
4. [ ] `configs/benchmark/<name>.yaml` — Hydra 配置
5. [ ] 如需自定义 TaskDescriptor，创建 `@dataclass` 子类
6. [ ] 用 mock_prompt_viewer 验证 prompt 拼装结果
7. [ ] 若与现有 benchmark 共享 runtime（如卡牌类共享 RLCard env），可复用 `benchmarks/rlcard_common.py:BaseRLCardEnvWrapper` 或仿照其形式提供共享 helper；这类 helper **不要** 放到 `agent/` 或 `reviser/` 下
8. [ ] 准备独立 conda 环境脚本 `scripts/<name>/setup_conda.sh`（参考 `scripts/osworld/`、`scripts/minecraft/`、`scripts/rlcard/`）
9. [ ] **无需修改 `agent/` 或 `reviser/` 下的任何文件**

> 动作格式不强求 `` ```python `` 块。Minecraft / RLCard 都使用 `<action>...</action>` 标签（thinking 模型更友好），由 kit 自行决定，但 `parse_actions()` 必须识别框架信号 `DONE` / `FAIL`。

参考实现：
- GUI / 单图观测：`benchmarks/osworld/kit.py`
- `<action>` 标签 + 物理交互：`benchmarks/minecraft/kit.py`
- 文本观测 + 共享 wrapper：`benchmarks/doudizhu/kit.py`（+ `benchmarks/rlcard_common.py`）

---

## 8. 用 mock_prompt_viewer 验证

```python
from anything2skill.tests.mock_prompt_viewer import view_prompts
from anything2skill.benchmarks.myworld.kit import MyWorldKit

# 传入你的 Kit，查看所有场景的 prompt 拼装结果
view_prompts(MyWorldKit())

# 只看 planner 场景
view_prompts(MyWorldKit(), scenarios=["planner"])

# 传入真实观测验证 encode_observation 编码效果
view_prompts(MyWorldKit(), obs={"your_obs_key": real_data})

# 导出 JSON 到 prompt_snapshots/ 离线查看
view_prompts(MyWorldKit(), export=True)
```

检查输出中你的 `system_prompt`、`bridging_text` 等文本是否正确出现在预期位置。
