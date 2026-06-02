---
paths:
  - "anything2skill/benchmarks/**/*.py"
  - "anything2skill/benchmark_kit.py"
---

# Benchmark 适配规则

## BenchmarkKit 子类实现

### 必须实现

| 方法 | 签名 | 说明 |
|------|------|------|
| `encode_observation` | `(obs: dict) -> list[dict]` | 环境观测 → VLM content blocks |
| `parse_actions` | `(response: str) -> list[str]` | LLM 回复 → 可执行 action 列表 |
| `system_prompt` | `@property -> str` | 领域身份 + 动作格式 + 特殊信号 |
| `create_env` | `(env_cfg: dict) -> EnvironmentInterface` | 创建环境实例 |
| `collect_tasks` | `(cfg: dict) -> list[TaskDescriptor]` | 收集任务列表 |

### 可选 property（有默认值）

- `bridging_text` — 历史轮次衔接文本
- `reflection_guidance` — 反思/错误恢复提示
- `planner_guidance` — 规划器领域提示
- `skill_extraction_guidance` — 技能抽取领域提示

### 可选方法

- `save_observation(obs, path)` — 保存观测到磁盘
- `get_result_subdir(task) -> str` — 结果目录相对路径

## 关键约束

- Kit **禁止** 拼装 VLM 消息列表（`list[dict]` 格式），这是 `MessageBuilder` 的职责
- prompt hook 均为 `@property`，不是方法（`self.bridging_text` 而非 `self.bridging_text()`）
- 领域 prompt 文本放在 `benchmarks/<name>/prompts.py` 中，kit 只引用常量
- `system_prompt` 中放：领域身份 + 动作格式 + 特殊动作/信号（DONE/FAIL/WAIT 的含义与格式）
- `system_prompt` 中 **不放**：skills 使用说明、planner 格式、多轮历史管理

## 框架级信号处理

- `EnvironmentInterface.step()` **必须** 处理框架级信号 `DONE` 和 `FAIL`（收到时设 `done=True`）
- 其他信号（如 `WAIT`）是领域特定的，由各 kit/env 自行定义和处理
- `parse_actions()` 需识别框架信号 `DONE`/`FAIL`，以及领域特定信号

## 动作格式

- 框架不强制 `` ```python `` 块。可选格式包括 `<action>...</action>` XML 标签（Minecraft / RLCard 用，对 thinking 模型更友好）或语言代码块（OSWorld 用 pyautogui）
- 每个 kit 的 `system_prompt` 自行声明本领域的动作格式，`parse_actions()` 自行识别
- 框架只关心 `DONE` / `FAIL` 是否被识别 + 返回的动作列表是否能被对应 env 执行

## 添加新 Benchmark

1. 创建 `benchmarks/<name>/kit.py`，实现 BenchmarkKit 子类
2. 创建 `benchmarks/<name>/prompts.py`，定义领域 prompt 常量（大写蛇形命名：`SYSTEM_PROMPT`）
3. 用 `@register_kit("<name>")` 装饰器注册
4. 创建 `configs/benchmark/<name>.yaml`（文件首行 `# @package _global_`）
5. 如需自定义 TaskDescriptor，创建 `@dataclass` 子类
6. 若与现有 benchmark 共享 runtime（如卡牌类共享 RLCard env），可复用 `benchmarks/rlcard_common.py:BaseRLCardEnvWrapper` 或仿照其形式提供共享 helper；这类 helper **不要** 放到 `agent/` 或 `reviser/` 下
7. 准备独立 conda 环境脚本 `scripts/<name>/setup_conda.sh`
8. **不需要修改 agent/ 或 reviser/ 下的任何文件**

参考实现：
- `benchmarks/osworld/kit.py` —— GUI / pyautogui 动作
- `benchmarks/minecraft/kit.py` —— `<action>` 标签 + 物理交互
- `benchmarks/doudizhu/kit.py` —— 文本观测 + 共享 wrapper
