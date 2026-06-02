# Anything2Skill 项目规则

## 项目概述

Agent 框架，从多模态教程中学习 Skills（SOP），在不同 benchmark 环境中执行任务。
当前支持：

- **OSWorld**（GUI 自动化，Ubuntu 桌面）
- **Minecraft**（开放世界物理交互，OpenHA 后端）
- **3 个 RLCard 卡牌游戏**（doudizhu / mahjong / nolimit_holdem）

每个 benchmark 通过 `BenchmarkKit` 注册，框架本身保持 benchmark 无关。

## 架构分层

```
BenchmarkKit (领域层)  →  只提供领域相关的 property/方法（详见 benchmark.md）
MessageBuilder (编排层)  →  统一管理消息拼装、历史编排、格式约束（详见 agent.md）
Agent (决策层)  →  SimpleAgent / PhasedAgent / VanillaAgent / VanillaTutorialAgent，调用 VLM 决策（详见 agent.md）
Reviser (修订层)  →  attempt 之间的 analyze→refine 循环，双桶布局，详见 docs/reviser-loop.md
```

`agent/` 与 `reviser/` 都禁止 import `benchmarks.*`，所有领域文本经由 `kit.*` property 注入。

## 全局代码规范

- Python 3.12+（RLCard 默认）/ 3.10（OSWorld、Minecraft conda 环境），统一 `from __future__ import annotations`
- prompt 字符串统一使用 `"""..."""` 三引号，保持可读性
- kit 的 prompt hook 均为 `@property`，不是方法
- 配置使用 OmegaConf/Hydra，合并优先级：`api/default.yaml` < `config.yaml` < `benchmark/*.yaml` < CLI；顶层除 `agent.*` 外还有 `reviser.*` / `skills.*` / `data.*` / `env.*` / `runner.*` / `tasks.*`
- 测试在 `anything2skill/tests/`，用 pytest 运行
- 每个 benchmark 独立 conda 环境（`openha` / `osworld` / `rlcard`），不再用 `uv`

## 目录结构

```
anything2skill/             # 主包
  agent/                    # 决策层（benchmark 无关）
  agent_factory.py          # create_agent() 模式分发
  benchmarks/               # 各 benchmark 的 BenchmarkKit 实现
    rlcard_common.py        # 9 个卡牌游戏共享的 env wrapper + GAME_OPPONENTS
    registry.py             # @register_kit 装饰器
  reviser/                  # 修订层：analyzer / refiner / reviser_runner
  metrics/                  # ExperimentTracker / AttemptResult / TaskResult
  parser/                   # skill 抽取、存储、教程加载
  vlm/                      # VLM client
  env_base.py               # 环境抽象接口
  benchmark_kit.py          # BenchmarkKit ABC
  runner.py                 # run_parallel + run_single_task + 双桶 run_name 计算
configs/                    # YAML 配置
data_tutorial/              # 教程素材（按 benchmark/task_id 组织）
data_collection/            # 教程素材收集脚本
skills_cache/               # skill 缓存（{model}/{benchmark}/{task_id}）
results/                    # 实验产物（双桶 run_name）
scripts/                    # 各 benchmark 的 setup_conda.sh / run.sh
notebooks/                  # 实验分析 notebook（per-model 双视图等）

OSWorld/  OpenHA/  RLCard/  # git submodule，提供各 benchmark 的 env 实现
```
