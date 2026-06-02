<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.png">
    <img alt="MMG2Skill" src="assets/banner-light.png" width="700">
  </picture>
</p>

<p align="center">
  <a href="https://www.nju.edu.cn"><img src="assets/nju_logo.png" height="72" alt="Nanjing University"></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/Kwaipilot"><img src="assets/kwaipilot_logo.png" height="72" alt="Kwaipilot"></a>
</p>
<p align="center">
  <b>NJU-LINK</b>&nbsp;&nbsp;×&nbsp;&nbsp;<b>Kwaipilot</b>
</p>

<h3 align="center">Can Agents Distill In-the-Wild Guides into Self-Evolving Skills?</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2606.01993"><img src="https://img.shields.io/badge/arXiv-2606.01993-b31b1b.svg?style=for-the-badge" alt="arXiv"></a>
  <a href="docs/README.zh.md"><img src="https://img.shields.io/badge/docs-架构-blue.svg?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Docs"></a>
</p>
<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10 | 3.12"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> &bull;
  <a href="#benchmarks">Benchmarks</a> &bull;
  <a href="docs/README.zh.md">架构</a> &bull;
  <a href="#文档">文档</a> &bull;
  <a href="#引用">引用</a>
</p>

<p align="center">
  <a href="README.md">English</a> &bull; <a href="README.zh.md">中文</a>
</p>

---

## 简介

MMG2Skill 把公开多模态教程自动转成可执行 **Skills**（SOP），让**固定**的 VLM agent 在自身交互轨迹中持续修订这些 Skills——**全程不读取 benchmark score**。它用一套 benchmark 无关的框架，覆盖 GUI 自动化、开放世界游戏与策略卡牌游戏。

> 源码包名仍为 `anything2skill`。

## 核心特性

| | |
|---|---|
| **教程 → Skills 抽取** | VLM 从 HTML + 图像资源中提炼 SOP，缓存为 `SKILL.md`；另支持 rendered-screenshot 教程通道。 |
| **Reviser 修订循环** | analyzer→refiner 循环把 agent-visible 轨迹诊断转成 skill edits，跨 attempt 产出 early-stop / full-run 双视图。 |
| **Score-free 更新** | skill 构建、执行、分析、修订全程不读 benchmark score；score 只用于离线评估。 |
| **四种 agent 模式** | `simple`（用 skills）、`phased`（planner→executor/reflector）、`vanilla`（无 skills baseline）、`vanilla_tutorial`（原始教程 baseline）——用于干净的消融。 |
| **Benchmark 无关** | `agent/` 与 `reviser/` 不依赖任何具体 benchmark；新增 benchmark 只需实现一个 `BenchmarkKit`。 |

## 快速开始

### 安装

每个 benchmark 使用独立 conda 环境（依赖差异较大）。先初始化子模块：

```bash
git submodule update --init OSWorld OpenHA RLCard
```

再按需准备环境：

```bash
# OSWorld（Python 3.10）
bash scripts/osworld/setup_conda.sh && conda activate osworld

# Minecraft（Python 3.10 + OpenJDK 8）
bash scripts/minecraft/setup_conda.sh && conda activate openha

# RLCard（Python 3.12）
bash scripts/rlcard/setup_conda.sh && conda activate rlcard
```

### API 配置

复制模板，填入端点与 key（任意 OpenAI 兼容接口均可）：

```bash
cp configs/api/default.yaml.example configs/api/default.yaml
```

```yaml
# configs/api/default.yaml（gitignored）
openai:
  base_url: https://api.openai.com/v1
  api_key: sk-your-key-here
```

也可以不建文件，改为导出环境变量 `OPENAI_BASE_URL` / `OPENAI_API_KEY`。

部分 reasoning 端点不接受旧的 `max_tokens`，要求改用 `max_completion_tokens`。这类模型在 `configs/api/model_overrides.json`（受版本控制的文件，直接编辑即可）中登记：

```json
{
  "gpt-5.5": { "use_max_completion_tokens": true }
}
```

每个 key 都是**完整 model id**——即你传给 `agent.model` 的那个精确值。查找规则：先精确匹配，再以最长前缀兜底（所以裸写 `"gpt-5"` 也能覆盖 `gpt-5.5`）。

### 使用

```bash
# 单任务
python -m anything2skill benchmark=osworld tasks.task_id=<UUID>
python -m anything2skill benchmark=minecraft tasks.task_id=<TASK_ID>
python -m anything2skill benchmark=doudizhu tasks.num_games=20

# N 进程并行
python -m anything2skill benchmark=osworld runner.num_envs=5

# 切换 agent 模式
python -m anything2skill benchmark=osworld agent.agent_mode=vanilla            # 无 skills baseline
python -m anything2skill benchmark=osworld agent.agent_mode=vanilla_tutorial   # 原始教程（跳过抽取）

# 启用 Reviser 修订循环（≥2 attempts，可选用不同 reviser 模型）
python -m anything2skill benchmark=osworld reviser.max_attempts=3
python -m anything2skill benchmark=osworld reviser.max_attempts=3 reviser.model=claude-sonnet-4

# 教程下载
python data_collection/osworld/download_tutorials.py
python data_collection/minecraft/download_tutorials.py
python data_collection/rlcard/download_tutorials.py

# 测试
python -m pytest anything2skill/tests/ -v
```

CLI 使用 Hydra，配置合并优先级：`api/default.yaml` < `config.yaml` < `benchmark/<name>.yaml` < CLI。

## Benchmarks

| 领域 | Benchmark | 说明 |
|---|---|---|
| GUI 自动化 | **OSWorld**（Ubuntu 桌面） | `pyautogui` 动作 |
| 开放世界游戏 | **OpenHA Minecraft** | 开放世界物理交互 |
| 策略 | **RLCard** doudizhu / mahjong | success-inferable 策略任务 |
| 边界诊断 | **RLCard** no-limit hold'em | 仅作 private-information boundary diagnostic |

## 目录结构

```
anything2skill/         # 主包（benchmark 无关核心）
  agent/                # 决策层：simple / phased / vanilla / vanilla_tutorial
  reviser/              # attempt 之间的 analyze→refine 循环
  benchmarks/           # 每个 benchmark 一个 BenchmarkKit
  parser/  vlm/  metrics/
  runner.py             # run_parallel + run_single_task + 双桶 run_name
  benchmark_kit.py      # BenchmarkKit ABC（新增 benchmark 的契约）
configs/                # Hydra YAML
data_tutorial/          # 教程素材（{tutorial_type}/{benchmark}/{task_id}）
data_collection/        # 教程采集脚本
skills_cache/           # 抽取出的 SKILL.md 缓存
results/                # 实验产物
docs/                   # 架构与开发者文档（见 docs/README.zh.md）
OSWorld/  OpenHA/  RLCard/   # benchmark 环境子模块
```

架构、数据流与实验产物布局见 [docs/README.zh.md](docs/README.zh.md)。

## 数据与来源

教程采自公开网络资源——官方文档、社区维护的 wiki、Q&A 页面与第三方 tutorial——以 HTML + `images/` 或 Playwright 滚动截图两种模态保存。

- **Provenance** —— 每条 guide 在 `{benchmark}_urls.json` 中记录 source URL 与 source category。
- **双轨发布** —— 仅当源 license 允许时再分发 cached raw guide contents；否则 release 只提供 task→URL 映射 + 元数据 + 采集脚本，使用者可自行复现采集。
- **复现** —— `python data_collection/{osworld,minecraft,rlcard}/download_tutorials.py`。
- **Takedown** —— rights holders 可申请从后续 release 中移除已缓存内容。

采集所得仅用于研究与评估。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/README.zh.md](docs/README.zh.md) | 架构、数据流、实验产物、概念速查、文档索引 |
| [docs/execution-flow.zh.md](docs/execution-flow.zh.md) | 从启动到结束的完整执行流程 |
| [docs/prompt-assembly.zh.md](docs/prompt-assembly.zh.md) | Kit prompt 如何拼装为 VLM 消息 |
| [docs/benchmark-adapter-guide.zh.md](docs/benchmark-adapter-guide.zh.md) | 编写新 `BenchmarkKit` 的步骤 |
| [docs/reviser-loop.zh.md](docs/reviser-loop.zh.md) | Reviser 双桶布局、`early_stop` 语义、配置项 |

## 引用

```bibtex
@misc{che2026mmg2skillagentsdistillinthewild,
  title         = {MMG2Skill: Can Agents Distill In-the-Wild Guides into Self-Evolving Skills?},
  author        = {Xinyu Che and Junqi Xiong and Yunfei Ge and Xinping Lei and Shihao Li and Hang Yan and Han Li and Yuanxing Zhang and Zhiqi Bai and Jinhua Hao and Ming Sun and Han Li and Jiaheng Liu},
  year          = {2026},
  eprint        = {2606.01993},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2606.01993},
}
```

## 许可证

以 [MIT License](LICENSE) 发布。

## 致谢

基于以下上游环境构建：

| 子模块 | 用途 | 上游 |
|---|---|---|
| `OSWorld/` | OSWorld 桌面环境 | [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld) |
| `OpenHA/` | Minecraft 物理后端 | [CraftJarvis/OpenHA](https://github.com/CraftJarvis/OpenHA) |
| `RLCard/` | 卡牌游戏环境 | [lemondinosaur/rlcard](https://github.com/lemondinosaur/rlcard) |
