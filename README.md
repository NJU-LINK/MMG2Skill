<p align="center">
<pre align="center">
<b>
███╗   ███╗███╗   ███╗ ██████╗ ██████╗ ███████╗██╗  ██╗██╗██╗     ██╗     
████╗ ████║████╗ ████║██╔════╝ ╚════██╗██╔════╝██║ ██╔╝██║██║     ██║     
██╔████╔██║██╔████╔██║██║  ███╗ █████╔╝███████╗█████╔╝ ██║██║     ██║     
██║╚██╔╝██║██║╚██╔╝██║██║   ██║██╔═══╝ ╚════██║██╔═██╗ ██║██║     ██║     
██║ ╚═╝ ██║██║ ╚═╝ ██║╚██████╔╝███████╗███████║██║  ██╗██║███████╗███████╗
╚═╝     ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝
</b>
</pre>
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
  <a href="docs/README.md"><img src="https://img.shields.io/badge/docs-Architecture-blue.svg?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Docs"></a>
</p>
<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.12-blue.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10 | 3.12"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="License: MIT"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#benchmarks">Benchmarks</a> &bull;
  <a href="docs/README.md">Architecture</a> &bull;
  <a href="#documentation">Documentation</a> &bull;
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <a href="README.md">English</a> &bull; <a href="README.zh.md">中文</a>
</p>

---

## Introduction

MMG2Skill automatically converts public multimodal tutorials into executable **Skills** (SOPs), and lets a **fixed** VLM agent continuously revise those Skills from its own interaction trajectories — **without ever reading the benchmark score**. It spans GUI automation, open-world games, and strategy card games behind a single benchmark-agnostic framework.

> The source package name remains `anything2skill`.

## Highlights

| | |
|---|---|
| **Tutorial → Skills extraction** | A VLM distills SOPs from HTML + image assets into cached `SKILL.md` files; a rendered-screenshot tutorial channel is also supported. |
| **Reviser revision loop** | An analyzer→refiner loop turns agent-visible trajectory diagnoses into skill edits across attempts, producing both early-stop and full-run views. |
| **Score-free updates** | Skill construction, execution, analysis, and refinement never read the benchmark score; scores are used only for offline evaluation. |
| **Four agent modes** | `simple` (skill-using), `phased` (planner→executor/reflector), `vanilla` (no-skills baseline), `vanilla_tutorial` (raw-tutorial baseline) — for clean ablations. |
| **Benchmark-agnostic** | `agent/` and `reviser/` depend on no concrete benchmark; a new benchmark only needs a `BenchmarkKit`. |

## Quick Start

### Installation

Each benchmark uses its own conda environment because dependencies differ substantially. Initialize the submodules first:

```bash
git submodule update --init OSWorld OpenHA RLCard
```

Then prepare the environment you need:

```bash
# OSWorld (Python 3.10)
bash scripts/osworld/setup_conda.sh && conda activate osworld

# Minecraft (Python 3.10 + OpenJDK 8)
bash scripts/minecraft/setup_conda.sh && conda activate openha

# RLCard (Python 3.12)
bash scripts/rlcard/setup_conda.sh && conda activate rlcard
```

### API configuration

Copy the template and fill in your endpoint + key (any OpenAI-compatible API works):

```bash
cp configs/api/default.yaml.example configs/api/default.yaml
```

```yaml
# configs/api/default.yaml  (gitignored)
openai:
  base_url: https://api.openai.com/v1
  api_key: sk-your-key-here
```

Or skip the file and export `OPENAI_BASE_URL` / `OPENAI_API_KEY` instead.

Some reasoning endpoints (the o-series, certain `ep-*` endpoints) reject the legacy `max_tokens` parameter and require `max_completion_tokens`. Register them in `configs/api/model_overrides.json` (a tracked file, edit it directly):

```json
{
  "gpt-5.5": { "use_max_completion_tokens": true }
}
```

Each key is the **complete model id** — the exact value you pass as `agent.model`. Lookup is exact-match first, then longest prefix as a fallback (so a bare `"gpt-5"` would also cover `gpt-5.5`).

### Usage

```bash
# Single task
python -m anything2skill benchmark=osworld tasks.task_id=<UUID>
python -m anything2skill benchmark=minecraft tasks.task_id=<TASK_ID>
python -m anything2skill benchmark=doudizhu tasks.num_games=20

# N-process parallelism
python -m anything2skill benchmark=osworld runner.num_envs=5

# Switch agent mode
python -m anything2skill benchmark=osworld agent.agent_mode=vanilla            # no-skills baseline
python -m anything2skill benchmark=osworld agent.agent_mode=vanilla_tutorial   # raw tutorial (skip extraction)

# Enable the Reviser revision loop (≥2 attempts; optionally a different reviser model)
python -m anything2skill benchmark=osworld reviser.max_attempts=3
python -m anything2skill benchmark=osworld reviser.max_attempts=3 reviser.model=claude-sonnet-4

# Download tutorials
python data_collection/osworld/download_tutorials.py
python data_collection/minecraft/download_tutorials.py
python data_collection/rlcard/download_tutorials.py

# Tests
python -m pytest anything2skill/tests/ -v
```

The CLI uses Hydra. Config merge priority: `api/default.yaml` < `config.yaml` < `benchmark/<name>.yaml` < CLI.

## Benchmarks

| Domain | Benchmark | Notes |
|---|---|---|
| GUI automation | **OSWorld** (Ubuntu desktop) | `pyautogui` actions |
| Open-world game | **OpenHA Minecraft** | open-world physical interaction |
| Strategy | **RLCard** doudizhu / mahjong | success-inferable strategy tasks |
| Boundary diagnostic | **RLCard** no-limit hold'em | private-information boundary diagnostic only |

## Project Structure

```
anything2skill/         # main package (benchmark-agnostic core)
  agent/                # decision layer: simple / phased / vanilla / vanilla_tutorial
  reviser/              # analyze→refine loop between attempts
  benchmarks/           # one BenchmarkKit per benchmark
  parser/  vlm/  metrics/
  runner.py             # run_parallel + run_single_task + dual-bucket run_name
  benchmark_kit.py      # BenchmarkKit ABC (contract for new benchmarks)
configs/                # Hydra YAML
data_tutorial/          # tutorial assets ({tutorial_type}/{benchmark}/{task_id})
data_collection/        # tutorial collection scripts
skills_cache/           # extracted SKILL.md cache
results/                # experiment artifacts
docs/                   # architecture & developer docs (see docs/README.md)
OSWorld/  OpenHA/  RLCard/   # benchmark env submodules
```

See [docs/README.md](docs/README.md) for the architecture, data flow, and experiment-artifact layout.

## Data & Provenance

Tutorials are collected from public web materials — official docs, community wikis, Q&A pages, and third-party tutorials — captured as HTML + `images/` or Playwright scrolling screenshots.

- **Provenance** — every guide records its source URL and category in `{benchmark}_urls.json`.
- **Dual-track release** — cached raw guide contents are redistributed only when the source license permits; otherwise the release ships task→URL mappings + metadata + collection scripts so users can re-collect.
- **Reproduce** — `python data_collection/{osworld,minecraft,rlcard}/download_tutorials.py`.
- **Takedown** — rights holders may request removal of cached contents from future releases.

Collected materials are used for research and evaluation only.

## Documentation

| Document | Contents |
|---|---|
| [docs/README.md](docs/README.md) | Architecture, data flow, experiment artifacts, concept glossary, doc index |
| [docs/execution-flow.md](docs/execution-flow.md) | Full execution flow from startup to completion |
| [docs/prompt-assembly.md](docs/prompt-assembly.md) | How Kit prompts are assembled into VLM messages |
| [docs/benchmark-adapter-guide.md](docs/benchmark-adapter-guide.md) | Steps for writing a new `BenchmarkKit` |
| [docs/reviser-loop.md](docs/reviser-loop.md) | Reviser dual-bucket layout, `early_stop` semantics, configuration |

## Citation

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

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Built on these upstream environments:

| Submodule | Purpose | Upstream |
|---|---|---|
| `OSWorld/` | OSWorld desktop environment | [xlang-ai/OSWorld](https://github.com/xlang-ai/OSWorld) |
| `OpenHA/` | Minecraft physical backend | [CraftJarvis/OpenHA](https://github.com/CraftJarvis/OpenHA) |
| `RLCard/` | Card-game environments | [lemondinosaur/rlcard](https://github.com/lemondinosaur/rlcard) |
