# Benchmark Adapter Guide

This document is for developers who need to integrate a new benchmark. It explains how to write a `BenchmarkKit` subclass and how the prompts you provide flow into the agent layer.

---

## 1. Responsibility Boundary

**You are responsible for (BenchmarkKit)**: providing `system_prompt` text, providing optional text such as `bridging_text`, encoding observations into VLM content blocks, parsing LLM responses into action lists, creating environment instances, and collecting task lists.

**The framework is responsible for (MessageBuilder)**: filling `system_prompt` into templates, orchestrating multi-turn history, assembling skills/planner instructions, controlling message structure and role order, and managing embedded skill images.

**Core principle**: a Kit only provides domain text and behavior. It does not assemble `list[dict]` message lists.

---

## 2. Required Implementations

### `system_prompt` (@property)

Domain identity + action format + special signal definitions.

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

**Should include**: what the environment is, the action format, and what the special signals mean.
**Should not include**: instructions for using skills, planner format, or multi-turn history management. These are handled by framework templates.

### `encode_observation(obs: dict) -> list[dict]`

Convert environment observations into content blocks that the VLM can understand.

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

Extract executable actions from the LLM response. It must recognize the framework signals `DONE` and `FAIL`.

```python
def parse_actions(self, response: str) -> list[str]:
    # Recognize DONE/FAIL/WAIT
    for signal in ("DONE", "FAIL", "WAIT"):
        if signal in response:
            return [signal]
    # Extract code blocks or other formats
    ...
```

### `create_env(env_cfg: dict) -> EnvironmentInterface`

Create and return an environment instance. Called once by each worker.

### `collect_tasks(cfg: dict) -> list[TaskDescriptor]`

Collect the task list from the configuration. You may return a `TaskDescriptor` subclass.

---

## 3. Optional Properties (With Defaults)

| Property | Default | Purpose |
|----------|--------|------|
| `bridging_text` | `"Given the current observation. What's the next step?"` | Bridging text between historical turns |
| `reflection_guidance` | `""` | Domain guidance for reflection mode |
| `planner_guidance` | `""` | Domain context for the Planner |
| `skill_extraction_guidance` | `""` | Domain guidance for skill extraction |

Override them only when the defaults do not meet your needs.

---

## 4. How Your Prompt Flows Into the Agent

Using `system_prompt` as an example, the full flow is:

```
1. You define constants in benchmarks/<name>/prompts.py
   SYSTEM_PROMPT = "You are an agent that..."

2. kit.py references the constant as a property
   @property
   def system_prompt(self) -> str:
       return SYSTEM_PROMPT

3. MessageBuilder reads the property when building messages
   system_text = SIMPLE_ACTION_SYSTEM_TMPL.format(
       domain_system_prompt=self.kit.system_prompt
   )

4. Template assembly produces the complete system message
   → your domain text + the framework's SOP/observation/reasoning instructions
```

Injection locations for each property:

| Kit Property | Injection Location |
|---|---|
| `kit.system_prompt` | system messages (Autonomous/Phased/Reflection) |
| `kit.bridging_text` | Prefix of historical user messages |
| `kit.reflection_guidance` | Tail of the Reflection system message |
| `kit.planner_guidance` | Middle of the Planner system message |
| `kit.skill_extraction_guidance` | Middle of the Extraction system message |

See Section 3 of [prompt-assembly.md](prompt-assembly.md) for the detailed structure diagram.

---

## 5. Registration

```python
from anything2skill.benchmarks.registry import register_kit

@register_kit("myworld")
class MyWorldKit(BenchmarkKit):
    ...
```

After registration, `benchmark=myworld` can be used in the configuration.

---

## 6. Configuration

Create `configs/benchmark/myworld.yaml`; the first line must be `# @package _global_`:

```yaml
# @package _global_

benchmark: myworld

env:
  # Environment parameters
  ...

agent:
  agent_mode: simple  # or phased
  max_steps: 15

data:
  dir: data/myworld

tasks:
  # Task filtering parameters
  ...
```

---

## 7. Checklist

1. [ ] `benchmarks/<name>/__init__.py` — empty file
2. [ ] `benchmarks/<name>/prompts.py` — domain prompt constants (UPPER_SNAKE_CASE names)
3. [ ] `benchmarks/<name>/kit.py` — `BenchmarkKit` subclass + `@register_kit`
4. [ ] `configs/benchmark/<name>.yaml` — Hydra configuration
5. [ ] If you need a custom `TaskDescriptor`, create a `@dataclass` subclass
6. [ ] Validate prompt assembly with `mock_prompt_viewer`
7. [ ] If you share a runtime with an existing benchmark (for example, card games sharing the RLCard env), reuse `benchmarks/rlcard_common.py:BaseRLCardEnvWrapper` or provide a shared helper in the same style; do **not** put such helpers under `agent/` or `reviser/`
8. [ ] Prepare an independent conda environment script at `scripts/<name>/setup_conda.sh` (see `scripts/osworld/`, `scripts/minecraft/`, and `scripts/rlcard/`)
9. [ ] **No files under `agent/` or `reviser/` need to be modified**

> The action format does not have to use `` ```python `` blocks. Minecraft / RLCard both use `<action>...</action>` tags, which are friendlier to thinking models. The kit decides the format, but `parse_actions()` must recognize the framework signals `DONE` / `FAIL`.

Reference implementations:
- GUI / single-image observations: `benchmarks/osworld/kit.py`
- `<action>` tags + physical interaction: `benchmarks/minecraft/kit.py`
- Text observations + shared wrapper: `benchmarks/doudizhu/kit.py` (+ `benchmarks/rlcard_common.py`)

---

## 8. Validate With mock_prompt_viewer

```python
from anything2skill.tests.mock_prompt_viewer import view_prompts
from anything2skill.benchmarks.myworld.kit import MyWorldKit

# Pass your Kit to inspect prompt assembly for all scenarios
view_prompts(MyWorldKit())

# Inspect only the planner scenario
view_prompts(MyWorldKit(), scenarios=["planner"])

# Pass a real observation to validate encode_observation output
view_prompts(MyWorldKit(), obs={"your_obs_key": real_data})

# Export JSON to prompt_snapshots/ for offline inspection
view_prompts(MyWorldKit(), export=True)
```

Check whether your `system_prompt`, `bridging_text`, and other text appear in the expected positions in the output.
