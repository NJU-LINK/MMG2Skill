# Prompt Assembly Flow

This document is the central reference for explaining **how the domain prompts provided by `BenchmarkKit` are assembled by the framework into complete messages sent to the VLM**.

> If you are a benchmark adapter, read this document first, then read [benchmark-adapter-guide.md](benchmark-adapter-guide.md).

---

## 1. Template Architecture

All prompt templates are defined in `anything2skill/agent/prompts.py` as string constants, such as `SIMPLE_ACTION_SYSTEM_TMPL`.

Templates are filled with Python's `.format()`. Placeholder naming convention:

| Prefix | Source | Example |
|------|------|------|
| `{domain_*}` | BenchmarkKit property | `{domain_system_prompt}` ← `kit.system_prompt` |

**Core principle**: the Kit only provides domain text (`system_prompt`, `bridging_text`, etc.); `MessageBuilder` is responsible for filling them into templates and assembling messages. Kit developers do not need to touch message assembly logic.

---

## 2. Placeholder Mapping Table

| Placeholder | Data Source | Usage Location |
|--------|---------|---------|
| `{domain_system_prompt}` | `kit.system_prompt` | system messages for SimpleAgent/Executor/Reflector/Vanilla/VanillaTutorial |
| `{domain_reflection_guidance}` | `kit.reflection_guidance` | Reflector system message |
| `{domain_planner_guidance}` | `kit.planner_guidance` | Planner system message |
| `{domain_guidance}` | `kit.skill_extraction_guidance` | Skill Extraction system message |

Content in user messages (instruction, skills, planner guidance, etc.) is assembled directly by `MessageBuilder` and does not go through template placeholders.

---

## 3. Seven Message Assembly Paths

`MessageBuilder` provides 7 builder methods, each corresponding to one scenario.

### Unified Style

All agents follow the OSWorld style:
- **system**: only the domain prompt + role instructions
- **first user turn**: skills (if any) + `# Your Task` + instruction + bridging_text (multi-turn) or directly followed by obs (single-turn)
- **subsequent user turns**: bridging_text + obs

### 3.1 SimpleAgent

**Entry point**: `build_action_messages(skills, obs, history, instruction)`

```
[system]: SIMPLE_ACTION_SYSTEM_TMPL (domain_prompt + skill usage guidance + four responsibilities)
[user]:   skills_blocks (text + inline images) + "# Your Task" + instruction + bridging_text + obs_1
[asst]:   response_1
[user]:   bridging_text + obs_current
```

**Characteristics**: all skills are in the first user turn; multi-turn history.

### 3.2 VanillaAgent

**Entry point**: `build_vanilla_messages(obs, history, instruction)`

```
[system]: VANILLA_ACTION_SYSTEM_TMPL (domain_prompt + response format)
[user]:   "# Your Task" + instruction + bridging_text + obs_1
[asst]:   response_1
[user]:   bridging_text + obs_current
```

**Characteristics**: no skills; same structure as SimpleAgent.

### 3.3 VanillaTutorialAgent

**Entry point**: `build_vanilla_tutorial_messages(tutorial, max_images, obs, history, instruction)`

```
[system]: VANILLA_TUTORIAL_ACTION_SYSTEM_TMPL (domain_prompt + "RAW source material" notice + five responsibilities)
[user]:   "## Reference Tutorial" + tutorial.body
          + "Tutorial images (N):" + interleaved [Image: filename] + image_url (≤ max_images; 0 = unlimited)
          + "# Your Task" + instruction + bridging_text + obs_1
[asst]:   response_1
[user]:   bridging_text + obs_current
```

**Characteristics**: skips skill extraction and feeds the raw tutorial body + images directly. When `tutorial=None`, the entire tutorial block is omitted, making it equivalent to VanillaAgent. `max_images <= 0` means no truncation. The system prompt explicitly tells the model that the tutorial is RAW source material, may include steps inconsistent with the current environment, and must be filtered and adapted by the model itself.

### 3.4 Executor (PhasedAgent EXECUTE)

**Entry point**: `build_executor_messages(obs, instruction, decision)`

```
[system]: EXECUTOR_SYSTEM_TMPL (domain_prompt + follow planner guidance)
[user]:   "## Your Task" + instruction + planner_reasoning + planner_guidance + bridging_text + obs_current
```

**Characteristics**: no skills, no history, single turn. The Executor focuses on executing the planner's instructions.

### 3.5 Reflector (PhasedAgent REFLECT)

**Entry point**: `build_reflect_messages(skills, obs, history, instruction, decision)`

```
[system]: REFLECTOR_SYSTEM_TMPL (domain_prompt + diagnostic flow + reflection_guidance)
[user]:   skills_blocks (text + inline images) + "# Your Task" + instruction
          + planner_reasoning + planner_guidance
          + "## Recent Steps"
          + Step 1: action + obs_screenshot
          + Step 2: action + obs_screenshot
          + "## Current Observation" + screenshot_current
```

**Characteristics**: all skills + recent trajectory (aggregated by step as action+obs), single turn, used for comparative diagnosis.

### 3.6 Planner (SoftPlanner Evaluation)

**Entry point**: `build_planner_messages(skills, state, obs, instruction)`

```
[system]: PLANNER_SYSTEM_TMPL (evaluator role + planner_guidance + JSON output format)
[user]:   skills_blocks (text + inline images) + "# Your Task" + instruction
          + "## Recent Steps"
          + Step 1: action + obs_screenshot
          + Step 2: action + obs_screenshot
          + "## Current Observation" + screenshot_current
```

**Characteristics**: the Planner outputs JSON `{action, reasoning, guidance}` and does not select skills.

### 3.7 Skill Extraction (Tutorial→Skill)

**Entry point**: `build_skill_extraction_messages(tutorial_content, instruction, image_entries)`

```
[system]: SKILL_EXTRACTION_SYSTEM_TMPL (skill extraction expert + domain guidance + output format)
[user]:   "TASK: {instruction}\n\nTUTORIAL:\n{tutorial_content}" + image list
```

---

## 4. History Embedding

### Multi-Turn History (SimpleAgent / VanillaAgent / VanillaTutorialAgent)

`_build_multiturn()` embeds trajectory history as multi-turn dialogue. The first user turn is customized by the caller; subsequent turns are uniformly `bridging_text + obs`:

```
[user: first_content + obs₁]  [assistant: response₁]    ← historical turn 1
[user: bridging + obs₂]       [assistant: response₂]    ← historical turn 2
[user: bridging + obs_current]                           ← current turn
```

- The `history_window` configuration controls how many historical entries are injected (default: 3)
- `bridging_text` is the bridging text provided by the Kit (in OSWorld: `"Given the screenshot below. What's the next step?"`)

### Step-Aggregated History (Planner / Reflector)

`_build_history_steps()` aggregates history by step into single-turn content:

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

## 5. Skill Image Embedding

Inline image references in Skill content, such as `![alt](filename)`, are parsed into interleaved text + image blocks when injected into the agent:

- The original `![alt](filename)` reference is preserved as a text block before each image, retaining descriptive alt text
- The image immediately follows in `{"type": "image_url", "image_url": {"url": "data:image/...", "detail": "high"}}` format
- Images are inserted inline in the order they are referenced in the content, rather than being piled up at the end

---

## 6. OSWorld Concrete Example

Using the first SimpleAgent call in OSWorld as an example, trace the full assembly process.

**Domain text provided by the Kit** (defined in `benchmarks/osworld/prompts.py`):

```python
# kit.system_prompt →
"You are an agent that performs desktop computer tasks on Ubuntu..."

# kit.bridging_text →
"Given the screenshot below. What's the next step?"

# kit.planner_guidance →
"Observations are screenshots of an Ubuntu desktop. Actions are pyautogui Python code."
```

**Final assembled messages**:

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
  // When there are no inline image references, the whole skill content is one text block
  {"type": "text", "text": "# Your Task\nOpen Settings and turn on Bluetooth\n\n
   Given the screenshot below. What's the next step?"}
  + [screenshot image block]                          ← current observation
```

> Minecraft and RLCard use the same assembly structure as OSWorld; only `system_prompt` / `bridging_text` / action format differ. The latter two use `<action>...</action>` tags. You can verify this yourself with `view_prompts(MinecraftKit())` or `view_prompts(BlackjackKit())`.

---

## 7. Visualization Tool

`anything2skill/tests/mock_prompt_viewer.py` provides the `view_prompts()` function. Pass any Kit to inspect the assembly output.

### Basic Usage

```python
from anything2skill.tests.mock_prompt_viewer import view_prompts
from anything2skill.benchmarks.osworld.kit import OSWorldKit

# Pass a Kit and inspect all scenarios (obs uses a generic placeholder)
view_prompts(OSWorldKit())

# Inspect only specific scenarios
view_prompts(OSWorldKit(), scenarios=["planner"])
view_prompts(OSWorldKit(), scenarios=["simple", "executor"])
```

Available scenarios: `simple`, `vanilla`, `executor`, `reflect`, `planner`, `extraction`. `VanillaTutorialAgent` is not yet integrated into the viewer; you can directly call `MessageBuilder.build_vanilla_tutorial_messages()` to verify it yourself.

### Passing a Real Observation

By default, a `<Observation>` text placeholder is used. To validate how a real obs is encoded:

```python
# Passing a real obs dict calls kit.encode_observation() for encoding
view_prompts(OSWorldKit(), obs={"screenshot": png_bytes})
```

### Export JSON

```python
# Export to anything2skill/tests/prompt_snapshots/*.json
view_prompts(OSWorldKit(), export=True)
```

Each scenario generates one JSON file (for example, `simple_with_history.json`) containing the complete `messages` array sent to the VLM, which can be inspected offline or used with any compatible API client for validation.

### Quick CLI Run

```bash
# Uses OSWorldKit by default and prints all scenarios
python -m anything2skill.tests.mock_prompt_viewer
```

---

## 8. Source File Index

| File | Responsibility |
|------|------|
| `agent/prompts.py` | All prompt template constants |
| `agent/message_builder.py` | `MessageBuilder` — message assembly logic |
| `agent/skill_utils.py` | Skills formatting helper functions |
| `benchmark_kit.py` | `BenchmarkKit` ABC — properties that Kits must implement |
| `benchmarks/osworld/prompts.py` | OSWorld domain prompt text |
| `benchmarks/osworld/kit.py` | OSWorldKit — reference implementation |
