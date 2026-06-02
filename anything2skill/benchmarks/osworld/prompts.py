"""Domain-specific prompt text for OSWorld GUI benchmark.

These strings are referenced by OSWorldKit's domain hook methods.
Edit them here to tune the agent's behavior on OSWorld tasks.
"""

# ── System prompt (domain identity + action format) ────────────────────

SYSTEM_PROMPT = """
You are an agent that performs desktop computer tasks on an Ubuntu desktop.

You use `pyautogui` to interact with the desktop.
DO NOT use `pyautogui.locateCenterOnScreen` since we have no reference images.
DO NOT use `pyautogui.screenshot()`.

Return one or multiple lines of python code to perform the action.
Be time efficient. When predicting multiple lines of code, add small sleeps
like `time.sleep(0.5)` between them. Each time you need to predict complete
code — no variables or functions can be shared from history.

You need to specify coordinates yourself based on the current screenshot.
Be careful to ensure coordinates are correct.

Return code inside a code block:
```python
# your code here
```

Special codes:
- ```WAIT``` — when you need to wait for something to load or appear
- ```FAIL``` — when the task truly cannot be done (try hard before using this)
- ```DONE``` — when the overall task goal is fully achieved

My computer's password is '{client_password}', feel free to use it for sudo.
"""

# ── Bridging text (history turn transition) ────────────────────────────

BRIDGING_TEXT = """
Given the screenshot below. What's the next step?
"""

# ── Reflection guidance ────────────────────────────────────────────────

REFLECTION_GUIDANCE = """
Re-examine the skill's SOP and tutorial reference images carefully.
Compare the expected procedure with what you see on the current screenshot.
Specify coordinates from the current screenshot for corrective pyautogui code.
"""

# ── Planner guidance ──────────────────────────────────────────────────

PLANNER_GUIDANCE = """
Observations are screenshots of an Ubuntu desktop.
Actions are pyautogui Python code.
"""

# ── Skill extraction guidance ─────────────────────────────────────────

SKILL_EXTRACTION_GUIDANCE = """
Target environment: Ubuntu 22.04 desktop. Every skill is a concrete
procedure the agent will execute on that system — not on Windows, not
on macOS, not on a distro-agnostic "Linux". Author every step assuming
Ubuntu 22.04 conventions (apt package manager, systemd, /home/<user>/
home directories, GNOME desktop).

Action space: the agent executes each step as `pyautogui` Python code
against the live desktop (keyboard, mouse, hotkeys); there is no shell
API, no filesystem API, and no direct process control outside of what
pyautogui can drive through the UI. Author every step so that it can
be expressed as pyautogui calls — e.g. "open a terminal via
`pyautogui.hotkey('ctrl', 'alt', 't')` then type the command with
`pyautogui.typewrite(...)` and press Enter", rather than "run `apt
install ...`" in the abstract. If a step fundamentally cannot be
driven through pyautogui, state that explicitly so the agent doesn't
silently fake it.

- If the tutorial uses Windows/macOS tools, translate to the Ubuntu
  22.04 equivalent when the tutorial provides enough information to do so
- Replace placeholder paths with concrete paths only when they are given by
  the task instruction, tutorial, or visible desktop state
"""

# ── Reviser guidance ─────────────────────────────────────────────────

REVISER_GUIDANCE = """
Environment: Ubuntu 22.04 desktop. Actions are pyautogui Python code executed
via exec(). Observations are desktop screenshots (PNG).

Common failure patterns to look for in the trajectory:
- GUI interaction loops: the agent repeatedly clicks at wrong coordinates
  or gets stuck navigating menus / file managers without making progress
  toward the goal visible on screen.
- Fragile keyboard simulation: pyautogui.typewrite() fails on special
  characters and very long shell commands — the agent must split such
  input across multiple typewrite calls or paste via the clipboard
  (xdotool / pyperclip), still going through the GUI surface.
- Placeholder paths: the agent emits commands containing <path-to-X> or
  /path/to/... that are not real paths on the system.
"""
