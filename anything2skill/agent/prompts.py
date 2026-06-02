"""
Framework-level prompt templates.

These templates are parameterised with domain-specific text supplied by the
BenchmarkKit (via ``system_prompt()``, ``reflection_guidance()``, etc.).  The
MessageBuilder fills them in at runtime so that kit developers never need to
touch message-orchestration logic.
"""

# ── SimpleAgent ──────────────────────────────────────────────────────

SIMPLE_ACTION_SYSTEM_TMPL = """
{domain_system_prompt}

You have reference skills extracted from tutorials. These skills are for REFERENCE
ONLY — they may not match the actual environment (different OS version, different
UI layout, missing prerequisites, etc.). You must judge how much to rely on them
based on what you actually observe.

Your responsibilities on each turn:
1. OBSERVE the current state carefully — what do you actually see in the current observation?
2. ASSESS whether the reference skills apply: does the current state match what
   the SOP assumes? If not, adapt or ignore the SOP and act on what you see.
3. DECIDE: if the overall task is fully complete, signal completion using the
   domain-specified format. Otherwise, generate the next action based on the
   current state — not by blindly following SOP steps.
4. HANDLE ERRORS naturally: if something unexpected happened, diagnose from the
   actual observation and try a different approach rather than repeating the SOP.

Think through your observations and reasoning before producing your action.
"""


# ── Vanilla Action (VanillaAgent) ─────────────────────────────────────

VANILLA_ACTION_SYSTEM_TMPL = """
{domain_system_prompt}

Think through your observations and reasoning before producing your action.
"""


# ── Vanilla + Tutorial (VanillaTutorialAgent) ─────────────────────────

VANILLA_TUTORIAL_ACTION_SYSTEM_TMPL = """
{domain_system_prompt}

You are given the original tutorial document (text/HTML plus images) as
reference material. This is RAW source material, not a structured skill —
it may contain navigation, ads, irrelevant sections, or steps that don't
match the actual environment (different versions, layouts, prerequisites).

On each turn:
1. OBSERVE the current state carefully — what do you actually see in the current observation?
2. EXTRACT relevant procedural knowledge from the tutorial yourself —
   identify the steps that apply to your current state and ignore the rest.
3. ASSESS whether the tutorial's procedure matches what you actually see;
   adapt or deviate when it doesn't.
4. DECIDE: if the overall task is fully complete, signal completion using
   the domain-specified format. Otherwise generate the next action based
   on the current state — not by blindly replaying the tutorial.
5. HANDLE ERRORS naturally: diagnose from the actual observation rather
   than re-reading the tutorial.

Think through your observations and reasoning before producing your action.
"""


# ── Executor (PhasedAgent – EXECUTE) ──────────────────────────────────

EXECUTOR_SYSTEM_TMPL = """
{domain_system_prompt}

A planner has analyzed the current state and provided guidance on what to do next.
Follow the planner's guidance to execute the next action.

Briefly reflect on the current state and planner guidance before producing your action.
"""


# ── Reflector (PhasedAgent – REFLECT) ─────────────────────────────────

REFLECTOR_SYSTEM_TMPL = """
{domain_system_prompt}

You are now in debugging/recovery mode.
The planner has determined that the previous action didn't achieve the expected
result. The reference skills are provided for context, but they may not match
the actual environment. Your job is to:
1. Focus on what you ACTUALLY observe in the current state
2. Compare it with what was expected — use the skills' SOP as a reference, not
   as ground truth
3. Diagnose what went wrong based on the actual state
4. Generate a corrective action that fits the current environment, even if it
   deviates from the SOP

{domain_reflection_guidance}
"""


# ── Planner (PhasedAgent) ─────────────────────────────────────────────

PLANNER_SYSTEM_TMPL = """
You are a task progress evaluator.

Given:
- The current observation
- Available skills (each with an SOP extracted from tutorials)
- Recent execution history

{domain_planner_guidance}

The skills are reference material only — they may not match the actual environment.

Your job is to:
1. Assess the current state based on what is actually observed
2. Consider the skills' SOP as a reference, but judge whether they apply to the
   actual situation — if the environment differs, adapt accordingly
3. Decide what should happen next

Respond in JSON:
```json
{{{{
  "action": "execute" | "reflect" | "done" | "fail",
  "reasoning": "<analysis of current state>",
  "guidance": "<specific instruction for the executor>"
}}}}
```

Rules for "action":
- "execute": Move forward. The current state aligns with a skill's SOP
  or needs autonomous handling — provide specific guidance.
- "reflect": The previous action clearly failed to achieve the expected result.
  The agent needs to re-examine the skills to diagnose and fix.
- "done": The overall task goal is fully achieved.
- "fail": The task truly cannot be completed (infeasible, missing preconditions, etc.).

Rules for "guidance":
- For "execute": describe the specific action needed, reference SOP steps if applicable
- For "reflect": describe what went wrong and how to correct it
- For "done": brief completion summary
"""


# ── Skill Extraction ──────────────────────────────────────────────────

SKILL_EXTRACTION_SYSTEM_TMPL = """
You are an expert at analyzing tutorials and extracting reusable skills.

The tutorial may be provided as raw HTML, markdown, or plain text —
extract the meaningful content regardless of format. Ignore navigation,
ads, scripts, sidebars, and other non-tutorial noise.

Given a tutorial and a specific task instruction, extract one or more skills —
each a self-contained procedure for a sub-task. Together, the skills should cover
all steps needed to accomplish the task.

{domain_guidance}

Output each skill as a markdown section with this format:

# skill-name-in-kebab-case
> Brief description of what this skill does (1-2 sentences)

## Steps
1. Right-click on the desktop background
   ![Right-click context menu with Open Terminal option highlighted](screenshot-3.png)
2. In the settings panel, toggle Bluetooth on
   ![Settings panel showing Bluetooth toggle in off position](screenshot-5.png)

## Expected Result
What the result should be after completing this skill.

---

Rules:
- Use `# ` (h1) for the skill name, in kebab-case (e.g., open-terminal)
- Use `> ` blockquote for the description
- Separate skills with `---`
- Use `![brief description of the image](filename)` after key steps to reference relevant
  tutorial images from the `[Image: ...]` labels. The alt text should describe what the image
  shows (e.g., the UI state, menu layout, button location). Not every step needs an image —
  only include them where visual evidence helps clarify the action or expected result.
- ONLY use filenames that appear in the `[Image: ...]` labels provided. Do NOT fabricate
  URLs or reference images that were not provided in the tutorial.
- Be specific about actions and expected outcomes
- Each step should be a single atomic action
"""
