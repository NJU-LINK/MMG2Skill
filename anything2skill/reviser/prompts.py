"""Prompt templates for the reviser two-phase pipeline.

All analyzer templates output XML (``<summary>`` / ``<root_cause>``) — LLMs
emit XML with fewer escaping issues than JSON, and per-tag regex fallback
stays reliable even when the model misbehaves.

The analyzer templates intentionally never mention skills or the tutorial:
phase 1's job is pure trajectory reading. Skill / tutorial alignment is
the refiner's responsibility.
"""

from __future__ import annotations

# ── Phase 1a: Rolling summary (intermediate chunks) ──────────────────────

REVISER_CHUNK_SYSTEM_TMPL = """
You are a trajectory analyst reading the execution trace of an AI agent.

{domain_reviser_guidance}

## How the trace is fed to you
The trace is delivered in fixed-size windows of predict-turns (one turn
= one agent LLM call). In each round you receive:
- the task instruction,
- (if any) a rolling summary of earlier turns,
- on the FIRST window only, the initial observation the agent
  perceived before it made any action — this is NOT a turn; do not
  index it as Turn 0 in `<where>`, only use it to ground the agent's
  opening decision,
- the detailed content of the current window: for every turn, the
  agent's own response (possibly truncated), the action(s) it
  emitted, and the saved observation for the turn's last action.

The saved observation is whatever the benchmark kit returns — it may
be an image, a block of text, or both, depending on what the agent
actually perceives in that domain.

You see exactly what the agent itself saw or wrote — no controller
errors, return codes, or env-side `done` flags. If an action failed
silently, the only evidence is the post-action observation, just like
the next agent's evidence will be.

## Your job this round
This is NOT the final round. Update the rolling summary so the next
round can continue the analysis without re-reading earlier turns.

Focus on:
- what the agent attempted and whether it succeeded,
- state transitions that matter (GUI opened / closed, inventory changed,
  new error emerged, etc.),
- recurring patterns (same failure loop, oscillating corrections, etc.),
- local sub-steps that clearly *worked* — the next round will need to
  know what to preserve, not only what went wrong.

Note: if the agent emits any terminal "cannot finish / task infeasible"
signal, treat it as a hypothesis from the agent, not a verdict. Record
the agent's justification verbatim so the next round can check it
against the observations — some tasks are genuinely infeasible and
bailing out is the correct answer. The exact syntax for such signals
is benchmark-specific; rely on the agent's wording and any domain hints
above to recognise them.

## Output format
Emit EXACTLY one block, nothing else:

<summary>
Plain prose describing the trajectory so far. Reference turns by index.
Keep it under {rolling_summary_char_limit} characters.
</summary>

Rules:
- Base the summary only on what is visible in the trace. Do NOT invent
  external context or information not shown.
- No markdown, no JSON, no preamble before the tag. Just `<summary>...</summary>`.
"""


# ── Phase 1b: Final round / single-chunk root-cause XML ──────────────────

REVISER_FINAL_SYSTEM_TMPL = """
You are a trajectory analyst reading the final window of an AI agent's
execution trace. Your job is to produce a root-cause analysis as XML.

{domain_reviser_guidance}

## How the trace is fed to you
The task instruction, (optionally) a rolling summary of earlier turns,
and the detailed content of the final window: for every turn, the
agent's own response (possibly truncated), the action(s) it emitted,
and the saved observation for the turn's last action. If this is also
the first window (single-chunk run), you additionally receive the
initial observation before Turn 1 — that block is NOT a turn; do not
index it as Turn 0 in `<where>`, only use it to ground the agent's
opening decision.

The saved observation is whatever the benchmark kit returns — it may
be an image, a block of text, or both, depending on what the agent
actually perceives in that domain.

You see exactly what the agent itself saw or wrote — no controller
errors, return codes, or env-side `done` flags. If an action failed
silently, the only evidence is the post-action observation, just like
the next agent's evidence will be.

## Judging the outcome (no grader available)
You do NOT have access to any rule-based grader. If the agent emitted
a terminal "done / task completed" signal, that is a self-claim — do
not assume the task actually succeeded just because the agent said so.
Conversely, a terminal "cannot finish / task infeasible" signal is a
hypothesis, not a verdict: some tasks are genuinely infeasible
(missing prerequisite, resource unavailable, contradictory instruction,
required tool not installed, etc.) and bailing out with well-grounded
reasoning is the *correct* output. The exact syntax of these terminal
signals is benchmark-specific; recognise them from the agent's wording
and any domain hints above, not from any assumed token.

You must therefore form your own judgment from the visible evidence.
Walk through the trace sub-step by sub-step: which intermediate goals
reached verifiable state, which did not, and whether the final
observation is consistent with the task having been completed.

## What to emit
A `<root_cause>` XML block describing:
1. a concise `<trajectory_summary>` of the whole run (what the agent
   attempted, the path it took, the final state it left behind);
2. `<what_worked>` — specific sub-steps that were executed correctly,
   listed as separate `<item>` entries. These must stay stable across
   refinements, so be concrete: quote the action / observation that
   evidences the success. Populate this even when the overall outcome
   is `likely_failure` — local correctness matters for preservation.
3. `<issues>` — concrete problems you observed, with `<where>`,
   `<evidence>`, and `<cause>`. May be empty if nothing is broken.
4. a final `<outcome_assessment>` judging the run as a whole. Put this
   LAST so your verdict follows from the evidence you just listed,
   not the other way around.

## Output format
Emit EXACTLY one block, nothing else:

<root_cause>
  <trajectory_summary>
  Plain prose summary of the trajectory. Reference turns by index where
  helpful. Keep under {rolling_summary_char_limit} characters.
  </trajectory_summary>
  <what_worked>
    <item>Concrete sub-step that was done correctly, grounded in evidence (turn index + quote / observation).</item>
    <!-- 0 or more item blocks. -->
  </what_worked>
  <issues>
    <issue>
      <where>turn 3 / turns 5-7 / etc.</where>
      <evidence>What you saw in the trace (quote or paraphrase the response, action, or observation). Be specific.</evidence>
      <cause>Why that behavior happened, inferred from the evidence.</cause>
    </issue>
    <!-- 0 or more issue blocks. If the run looks clean, it is OK to
         emit zero issues. -->
  </issues>
  <outcome_assessment value="likely_success">One sentence justifying the chosen value, grounded in the evidence above.</outcome_assessment>
</root_cause>

Rules:
- `outcome_assessment value` MUST be exactly one of
  `likely_success`, `uncertain`, `likely_failure`. Use `uncertain`
  when the evidence is mixed or you cannot confidently tell whether
  the task completed — do NOT force a binary verdict.
- Base every claim on concrete evidence from the trace (turn index,
  response quote, action, or observation).
- Use plain text inside tags. No markdown, no JSON, no CDATA sections.
- Do not emit anything before `<root_cause>` or after `</root_cause>`.
"""


# ── Phase 2: Refiner ─────────────────────────────────────────────────────

REVISER_REFINE_SYSTEM_TMPL = """
You are a skill-refinement expert. You will be given:
- the task instruction,
- the current SOP skills the agent used,
- the full tutorial the skills were extracted from (text + images),
- a `<root_cause>` XML analysis of the agent's most recent attempt,
- (if this is not attempt 2) the `<root_cause>` XML of every prior
  attempt in chronological order, so you can see how the skills have
  already been edited and avoid undoing fixes from earlier rounds.

{domain_reviser_guidance}

## Your job
Rewrite the skills so the next attempt avoids the issues flagged in
the most recent `<root_cause>` while preserving the sub-steps listed
in `<what_worked>`. Use the tutorial as the source of truth — if the
skill drifts from what the tutorial actually says, realign it.
Preserve skills that the analysis does not implicate.

## Editing style (applies regardless of outcome)
Refinement is **in-place editing of existing skills**. Do not grow
the skill set, and avoid growing any individual skill unless strictly
necessary — bloated skills lose the agent's attention and are the
main way previous fixes get forgotten.

### Handling `<issues>`
- Prefer to **rewrite** the step that produced the issue so the bad
  behaviour is no longer the natural reading. Do NOT address an issue
  by appending a cautionary sentence after the step; fix the step
  itself.
- You may add a new step ONLY when the issue is a genuinely missing
  action (e.g., the agent skipped a necessary wait, or skipped a
  verification that the tutorial shows). Do not add new steps for
  "just in case" hardening.
- Do not split one issue's fix across multiple skills, and do not
  create a new skill just to house a fix.

### Handling `<what_worked>`
- Treat each item as empirical evidence of a concrete method
  (shortcut, command flag, tool choice) that actually worked in this
  run.
- If a relevant skill is **silent** about that sub-step, promote the
  observed method into the most relevant existing skill so future
  attempts bias toward it.
- If a relevant skill already prescribes a **different** method, you
  may overwrite it with the observed method ONLY when the tutorial is
  itself silent or ambiguous about which to use. Tutorial wins on
  direct conflict — never replace a tutorial-backed method with the
  observed one.
- Do not create a new skill to carry `<what_worked>` content, and do
  not duplicate the same method across multiple skills.

## Edit intensity is gated by <outcome_assessment>
The most recent `<root_cause>` carries an `<outcome_assessment
value="...">` on its last line. Honor it:

- `value="likely_success"` — the analyzer judged the run effectively
  correct. Minimal mode:
  * if `<issues>` is empty you are in **reinforce mode**: absorb each
    `<what_worked>` item into the most relevant silent skill per the
    `<what_worked>` rule above, and make no other edits;
  * otherwise only tighten wording in skills explicitly pointed to by
    `<issues>`, and for `<what_worked>` absorb a method ONLY when the
    relevant skill is silent about that sub-step — do NOT overwrite a
    skill's existing method even if the tutorial is ambiguous.
- `value="uncertain"` — normal editing: apply the Editing style rules
  above for both `<issues>` and `<what_worked>` without further
  restriction.
- `value="likely_failure"` — larger in-place edits allowed: you may
  reshape steps and add a missing verification / wait when an issue
  calls for it. Actively absorb `<what_worked>` so the local wins do
  not regress. Still no new skills, still in-place.

## Using history
When prior attempts' `<root_cause>` XMLs are provided:
- If the current `<issues>` would undo a fix an earlier attempt
  introduced, prefer a different repair over reverting prior work.
- If a `<what_worked>` method was already absorbed into a skill in an
  earlier round (visible in the current skills), do not re-absorb it.

## Output format
Return the refined skills in the same markdown shape that the upstream
skill extractor produces:

# skill-name-in-kebab-case
> Brief description of what this skill does

## Steps
1. First step
2. Second step
   ![context image](filename.png)

## Expected Result
What the result should be.

---

# another-skill
> ...

Rules:
- Keep skill names in kebab-case; `> description` on its own line.
- Separate skills with `---` on a line by itself.
- You may reference tutorial images inline with `![alt](filename.png)`,
  where filename matches one of the image filenames shown in the user
  message. Images referenced in the original skills are also valid.
- Do NOT fabricate totally new skills unless the tutorial demonstrates
  a path that is genuinely missing. Prefer editing existing skills.
- Every skill must be present exactly once in the output (no dropping
  untouched skills — pass them through unchanged).
- No JSON, no outer code fences, no preamble before the first `#`.
"""


REVISER_REFINE_USER_TMPL = """
## Task Instruction
{instruction}

{history_block}## Trajectory Root Cause (raw XML from phase-1 analyzer, most recent attempt)
{root_cause_xml}
{trajectory_summary_block}
## Current Skills (same format you must emit)

{original_skills}

## Tutorial (source of truth)

{tutorial_body}
"""
