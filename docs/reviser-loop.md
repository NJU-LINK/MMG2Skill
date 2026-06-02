# Reviser Revision Loop

This document explains how the `reviser/` module works: it runs multiple attempts on the same task, revises skills with a VLM based on the previous trajectory + tutorial, and produces dual-view data that is convenient for ablations.

> Prerequisite: read [`execution-flow.md`](execution-flow.md) first to understand the step loop inside a single attempt.

---

## 1. What It Is

The "Reviser" is a two-stage pipeline inserted between multiple attempts:

1. attempt_j finishes, and `traj.jsonl` is written to disk.
2. **Phase 1 — `ReviserAnalyzer`**: slices the trajectory by `chunk_size`, outputs a rolling `<summary>` for each segment, outputs `<root_cause>` in the final segment, and writes the combined result to `attempt_{j+1}/root_cause.xml`.
3. **Phase 2 — `ReviserRefiner`**: produces new skills from four inputs and writes them to `attempt_{j+1}/skills/`:
   - **Current root_cause** (just obtained by analyzing attempt_j's trajectory)
   - **Old skills** (the version used by attempt_j)
   - **Original tutorial text** + tutorial images (limited by `tutorial_image_cap`)
   - **Historical root_cause chain**: `root_cause.xml` from attempt_2..attempt_j, preventing the refiner from reverting earlier fixes. See §3.
4. attempt_{j+1} reruns with the new skills, and the loop continues until `max_attempts`.

The design has two goals:
- **Revise skills**: prevent later attempts from repeating the same mistakes.
- **Produce ablation data**: the early-stop marker lets a complete `max_attempts` run answer two questions at once — "what score would I get with the early-stop strategy" and "would the score recover or fluctuate if I kept running more rounds".

---

## 2. When It Is Enabled

Controlled by `configs/config.yaml: reviser.max_attempts`:

| `max_attempts` | Behavior |
|---|---|
| `1` (default) | Reviser is disabled; only `attempt_1` runs and is written to the canonical bucket. |
| `≥ 2` | Reviser is enabled; `attempt_1` is written to the canonical bucket, and `attempt_2..N` are written to the reviser bucket. |

When `agent_mode in {vanilla, vanilla_tutorial}`, `_effective_max_attempts()` (`runner.py`, constant `_NO_REVISER_MODES`) forces the value back to 1 because neither mode has skills to modify.

---

## 3. Refine-Stage Inputs: Current root_cause + Historical root_cause Chain

Each refine step feeds both the "current-round root_cause" and "all previous rounds' root_cause" to the VLM. This section explains why it is needed, what it contains, and how to turn it off.

### Why History Is Needed

If the refiner only sees the current-round root_cause, it can easily undo a fix that was just added in the previous round. For example, attempt_2 may fix attempt_1's failure by adding a step such as "click Activities before searching," but attempt_2 may then fail for a different reason. If the refiner only reads attempt_2's root_cause, it may treat the Activities step as redundant and delete it, causing attempt_3 to regress to attempt_1's error. Feeding historical root_causes is equivalent to telling the refiner, "these fixes have already been validated; do not touch them."

### Exact Contents of the History

`_collect_history(target_attempt=N)` (`reviser_runner.py:506`) reads, in order, `attempt_2/root_cause.xml`, `attempt_3/root_cause.xml`, ..., `attempt_{N-1}/root_cause.xml`.

- `attempt_1` has no `root_cause.xml` because there is no preceding attempt, so history starts from `attempt_2`.
- The current round's own root_cause is not in `history`; it is passed as a separate parameter.

### Concrete Example

With `max_attempts=4`, each refine step sees:

| Produced attempt | Current root_cause (source) | history (= list) |
|---|---|---|
| `attempt_2` | Analysis of `attempt_1/traj.jsonl` | `[]` |
| `attempt_3` | Analysis of `attempt_2/traj.jsonl` | `[attempt_2/root_cause.xml]` |
| `attempt_4` | Analysis of `attempt_3/traj.jsonl` | `[attempt_2/root_cause.xml, attempt_3/root_cause.xml]` |

Note that `attempt_2/root_cause.xml` is actually the root_cause obtained by analyzing the attempt_1 trajectory. It lives under the attempt_2 directory because that root cause **is exactly what is used to produce the attempt_2 skills**. Therefore, the history chain reflects "what issue each previous refine round saw and what revision it made."

### Presentation in the Prompt

`refiner.py:_format_history_block()` renders the history as XML blocks such as `<previous_root_cause attempt="2">...</previous_root_cause>` and inserts them into the user prompt alongside the current round's `<root_cause>`. The VLM can compare issues horizontally: which issues are recurring, and which have already been fixed.

### Turning History Off (Ablation)

Set `reviser.history_in_refine=false` to make `_collect_history()` directly `return []` (`reviser_runner.py:520`). This switch is for ablation experiments and is enabled by default.

> Note: the analyzer's rolling `<summary>` is between chunks and **does not belong** to the "historical root_cause" described here; it is an intermediate artifact within the same attempt.

---

## 4. Dual-Bucket Result Layout

`runner.py:_compute_run_names()` determines the two bucket names:

```
results/{tutorial_type}/a2s-{mode}-{agent_model}/{benchmark}/{domain}/{task_id}/
  task.json
  attempt_1/                         ← always in the canonical bucket
    traj.jsonl, result.txt, runtime.log, recording.mp4, step_N_*/
    skills/, meta.json

results/{tutorial_type}/a2s-{mode}-{agent_model}-r_{reviser_model}/{benchmark}/{domain}/{task_id}/
  task.json
  attempt_2/                         ← attempt_2+ are all in the reviser bucket
    root_cause.xml                   ← analyzer output, determines which skills this attempt uses
    skills/                          ← refiner output
    traj.jsonl, result.txt, runtime.log, recording.mp4, step_N_*/
    meta.json
  attempt_3/  ...
```

Design notes:

- **The canonical bucket is always the bare-agent baseline** and is reused across reviser experiments. Once written, it is read-only; during resume, `reviser_runner.py` skips canonical cleanup logic (`reviser_runner.py:236`).
- **The reviser bucket is separated by reviser model**. Even when `reviser.model == agent.model`, as long as `max_attempts > 1`, an independent `-r_{reviser_model}` directory is still created and does not contaminate the baseline.
- When `max_attempts == 1`, both bucket names are the same (`_compute_run_names()` directly returns the same canonical name).

---

## 5. `meta.json` Fields

Each `attempt_N/meta.json`:

```json
{
  "score": 0.0,
  "steps_taken": 12,
  "skills_count": 4,
  "early_stop_triggered": false,
  "completed": true
}
```

`early_stop_triggered` is the most important field in this loop. It marks the first attempt under the current configuration that satisfies the early-stop signal. The default is `reviser.early_stop_signal=likely_success`, i.e., the analyzer's `outcome_assessment == "likely_success"`.

**Key point**: the loop does not actually break because of this marker. The code only writes the marker to the current attempt's `meta.json`, then force-continues to `max_attempts`. This lets one run produce two views:

- **Early-stop view**: use the score of the attempt with `early_stop_triggered=True` (or the final attempt's score if none exists).
- **Full-run view**: the score sequence of all attempts, used to inspect fluctuation, recovery, or degradation.

---

## 6. early_stop Trigger Conditions

`ReviserAnalyzer` is called after attempt_j and outputs `RootCauseAnalysis(issues=[...], outcome_assessment=...)`:

- Default `likely_success` strategy: triggers when `outcome_assessment == "likely_success"`, matching the analyzer-selected early-stop view used in evaluation.
- Compatible `no_issue` strategy: triggers when `issues == []`; explicitly switch with `reviser.early_stop_signal=no_issue`.
- On trigger, writes `early_stop_triggered=True` to attempt_j's `meta.json` (**not** attempt_{j+1}).
- The `already_marked` flag ensures the loop only marks the first trigger and avoids repeated rewrites.

Crash recovery path: if an attempt has already saved meta (default `early_stop_triggered=False`) but the worker crashes before the flip, the next startup calls `_backfill_early_stop_if_first()` (`reviser_runner.py:469`) while parsing `attempt_j/root_cause.xml` inside `_recover_skills_for_attempt()`, backfilling the marker.

---

## 7. Configuration Items

Defined under top-level `reviser.*` in `configs/config.yaml` and can be overridden by `configs/benchmark/<name>.yaml`:

| Field | Default | Description |
|---|---|---|
| `max_attempts` | `1` | Total number of attempts. `1` = disable reviser. |
| `model` / `base_url` / `api_key` | `null` | VLM used by the reviser. `null` = reuse the agent's client. |
| `analysis_temperature` | `null` | Analyzer temperature. `null` follows the reasoning endpoint compatibility configuration. |
| `refine_temperature` | `null` | Refiner temperature, same as above. |
| `analysis_max_tokens` | `32768` | Maximum response tokens for a single analyzer call. |
| `refine_max_tokens` | `32768` | Maximum response tokens for a single refiner call. |
| `response_char_limit` | `32768` | Truncation length when rendering agent responses in the trajectory. |
| `rolling_summary_char_limit` | `32768` | Character limit for rolling `<summary>`; longer summaries are compressed. |
| `tutorial_image_cap` | `20` | Maximum number of tutorial images injected into the refiner. |
| `include_tutorial_in_refine` | `true` | If disabled, the refiner only sees root_cause + old skills. |
| `early_stop_signal` | `likely_success` | Early-stop marker signal. Set to `no_issue` to use the old semantics. |
| `history_in_refine` | `true` | Also feed historical `root_cause.xml` files to the refiner to avoid overwriting earlier fixes. See §3. |
| `chunk_size` | Per benchmark YAML | OSWorld / Minecraft = 15, RLCard = 30. The analyzer's trajectory chunk size. |

---

## 8. Resume and Crash Recovery

`scan_attempts(canonical_dir, reviser_dir)` scans both buckets whenever `run_with_reviser()` starts and finds `last_complete`. The next round starts from `last_complete + 1`:

- `attempt_j/skills/` exists → use it directly to run (do not repeat refine).
- `attempt_j/root_cause.xml` exists but `skills/` is missing → skip analyze and only run refine.
- Neither exists → read `attempt_{j-1}/traj.jsonl` and run the full analyze + refine pipeline.

Cleanup policy (`reviser_runner.py:236`): execution artifacts (`traj.jsonl`, `step_*/`, `result.txt`, `response.log`, `reviser_chunks/`) are removed; persistent artifacts (`skills/`, `root_cause.xml`, `meta.json`) are retained. canonical attempt_1 is never cleaned.

---

## 9. Analysis Tools

- **`anything2skill/metrics/tracker.py`** — `ExperimentTracker.add_attempt()` aggregates `attempt_N/meta.json` at task level.
- **`results/{tutorial_type}/{run_name}/{benchmark}/experiment_results.json`** — rewritten by `_rebuild_experiment_results()` after each attempt finishes. The reviser bucket's `experiment_results.json` **actively merges in** the canonical `attempt_1`, by design: reviser experiments need to see the baseline.
- **`notebooks/analyze_one_model.ipynb`** — per-model, per-attempt analysis. Outputs success_rate, avg_score, and avg_steps for both views (early-stop / full-run), and exports to `RUN_DIR/analysis_one_model.json`.

---

## 10. Key File Index

| File | Responsibility |
|---|---|
| `anything2skill/reviser/reviser_runner.py` | `ReviserRunner.run_with_reviser()` — main loop, resume, `early_stop` marking, experiment_results refresh; `_collect_history()` loads historical root_causes |
| `anything2skill/reviser/refiner.py:_format_history_block` | Renders `list[RootCauseAnalysis]` into `<previous_root_cause>` XML blocks in the prompt |
| `anything2skill/reviser/analyzer.py` | `ReviserAnalyzer.analyze()` — Phase 1, trajectory → `<root_cause>` XML |
| `anything2skill/reviser/refiner.py` | `ReviserRefiner.refine()` — Phase 2, XML + skills + tutorial → new Skills |
| `anything2skill/reviser/data_types.py` | `RootCauseAnalysis` data class |
| `anything2skill/reviser/trajectory.py` | Trajectory chunking / rendering helpers |
| `anything2skill/reviser/skills_io.py` | Persisting / reading `Skills` (`SKILL.md` format) |
| `anything2skill/reviser/prompts.py` | Analyzer / refiner prompt templates (parallel to `agent/prompts.py`) |
| `anything2skill/runner.py:_compute_run_names` | Computes canonical / reviser bucket names |
| `anything2skill/runner.py:_effective_max_attempts` | Forces `max_attempts` to 1 for `vanilla` / `vanilla_tutorial` modes (see `_NO_REVISER_MODES`) |
| `anything2skill/runner.py:_make_reviser_vlm_factory` | Lazy construction of reviser `VLMClient` (reuses the agent VLM when the model is the same) |
| `anything2skill/metrics/tracker.py` | `ExperimentTracker` / `AttemptResult` / `TaskResult` |
