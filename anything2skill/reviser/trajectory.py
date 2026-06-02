"""Trajectory parsing, predict-turn grouping, and chunk rendering for the reviser."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from anything2skill.reviser._text_utils import truncate as _truncate

if TYPE_CHECKING:
    from anything2skill.benchmark_kit import BenchmarkKit

logger = logging.getLogger("anything2skill.reviser")


def iter_traj_steps(traj_path: str | Path) -> Iterator[dict]:
    """Yield each action row from ``traj.jsonl``.

    Rows that fail to parse are skipped with a warning rather than
    terminating the iteration.
    """
    path = Path(traj_path)
    if not path.is_file() or path.stat().st_size == 0:
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            yield json.loads(raw_line)
        except json.JSONDecodeError as e:
            logger.warning("Skipping malformed traj.jsonl line: %s", e)


def group_by_predict_turn(steps: Iterator[dict] | list[dict]) -> list[dict]:
    """Collapse action-level rows into predict-turn groups.

    A single ``agent.predict()`` call can emit multiple actions; they share
    the same ``response`` / ``predict_num`` and write sequential
    ``step_num`` rows. This groups them back into one "turn" for the
    analyzer, preserving the list of actions.

    Fallback: if ``predict_num`` is missing (corrupt / legacy data), falls
    back to grouping by ``step_num`` so every row becomes its own turn.

    The analyzer must not see env-side oracle signals that never reach
    the agent. This grouping drops:
      - ``info.controller_result.error`` (shell stderr)
      - ``info.controller_result.returncode``
      - ``done`` (env-level termination flag — the agent's own
        terminal action, however the domain spells it, is still
        visible inside ``actions``)

    Planner fields (both ``reasoning`` and ``guidance``) are the
    planner's own authored output within the agent ensemble — same
    category as ``response``; they are kept.

    Returns a list of dicts, each with keys:
        turn_index     (1-based, assigned here)
        predict_num    (original, or None)
        phase          (first non-empty phase seen in the group)
        response       (shared response text the agent itself emitted)
        step_nums      [int, ...]
        action_timestamps [str, ...] — one per action
        actions        [str, ...]
        planner_reasoning (first non-empty planner.reasoning seen, or "")
        planner_guidance (first non-empty planner.guidance seen, or "")
        last_step_num  (final step_num in the group; used for observation lookup)
        last_action_timestamp  (final action_timestamp; used for observation lookup)
    """
    rows = list(steps)
    if not rows:
        return []

    groups: list[list[dict]] = []
    current: list[dict] = []
    current_key: object = None

    for row in rows:
        pn = row.get("predict_num")
        key = ("p", pn) if pn is not None else ("s", row.get("step_num"))
        if current and key != current_key:
            groups.append(current)
            current = []
        current.append(row)
        current_key = key
    if current:
        groups.append(current)

    turns: list[dict] = []
    for i, group in enumerate(groups, start=1):
        first = group[0]
        last = group[-1]

        actions = [row.get("action", "") for row in group]
        step_nums = [row.get("step_num") for row in group]
        action_timestamps = [row.get("action_timestamp") for row in group]

        phase = ""
        for row in group:
            ph = row.get("phase", "")
            if ph:
                phase = ph
                break

        planner_reasoning = ""
        planner_guidance = ""
        for row in group:
            p = row.get("planner") or {}
            if not isinstance(p, dict):
                continue
            if not planner_reasoning:
                r = str(p.get("reasoning") or "").strip()
                if r:
                    planner_reasoning = r
            if not planner_guidance:
                g = str(p.get("guidance") or "").strip()
                if g:
                    planner_guidance = g
            if planner_reasoning and planner_guidance:
                break

        turns.append(
            {
                "turn_index": i,
                "predict_num": first.get("predict_num"),
                "phase": phase,
                "response": first.get("response", ""),
                "step_nums": step_nums,
                "action_timestamps": action_timestamps,
                "actions": actions,
                "planner_reasoning": planner_reasoning,
                "planner_guidance": planner_guidance,
                "last_step_num": last.get("step_num"),
                "last_action_timestamp": last.get("action_timestamp"),
            }
        )

    return turns


def chunk_steps(turns: list[dict], size: int) -> list[list[dict]]:
    """Split a list of predict-turns into fixed-size chunks.

    The final chunk may be smaller than ``size``. An empty input or
    non-positive size yields an empty list.
    """
    if size <= 0 or not turns:
        return []
    return [turns[i : i + size] for i in range(0, len(turns), size)]


def _step_dir_for_turn(result_dir: str | Path, turn: dict) -> Path | None:
    """Locate the ``step_{N}_{ts}`` directory for the last action in *turn*."""
    step_num = turn.get("last_step_num")
    ts = turn.get("last_action_timestamp")
    if step_num is None or not ts:
        return None
    return Path(result_dir) / f"step_{step_num}_{ts}"


def render_step_detailed(
    turn: dict,
    kit: BenchmarkKit,
    result_dir: str | Path,
    response_char_limit: int = 4000,
) -> list[dict]:
    """Render a single predict-turn as VLM content blocks.

    The rendering surface mirrors what the agent itself saw / wrote:
        - response (agent's own output), planner reasoning / guidance
          (also agent-authored), action(s), post-action observation
          (what the next turn would perceive — may be an image, text,
          or any combination the kit chooses).

    Env-side oracle fields (controller errors, return codes, env `done`
    flag) are intentionally omitted — the reviser must reason from the
    same information surface the next agent will have.

    Produces:
        1. one text block with the turn index, step range, phase label
           (if any), planner reasoning + guidance (if any), response,
           action(s);
        2. the saved observation for the turn's final step (via
           ``kit.load_saved_observation``) — whatever VLM content
           blocks the kit returns. Silently degrades to a single
           ``(observation unavailable)`` text block if the directory
           is missing or the kit returns no blocks.
    """
    turn_idx = turn.get("turn_index")
    step_nums = turn.get("step_nums") or []
    if step_nums:
        if len(step_nums) == 1:
            step_range = f"step_num={step_nums[0]}"
        else:
            step_range = f"step_nums={step_nums[0]}..{step_nums[-1]}"
    else:
        step_range = "step_nums=?"

    response_txt = _truncate(turn.get("response", ""), response_char_limit)

    actions = turn.get("actions") or []
    if len(actions) == 1:
        action_line = f"**Action:** {actions[0]}"
    else:
        joined = " / ".join(
            f"{i + 1}. {a}" for i, a in enumerate(actions)
        )
        action_line = f"**Actions ({len(actions)}):** {joined}"

    header_lines = [f"### Turn {turn_idx} ({step_range})"]
    phase = turn.get("phase") or ""
    if phase:
        header_lines.append(f"**Phase:** {phase}")
    reasoning = (turn.get("planner_reasoning") or "").strip()
    if reasoning:
        header_lines.append(
            f"**Planner reasoning:** {_truncate(reasoning, 400)}"
        )
    guidance = (turn.get("planner_guidance") or "").strip()
    if guidance:
        header_lines.append(
            f"**Planner guidance:** {_truncate(guidance, 400)}"
        )
    header_lines.append(f"**Response:** {response_txt}")
    header_lines.append(action_line)

    text_block: dict = {
        "type": "text",
        "text": "\n".join(header_lines),
    }

    blocks: list[dict] = [text_block]
    step_dir = _step_dir_for_turn(result_dir, turn)
    image_blocks: list[dict] = []
    if step_dir is not None and step_dir.is_dir():
        try:
            image_blocks = kit.load_saved_observation(step_dir)
        except Exception as e:
            logger.warning(
                "load_saved_observation failed for %s: %s", step_dir, e
            )
            image_blocks = []
    if image_blocks:
        blocks.extend(image_blocks)
    else:
        blocks.append(
            {"type": "text", "text": "(observation unavailable)"}
        )
    return blocks


def load_initial_obs(
    result_dir: str | Path, kit: BenchmarkKit
) -> list[dict]:
    """Load the initial observation (``step_0_initial/``) as VLM blocks.

    The runner writes ``step_0_initial/`` immediately after ``env.reset``
    and before the agent loop starts, so this is the exact state the
    agent perceived when it made its first predict. Used by the analyzer
    as a chunk-1-only grounding header — NOT as a turn.

    Returns ``[]`` if the directory is missing (older runs or benchmarks
    that do not save an initial observation) — the analyzer degrades
    to rendering the trajectory without obs_0.
    """
    step0 = Path(result_dir) / "step_0_initial"
    if not step0.is_dir():
        return []
    try:
        return kit.load_saved_observation(step0)
    except Exception as e:
        logger.warning("load_initial_obs failed for %s: %s", step0, e)
        return []


def render_chunk_detailed(
    chunk: list[dict],
    kit: BenchmarkKit,
    result_dir: str | Path,
    response_char_limit: int = 4000,
) -> list[dict]:
    """Concatenate the content blocks of every turn in *chunk*."""
    out: list[dict] = []
    for turn in chunk:
        out.extend(
            render_step_detailed(turn, kit, result_dir, response_char_limit)
        )
    return out
