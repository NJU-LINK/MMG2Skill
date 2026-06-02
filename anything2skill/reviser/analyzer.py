"""ReviserAnalyzer: two-phase XML-driven trajectory analysis.

Phase 1 walks the trajectory in chunks of ``chunk_size`` predict-turns:

- Chunks 1..k-1 produce a rolling text summary (wrapped in ``<summary>``).
- The final chunk (or the only chunk, if N ≤ chunk_size) produces a
  ``<root_cause>`` XML blob parsed into :class:`RootCauseAnalysis`.

Each chunk is a single independent ``vlm.chat()`` call — there is no
multi-turn history between chunks; state transfers only through the
rolling summary text.

Design invariant: the analyzer does NOT consume the current skills or
the tutorial. Phase-1 is pure trajectory reading. Skill / tutorial
alignment is the refiner's concern (phase 2).
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

from anything2skill.reviser._text_utils import truncate as _truncate
from anything2skill.reviser.data_types import RootCauseAnalysis
from anything2skill.reviser.prompts import (
    REVISER_CHUNK_SYSTEM_TMPL,
    REVISER_FINAL_SYSTEM_TMPL,
)
from anything2skill.reviser.trajectory import (
    chunk_steps,
    group_by_predict_turn,
    iter_traj_steps,
    load_initial_obs,
    render_chunk_detailed,
)

if TYPE_CHECKING:
    from anything2skill.benchmark_kit import BenchmarkKit
    from anything2skill.vlm.client import VLMClient

logger = logging.getLogger("anything2skill.reviser")


class ReviserAnalyzer:
    """Phase-1 analyzer: rolling summary + final root-cause XML."""

    def __init__(
        self,
        vlm: VLMClient,
        kit: BenchmarkKit,
        *,
        chunk_size: int = 15,
        max_tokens: int = 2000,
        temperature: float | None = None,
        response_char_limit: int = 4000,
        rolling_summary_char_limit: int = 1500,
    ):
        self.vlm = vlm
        self.kit = kit
        self.chunk_size = max(1, int(chunk_size))
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.response_char_limit = response_char_limit
        self.rolling_summary_char_limit = rolling_summary_char_limit

    def analyze(
        self,
        traj_path: str,
        instruction: str,
        result_dir: str,
        audit_dir: str | None = None,
    ) -> RootCauseAnalysis:
        """Analyze a trajectory and return a :class:`RootCauseAnalysis`.

        Parameters
        ----------
        traj_path : str
            Path to the ``traj.jsonl`` to analyze.
        instruction : str
            Task instruction injected into the analyzer prompt.
        result_dir : str
            Directory containing the ``step_{N}_{ts}/`` observation
            dirs referenced by the trajectory. Used only for observation
            rehydration via ``kit.load_saved_observation``.
        audit_dir : str, optional
            Directory under which ``reviser_chunks/chunk_i.json`` audit
            dumps are written. **Must** point to a directory the caller
            considers writable for this analysis pass — e.g. the
            *next* attempt's dir when analyzing attempt_j, so canonical
            attempt_1 is never mutated when a non-canonical reviser
            model inspects it. Defaults to ``result_dir`` for backward
            compat with older call sites / tests.

        Signature intentionally does NOT take ``skills`` — phase-1 must
        not be influenced by skill text.
        """
        turns = group_by_predict_turn(iter_traj_steps(traj_path))
        if not turns:
            logger.warning("[Reviser] Empty trajectory at %s", traj_path)
            return RootCauseAnalysis(trajectory_summary="No trajectory data")

        chunks = chunk_steps(turns, self.chunk_size)
        total_turns = len(turns)
        k = len(chunks)
        rolling_summary = ""

        chunks_dir = Path(audit_dir if audit_dir is not None else result_dir) / "reviser_chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        for i, chunk in enumerate(chunks, start=1):
            is_last = i == k
            # Ground the analyzer on the pre-turn state only on the first
            # chunk — later chunks either inherit it via rolling summary or
            # don't need it.
            messages = self._build_messages(
                instruction=instruction,
                rolling_summary=rolling_summary,
                chunk=chunk,
                result_dir=result_dir,
                chunk_index=i,
                total_chunks=k,
                total_turns=total_turns,
                is_last=is_last,
                include_initial_obs=(i == 1),
            )

            try:
                response = self.vlm.chat(
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            except Exception as e:
                logger.error(
                    "[Reviser] Analyzer chunk %d/%d VLM call failed: %s",
                    i, k, e,
                )
                if is_last:
                    return RootCauseAnalysis(
                        trajectory_summary=f"Analyzer VLM error: {e}",
                    )
                # For a mid-chunk failure we still try the next chunk —
                # keep the rolling summary as-is.
                self._dump_chunk_audit(
                    chunks_dir, i, messages, response=None, error=str(e),
                )
                continue

            self._dump_chunk_audit(chunks_dir, i, messages, response=response)

            if is_last:
                return _parse_root_cause(response)
            rolling_summary = _truncate(
                _parse_summary(response),
                self.rolling_summary_char_limit,
            )

        # Shouldn't reach here (empty chunks handled above), but keep a
        # safe default.
        return RootCauseAnalysis(trajectory_summary="No chunks produced")

    # ------------------------------------------------------------------

    def _build_messages(
        self,
        *,
        instruction: str,
        rolling_summary: str,
        chunk: list[dict],
        result_dir: str,
        chunk_index: int,
        total_chunks: int,
        total_turns: int,
        is_last: bool,
        include_initial_obs: bool,
    ) -> list[dict]:
        domain_guidance = (self.kit.reviser_guidance or "").strip()

        if is_last:
            system = REVISER_FINAL_SYSTEM_TMPL.format(
                domain_reviser_guidance=domain_guidance,
                rolling_summary_char_limit=self.rolling_summary_char_limit,
            ).strip()
        else:
            system = REVISER_CHUNK_SYSTEM_TMPL.format(
                domain_reviser_guidance=domain_guidance,
                rolling_summary_char_limit=self.rolling_summary_char_limit,
            ).strip()

        # Window bounds (turn indices)
        first_turn_idx = chunk[0]["turn_index"]
        last_turn_idx = chunk[-1]["turn_index"]
        if total_chunks == 1:
            window_label = (
                f"## Execution Trajectory (Turns {first_turn_idx}–{last_turn_idx} of {total_turns}, full detail)"
            )
        elif is_last:
            window_label = (
                f"## Final chunk: Turns {first_turn_idx}–{last_turn_idx} of {total_turns} (full detail)"
            )
        else:
            window_label = (
                f"## Current chunk: Turns {first_turn_idx}–{last_turn_idx} of {total_turns} (full detail)"
            )

        header_parts = [f"## Task Instruction\n{instruction}"]
        if rolling_summary:
            header_parts.append(
                f"## Rolling summary of earlier turns\n{rolling_summary}"
            )
        header_parts.append(window_label)

        user_content: list[dict] = [
            {"type": "text", "text": "\n\n".join(header_parts)},
        ]

        # Prepend the initial observation for grounding when the caller
        # requests it. The label is explicit ("not a turn") so the analyzer
        # does not index it as "Turn 0" and misattribute root cause.
        if include_initial_obs:
            initial_blocks = load_initial_obs(result_dir, self.kit)
            if initial_blocks:
                user_content.append(
                    {
                        "type": "text",
                        "text": (
                            "## Initial observation (before Turn 1)\n"
                            "This is the state the agent saw when making "
                            "its first predict. It is NOT a turn — do not "
                            "reference it as Turn 0 in <where>; use it "
                            "only to ground the agent's opening decision."
                        ),
                    }
                )
                user_content.extend(initial_blocks)

        user_content.extend(
            render_chunk_detailed(
                chunk,
                self.kit,
                result_dir,
                response_char_limit=self.response_char_limit,
            )
        )
        if is_last:
            footer = "Now produce the <root_cause> XML."
        else:
            footer = (
                "Update the rolling summary. Output only a single "
                "<summary>...</summary> block."
            )
        user_content.append({"type": "text", "text": footer})

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _dump_chunk_audit(
        chunks_dir: Path,
        chunk_index: int,
        messages: list[dict],
        *,
        response: str | None = None,
        error: str | None = None,
    ) -> None:
        """Persist the full round's messages + assistant reply for debug."""
        safe_messages = _strip_images_for_audit(messages)
        payload = {
            "chunk_index": chunk_index,
            "messages": safe_messages,
            "response": response,
            "error": error,
        }
        try:
            (chunks_dir / f"chunk_{chunk_index}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(
                "[Reviser] Failed to persist chunk %d audit: %s",
                chunk_index, e,
            )


# ── Helpers ──────────────────────────────────────────────────────────────


def _strip_images_for_audit(messages: list[dict]) -> list[dict]:
    """Replace ``image_url`` blocks with a placeholder for JSON audit dumps."""
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            safe_blocks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    safe_blocks.append({"type": "image_url", "image_url": "<redacted>"})
                else:
                    safe_blocks.append(block)
            out.append({**msg, "content": safe_blocks})
        else:
            out.append(msg)
    return out


def _parse_summary(response: str) -> str:
    """Pull the text out of ``<summary>...</summary>``.

    If the tag is missing, falls back to the stripped full response.
    """
    match = re.search(r"<summary>\s*(.*?)\s*</summary>", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()


def _parse_root_cause(response: str) -> RootCauseAnalysis:
    """Parse an assistant response into :class:`RootCauseAnalysis`.

    Three-level tolerance:
    1. regex-isolate the outer ``<root_cause>`` block,
    2. ``xml.etree`` parse that block,
    3. if ET fails, per-field regex fallback (scoped to the block).

    On any path where the resulting analysis carries no fields, we log a
    warning and preserve the raw response in ``raw_response`` so an operator
    can inspect the model's actual text.
    """
    # Greedy match so any legacy `<root_cause>` inner tag (pre-rename) doesn't
    # truncate the outer block. The prompt only ever emits one outer block so
    # this is safe.
    match = re.search(
        r"<root_cause>.*</root_cause>", response, re.DOTALL | re.IGNORECASE,
    )
    if not match:
        logger.warning(
            "[Reviser] No <root_cause> tag in response (first 200 chars: %r)",
            response[:200],
        )
        return RootCauseAnalysis(raw_response=response)

    raw_xml = match.group(0)

    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as e:
        logger.warning("[Reviser] ET parse failed (%s), falling back to regex", e)
        return _regex_root_cause_fallback(raw_xml, response)

    trajectory_summary = (root.findtext("trajectory_summary") or "").strip()

    what_worked: list[str] = []
    ww_el = root.find("what_worked")
    if ww_el is not None:
        for item_el in ww_el.findall("item"):
            text = (item_el.text or "").strip()
            if text:
                what_worked.append(text)

    issues: list[dict] = []
    issues_el = root.find("issues")
    if issues_el is not None:
        for issue_el in issues_el.findall("issue"):
            issues.append(
                {
                    "where": (issue_el.findtext("where") or "").strip(),
                    "evidence": (issue_el.findtext("evidence") or "").strip(),
                    "cause": (issue_el.findtext("cause") or "").strip(),
                }
            )

    outcome_el = root.find("outcome_assessment")
    outcome_assessment = ""
    outcome_rationale = ""
    if outcome_el is not None:
        outcome_assessment = (outcome_el.attrib.get("value") or "").strip()
        outcome_rationale = (outcome_el.text or "").strip()

    return RootCauseAnalysis(
        trajectory_summary=trajectory_summary,
        what_worked=what_worked,
        issues=issues,
        outcome_assessment=outcome_assessment,
        outcome_rationale=outcome_rationale,
        raw_xml=raw_xml,
    )


_TRAJ_SUMMARY_RE = re.compile(
    r"<trajectory_summary>(.*?)</trajectory_summary>", re.DOTALL | re.IGNORECASE,
)
_WHAT_WORKED_BLOCK_RE = re.compile(
    r"<what_worked>(.*?)</what_worked>", re.DOTALL | re.IGNORECASE,
)
_WHAT_WORKED_ITEM_RE = re.compile(
    r"<item>(.*?)</item>", re.DOTALL | re.IGNORECASE,
)
_ISSUES_BLOCK_RE = re.compile(
    r"<issues>(.*?)</issues>", re.DOTALL | re.IGNORECASE,
)
_ISSUE_RE = re.compile(r"<issue>(.*?)</issue>", re.DOTALL | re.IGNORECASE)
_WHERE_RE = re.compile(r"<where>(.*?)</where>", re.DOTALL | re.IGNORECASE)
_EVIDENCE_RE = re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL | re.IGNORECASE)
_CAUSE_RE = re.compile(r"<cause>(.*?)</cause>", re.DOTALL | re.IGNORECASE)
_OUTCOME_RE = re.compile(
    r"<outcome_assessment\s+value=\"([^\"]*)\"\s*>(.*?)</outcome_assessment>",
    re.DOTALL | re.IGNORECASE,
)


def _regex_root_cause_fallback(raw_xml: str, raw_response: str) -> RootCauseAnalysis:
    """Per-field regex fallback when ET.fromstring can't parse the block.

    The fallback's job is to salvage *malformed* XML — common failure modes
    are missing closing tags around the wrapper containers. We try the
    scoped <issues>...</issues> haystack first (so a stray <issue> inside an
    <evidence> body doesn't get hoisted), and fall back to a global search
    if scoping yields nothing. Same for <what_worked>.
    """
    traj_match = _TRAJ_SUMMARY_RE.search(raw_xml)
    trajectory_summary = traj_match.group(1).strip() if traj_match else ""

    what_worked = _extract_what_worked(raw_xml)
    issues = _extract_issues(raw_xml)

    outcome_match = _OUTCOME_RE.search(raw_xml)
    outcome_assessment = outcome_match.group(1).strip() if outcome_match else ""
    outcome_rationale = outcome_match.group(2).strip() if outcome_match else ""

    if (
        not trajectory_summary
        and not what_worked
        and not issues
        and not outcome_assessment
    ):
        logger.warning(
            "[Reviser] Regex fallback extracted no fields from <root_cause> "
            "block (first 200 chars: %r)",
            raw_xml[:200],
        )
        return RootCauseAnalysis(raw_xml=raw_xml, raw_response=raw_response)

    return RootCauseAnalysis(
        trajectory_summary=trajectory_summary,
        what_worked=what_worked,
        issues=issues,
        outcome_assessment=outcome_assessment,
        outcome_rationale=outcome_rationale,
        raw_xml=raw_xml,
    )


def _extract_issues(raw_xml: str) -> list[dict]:
    """Pull <issue> entries, preferring the scoped container, then global."""
    issues_block_match = _ISSUES_BLOCK_RE.search(raw_xml)
    if issues_block_match is not None:
        scoped = list(_ISSUE_RE.finditer(issues_block_match.group(1)))
        if scoped:
            return [_parse_issue_body(m.group(1)) for m in scoped]
    # Wrapper missing or scoped pass empty — salvage from full raw_xml.
    return [_parse_issue_body(m.group(1)) for m in _ISSUE_RE.finditer(raw_xml)]


def _parse_issue_body(body: str) -> dict:
    where = _WHERE_RE.search(body)
    evidence = _EVIDENCE_RE.search(body)
    cause = _CAUSE_RE.search(body)
    return {
        "where": where.group(1).strip() if where else "",
        "evidence": evidence.group(1).strip() if evidence else "",
        "cause": cause.group(1).strip() if cause else "",
    }


def _extract_what_worked(raw_xml: str) -> list[str]:
    """Pull <item> entries from <what_worked>, preferring scoped then global."""
    block_match = _WHAT_WORKED_BLOCK_RE.search(raw_xml)
    if block_match is not None:
        scoped = [
            m.group(1).strip()
            for m in _WHAT_WORKED_ITEM_RE.finditer(block_match.group(1))
        ]
        scoped = [s for s in scoped if s]
        if scoped:
            return scoped
    # No <what_worked> wrapper found — no global fallback, because <item>
    # could legitimately appear in other contexts later.
    return []
