"""Tests for the reviser module + runner dual-bucket orchestration."""

from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from anything2skill.parser.data_types import Skill, Skills, TutorialMaterial
from anything2skill.reviser.analyzer import (
    ReviserAnalyzer,
    _parse_root_cause,
    _parse_summary,
    _regex_root_cause_fallback,
)
from anything2skill.reviser.data_types import RootCauseAnalysis
from anything2skill.reviser.refiner import ReviserRefiner
from anything2skill.reviser.reviser_runner import (
    ReviserRunner,
    _cleanup_runtime_artifacts,
    _latest_attempt_scores,
    _matches_early_stop_signal,
    _normalize_early_stop_signal,
    _save_attempt_meta,
    _save_root_cause,
    scan_attempts,
)
from anything2skill.reviser.skills_io import (
    load_skills_from_dir,
    save_skills_to_dir,
)
from anything2skill.reviser.trajectory import (
    chunk_steps,
    group_by_predict_turn,
    iter_traj_steps,
    load_initial_obs,
    render_step_detailed,
)


# ----------------------------------------------------------------------
# Tiny kit stub for tests — no benchmark specifics.
# ----------------------------------------------------------------------


@dataclass
class _FakeTask:
    task_id: str = "t1"
    instruction: str = "Do the thing"


class _StubKit:
    """Minimum surface the reviser touches."""

    def __init__(self, guidance: str = "STUB_GUIDANCE"):
        self._guidance = guidance

    @property
    def reviser_guidance(self) -> str:
        return self._guidance

    # The analyzer calls this per turn; default returns nothing.
    def load_saved_observation(self, step_dir: Path) -> list[dict]:
        screenshot = step_dir / "screenshot.png"
        if not screenshot.is_file():
            return []
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,FAKE",
                    "detail": "high",
                },
            }
        ]

    def get_result_subdir(self, task) -> str:
        return task.task_id


def _make_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (20, 20), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    path.write_bytes(buf.getvalue())


def _write_traj(
    traj_path: Path, rows: list[dict], step_dirs: dict[int, Path] | None = None
) -> None:
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(traj_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")
    if step_dirs:
        for step_num, step_dir in step_dirs.items():
            step_dir.mkdir(parents=True, exist_ok=True)
            _make_png(step_dir / "screenshot.png")


# ----------------------------------------------------------------------
# Trajectory iter / group / chunk / render
# ----------------------------------------------------------------------


class TestTrajectory:
    def test_iter_traj_steps_skips_malformed(self, tmp_path, caplog):
        traj = tmp_path / "traj.jsonl"
        traj.write_text(
            "\n"
            + json.dumps({"step_num": 1, "action": "a"}) + "\n"
            + "not-json\n"
            + json.dumps({"step_num": 2, "action": "b"}) + "\n",
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING, logger="anything2skill.reviser"):
            rows = list(iter_traj_steps(traj))
        assert len(rows) == 2
        assert rows[0]["step_num"] == 1
        assert rows[1]["step_num"] == 2
        assert any("malformed" in r.message for r in caplog.records)

    def test_iter_traj_steps_missing_file(self, tmp_path):
        assert list(iter_traj_steps(tmp_path / "nothing.jsonl")) == []

    def test_group_by_predict_turn_merges_multi_action_shared_predict(self):
        rows = [
            {
                "step_num": 1, "predict_num": 1, "phase": "act",
                "action": "A", "response": "resp1",
                "action_timestamp": "T1", "done": False, "info": {},
            },
            {
                "step_num": 2, "predict_num": 1, "phase": "act",
                "action": "B", "response": "resp1",
                "action_timestamp": "T2", "done": False, "info": {},
            },
            {
                "step_num": 3, "predict_num": 2, "phase": "act",
                "action": "C", "response": "resp2",
                "action_timestamp": "T3", "done": True, "info": {},
            },
        ]
        turns = group_by_predict_turn(iter(rows))
        assert len(turns) == 2
        t1, t2 = turns
        assert t1["turn_index"] == 1
        assert t1["predict_num"] == 1
        assert t1["actions"] == ["A", "B"]
        assert t1["step_nums"] == [1, 2]
        assert t1["last_action_timestamp"] == "T2"
        assert t1["response"] == "resp1"
        assert t2["actions"] == ["C"]

    def test_group_by_predict_turn_drops_env_oracle_fields(self):
        # Env-side oracle signals (shell errors, return codes, env
        # done flag) never reach the agent through its obs channel —
        # the reviser must not see them either. Planner reasoning /
        # guidance are agent-authored (same category as `response`)
        # and stay visible.
        rows = [
            {
                "step_num": 1, "predict_num": 1, "phase": "executor",
                "action": "cmd1", "response": "r",
                "action_timestamp": "T1", "done": True,
                "info": {
                    "controller_result": {
                        "error": "command not found: foo",
                        "returncode": 127,
                    }
                },
                "planner": {
                    "reasoning": "planner's own chain of thought",
                    "guidance": "nudge for executor",
                },
            },
        ]
        turns = group_by_predict_turn(rows)
        assert len(turns) == 1
        t = turns[0]
        # Env-oracle fields must not survive into the turn dict.
        for forbidden in ("errors", "return_codes", "done", "planner"):
            assert forbidden not in t, (
                f"turn dict must not expose env-oracle field {forbidden!r}"
            )
        # Agent-authored fields survive (including planner's own output).
        assert t["response"] == "r"
        assert t["actions"] == ["cmd1"]
        assert t["phase"] == "executor"
        assert t["planner_reasoning"] == "planner's own chain of thought"
        assert t["planner_guidance"] == "nudge for executor"

    def test_render_strips_env_oracle_fields_end_to_end(self, tmp_path):
        # End-to-end: rows with controller errors / nonzero rc / done
        # true → rendered text must not surface any of those env-side
        # signals. Agent-authored fields (response, planner reasoning +
        # guidance, actions) all stay visible.
        rows = [
            {
                "step_num": 1, "predict_num": 1, "phase": "executor",
                "action": "cmd1", "response": "the agent's own words",
                "action_timestamp": "T1", "done": True,
                "info": {
                    "controller_result": {
                        "error": "SECRET_ERROR_STRING",
                        "returncode": 42,
                    }
                },
                "planner": {
                    "reasoning": "planner's own chain of thought",
                    "guidance": "guidance for executor",
                },
            },
        ]
        turns = group_by_predict_turn(rows)
        blocks = render_step_detailed(turns[0], _StubKit(), tmp_path)
        rendered = blocks[0]["text"]
        for forbidden in (
            "**Error:**", "**Return codes:**", "**Done:**",
            "SECRET_ERROR_STRING", "127", "42",
        ):
            assert forbidden not in rendered, (
                f"render leaked env-oracle signal {forbidden!r}:\n{rendered}"
            )
        # Agent-authored content IS present.
        assert "the agent's own words" in rendered
        assert "cmd1" in rendered
        assert "planner's own chain of thought" in rendered
        assert "guidance for executor" in rendered

    def test_group_by_predict_turn_falls_back_to_step_num(self):
        rows = [
            {"step_num": 1, "action": "A", "response": "r"},
            {"step_num": 2, "action": "B", "response": "r"},
        ]
        turns = group_by_predict_turn(rows)
        assert len(turns) == 2  # no predict_num → per-step groups

    def test_chunk_steps_boundaries(self):
        assert chunk_steps([], 3) == []
        turns = [{"turn_index": i + 1} for i in range(7)]
        assert len(chunk_steps(turns, 3)) == 3  # 3 + 3 + 1
        assert len(chunk_steps(turns, 8)) == 1  # all in one chunk
        assert chunk_steps(turns, 0) == []

    def test_render_step_detailed_degrades_on_missing_step_dir(self, tmp_path):
        turn = {
            "turn_index": 1, "predict_num": 1, "phase": "act",
            "response": "hello", "step_nums": [1], "action_timestamps": ["T1"],
            "actions": ["clk"], "errors": [""], "return_codes": [0],
            "done": False, "planner": None, "total_history": 0,
            "last_step_num": 1, "last_action_timestamp": "T1",
        }
        blocks = render_step_detailed(turn, _StubKit(), tmp_path)
        # text header + screenshot-unavailable text block (no actual step dir)
        assert len(blocks) == 2
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "text"
        assert "observation unavailable" in blocks[1]["text"]

    def test_render_step_detailed_truncates_response(self, tmp_path):
        turn = {
            "turn_index": 1, "predict_num": 1, "phase": "",
            "response": "x" * 1000,
            "step_nums": [1], "action_timestamps": ["T1"],
            "actions": ["clk"],
            "last_step_num": 1, "last_action_timestamp": "T1",
        }
        blocks = render_step_detailed(turn, _StubKit(), tmp_path, response_char_limit=100)
        assert "...[truncated]" in blocks[0]["text"]

    def test_load_initial_obs_missing_step0(self, tmp_path):
        # No step_0_initial/ → empty list (older runs / benchmarks that
        # never saved obs_0); analyzer must degrade gracefully.
        assert load_initial_obs(tmp_path, _StubKit()) == []

    def test_load_initial_obs_with_screenshot(self, tmp_path):
        step0 = tmp_path / "step_0_initial"
        step0.mkdir()
        _make_png(step0 / "screenshot.png")
        blocks = load_initial_obs(tmp_path, _StubKit())
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image_url"


# ----------------------------------------------------------------------
# XML parsing — 3-level tolerance
# ----------------------------------------------------------------------


class TestXmlParsing:
    def test_parse_root_cause_clean(self):
        xml = """
<root_cause>
  <trajectory_summary>Agent did x, y, z. Final state looks complete.</trajectory_summary>
  <what_worked>
    <item>Turn 1: opened terminal with Ctrl+Alt+T as tutorial suggested.</item>
  </what_worked>
  <issues></issues>
  <outcome_assessment value="likely_success">DONE reached and final screenshot matches the task description.</outcome_assessment>
</root_cause>
"""
        rc = _parse_root_cause(xml)
        assert rc.trajectory_summary.startswith("Agent did x")
        assert rc.what_worked == [
            "Turn 1: opened terminal with Ctrl+Alt+T as tutorial suggested."
        ]
        assert rc.issues == []
        assert rc.outcome_assessment == "likely_success"
        assert "DONE reached" in rc.outcome_rationale
        assert "<root_cause>" in rc.raw_xml

    def test_parse_root_cause_multiple_issues(self):
        xml = """
blah some preamble
<root_cause>
  <trajectory_summary>Agent loops on mis-click recovery.</trajectory_summary>
  <what_worked></what_worked>
  <issues>
    <issue>
      <where>turn 3</where>
      <evidence>clicked wrong spot</evidence>
      <cause>wrong assumption about icon location</cause>
    </issue>
    <issue>
      <where>turn 5</where>
      <evidence>retried identical action</evidence>
      <cause>no reflection on prior failure</cause>
    </issue>
  </issues>
  <outcome_assessment value="likely_failure">Agent never reached the goal state; final screenshot still shows pre-task state.</outcome_assessment>
</root_cause>
trailing garbage
"""
        rc = _parse_root_cause(xml)
        assert rc.trajectory_summary == "Agent loops on mis-click recovery."
        assert len(rc.issues) == 2
        assert rc.issues[0]["where"] == "turn 3"
        assert rc.issues[0]["cause"] == "wrong assumption about icon location"
        assert rc.issues[1]["where"] == "turn 5"
        assert rc.outcome_assessment == "likely_failure"

    def test_parse_root_cause_et_fail_regex_fallback(self):
        """Unescaped ampersand makes ET.fromstring fail; regex fallback wins."""
        xml = """
<root_cause>
  <trajectory_summary>A & B triggered issues.</trajectory_summary>
  <what_worked></what_worked>
  <issues>
    <issue>
      <where>turn 2</where>
      <evidence>Action emitted & failed</evidence>
      <cause>Ampersand in response & not escaped</cause>
    </issue>
  </issues>
  <outcome_assessment value="uncertain">Mixed evidence & hard to tell.</outcome_assessment>
</root_cause>
"""
        rc = _parse_root_cause(xml)
        assert rc.trajectory_summary  # non-empty
        assert len(rc.issues) >= 1
        assert rc.issues[0]["cause"]
        assert rc.outcome_assessment == "uncertain"

    def test_regex_root_cause_fallback_direct(self):
        raw = """
<root_cause>
  <trajectory_summary>Pure regex path.</trajectory_summary>
  <what_worked>
    <item>W1</item>
    <item>W2</item>
  </what_worked>
  <issues>
    <issue>
      <where>turn X</where>
      <evidence>E1</evidence>
      <cause>cause-inside-issue</cause>
    </issue>
  </issues>
  <outcome_assessment value="likely_success">clear finish</outcome_assessment>
</root_cause>
"""
        rc = _regex_root_cause_fallback(raw, raw)
        assert rc.trajectory_summary == "Pure regex path."
        assert rc.what_worked == ["W1", "W2"]
        assert rc.issues[0]["where"] == "turn X"
        assert rc.issues[0]["cause"] == "cause-inside-issue"
        assert rc.outcome_assessment == "likely_success"
        assert rc.outcome_rationale == "clear finish"

    def test_parse_root_cause_preamble_plus_malformed_xml_uses_regex_fallback(self):
        """Preamble prose + ampersand-broken XML: ET fails, regex wins."""
        response = """
Here is my analysis. I noticed a few issues below.

<root_cause>
  <trajectory_summary>Agent mis-clicks & retries.</trajectory_summary>
  <what_worked></what_worked>
  <issues>
    <issue>
      <where>turn 4</where>
      <evidence>clicked at (x, y) but target was (x', y') & no feedback was given</evidence>
      <cause>agent reasoned in absolute pixel coordinates</cause>
    </issue>
  </issues>
  <outcome_assessment value="likely_failure">task not completed</outcome_assessment>
</root_cause>

Hope this helps!
"""
        rc = _parse_root_cause(response)
        # Either ET parsed or the regex fallback did — either way we need
        # the key data populated.
        assert rc.trajectory_summary  # non-empty
        assert len(rc.issues) == 1
        assert "pixel" in rc.issues[0]["cause"].lower()
        assert rc.outcome_assessment == "likely_failure"
        # raw_xml must not include the preamble/trailer prose.
        assert rc.raw_xml.startswith("<root_cause>")
        assert rc.raw_xml.endswith("</root_cause>")

    def test_parse_root_cause_no_tag_returns_empty_with_warning(self, caplog):
        """Total parse failure returns empty + warn + preserves raw_response."""
        raw = "I could not figure out what to emit."
        with caplog.at_level(logging.WARNING, logger="anything2skill.reviser"):
            rc = _parse_root_cause(raw)
        assert rc.trajectory_summary == ""
        assert rc.issues == []
        assert rc.what_worked == []
        assert rc.outcome_assessment == ""
        assert rc.raw_xml == ""
        # raw_response preserved for debugging; never injected into refiner prompt.
        assert rc.raw_response == raw
        assert any(
            "No <root_cause> tag" in r.message for r in caplog.records
        )

    def test_regex_fallback_all_fields_empty_warns_and_preserves_raw(self, caplog):
        """Fallback that extracts no fields logs and preserves raw text."""
        # Outer <root_cause> present but contains nothing parsable.
        raw = "<root_cause>totally not tagged content</root_cause>"
        with caplog.at_level(logging.WARNING, logger="anything2skill.reviser"):
            rc = _regex_root_cause_fallback(raw, raw)
        assert rc.trajectory_summary == ""
        assert rc.issues == []
        assert rc.what_worked == []
        assert rc.outcome_assessment == ""
        assert rc.raw_response == raw
        assert any(
            "extracted no fields" in r.message for r in caplog.records
        )

    def test_regex_fallback_salvages_missing_issues_wrapper(self):
        """Salvage <issue> entries when the <issues>...</issues> wrapper is absent.

        Missing closing-wrapper tags are exactly the malformed-XML case the
        fallback exists for; scoped extraction must not regress that path.
        """
        raw = """<root_cause>
  <trajectory_summary>Wrapper tag dropped.</trajectory_summary>
  <issue>
    <where>turn 1</where>
    <evidence>no closing issues tag</evidence>
    <cause>VLM truncated mid-emit</cause>
  </issue>
  <outcome_assessment value="uncertain">truncated mid-emit</outcome_assessment>
</root_cause>"""
        rc = _regex_root_cause_fallback(raw, raw)
        assert len(rc.issues) == 1
        assert rc.issues[0]["where"] == "turn 1"
        assert rc.issues[0]["cause"] == "VLM truncated mid-emit"
        assert rc.outcome_assessment == "uncertain"

    def test_parse_summary_clean(self):
        assert _parse_summary("<summary>foo bar</summary>") == "foo bar"

    def test_parse_summary_missing_tag_uses_whole_response(self):
        assert _parse_summary("  no tag here  ") == "no tag here"

    def test_serialize_root_cause_roundtrip(self):
        """_serialize_root_cause_fallback → _parse_root_cause keeps all fields."""
        from anything2skill.reviser.refiner import _serialize_root_cause_fallback

        rc = RootCauseAnalysis(
            trajectory_summary="run completed cleanly",
            what_worked=["opened terminal via shortcut", "verified zip contents"],
            issues=[{"where": "turn 6", "evidence": "ev", "cause": "c"}],
            outcome_assessment="likely_success",
            outcome_rationale="final state matched task description",
        )
        xml = _serialize_root_cause_fallback(rc)
        parsed = _parse_root_cause(xml)
        assert parsed.trajectory_summary == rc.trajectory_summary
        assert parsed.what_worked == rc.what_worked
        assert parsed.issues == rc.issues
        assert parsed.outcome_assessment == rc.outcome_assessment
        assert parsed.outcome_rationale == rc.outcome_rationale

    def test_serialize_root_cause_roundtrip_escapes_special_chars(self):
        """Fields containing &/<> must produce well-formed XML that ET parses."""
        from anything2skill.reviser.refiner import _serialize_root_cause_fallback

        rc = RootCauseAnalysis(
            trajectory_summary="ran `cp a && cp b` then <check> output",
            what_worked=["cmd `grep foo & bar > out.txt` succeeded"],
            issues=[
                {
                    "where": "turn 2",
                    "evidence": "shell emitted A & B <error>",
                    "cause": "unescaped meta-chars <x>",
                }
            ],
            outcome_assessment="likely_failure",
            outcome_rationale="final state did not match <goal>",
        )
        xml = _serialize_root_cause_fallback(rc)
        # Must be well-formed XML (ET path, not regex fallback).
        import xml.etree.ElementTree as ET
        ET.fromstring(xml)
        # And roundtrip is lossless.
        parsed = _parse_root_cause(xml)
        assert parsed.trajectory_summary == rc.trajectory_summary
        assert parsed.what_worked == rc.what_worked
        assert parsed.issues == rc.issues
        assert parsed.outcome_assessment == rc.outcome_assessment
        assert parsed.outcome_rationale == rc.outcome_rationale


# ----------------------------------------------------------------------
# Analyzer — single/multi chunk + no-skills-in-prompt invariant
# ----------------------------------------------------------------------


class TestAnalyzer:
    def _make_traj(self, tmp_path, num_turns: int) -> Path:
        rows = []
        step_dirs = {}
        for i in range(num_turns):
            step_num = i + 1
            ts = f"T{step_num:05d}"
            rows.append(
                {
                    "step_num": step_num, "predict_num": step_num, "phase": "act",
                    "action": f"act{step_num}", "response": f"resp{step_num}",
                    "action_timestamp": ts, "done": False, "info": {},
                }
            )
            step_dirs[step_num] = tmp_path / f"step_{step_num}_{ts}"
        traj = tmp_path / "traj.jsonl"
        _write_traj(traj, rows, step_dirs)
        return traj

    def test_analyze_single_chunk(self, tmp_path):
        traj = self._make_traj(tmp_path, num_turns=3)
        vlm = MagicMock()
        vlm.chat.return_value = (
            "<root_cause>"
            "<trajectory_summary>summary</trajectory_summary>"
            "<what_worked><item>w</item></what_worked>"
            "<issues></issues>"
            "<outcome_assessment value=\"likely_success\">ok</outcome_assessment>"
            "</root_cause>"
        )
        analyzer = ReviserAnalyzer(vlm=vlm, kit=_StubKit(), chunk_size=15)
        rc = analyzer.analyze(
            traj_path=str(traj), instruction="do x", result_dir=str(tmp_path),
        )
        assert vlm.chat.call_count == 1
        assert rc.trajectory_summary == "summary"
        assert rc.what_worked == ["w"]
        assert rc.outcome_assessment == "likely_success"

    def test_analyze_multi_chunk(self, tmp_path):
        traj = self._make_traj(tmp_path, num_turns=5)
        vlm = MagicMock()
        vlm.chat.side_effect = [
            "<summary>mid 1</summary>",
            "<summary>mid 2</summary>",
            "<root_cause>"
            "<trajectory_summary>final</trajectory_summary>"
            "<what_worked></what_worked>"
            "<issues></issues>"
            "<outcome_assessment value=\"uncertain\">mixed</outcome_assessment>"
            "</root_cause>",
        ]
        analyzer = ReviserAnalyzer(vlm=vlm, kit=_StubKit(), chunk_size=2)
        rc = analyzer.analyze(
            traj_path=str(traj), instruction="do x", result_dir=str(tmp_path),
        )
        assert vlm.chat.call_count == 3
        assert rc.trajectory_summary == "final"
        assert rc.outcome_assessment == "uncertain"
        # All 3 chunks persisted as audit dumps
        chunks_dir = tmp_path / "reviser_chunks"
        assert (chunks_dir / "chunk_1.json").is_file()
        assert (chunks_dir / "chunk_3.json").is_file()

    def test_analyze_prompt_omits_skill_vocabulary(self, tmp_path):
        """Plan §1: analyzer prompts must not reference skills at all."""
        import re as _re

        traj = self._make_traj(tmp_path, num_turns=2)
        captured: list[dict] = []

        def _capture(messages, **_):
            captured.append({"messages": messages})
            return (
                "<root_cause>"
                "<trajectory_summary>ok</trajectory_summary>"
                "<what_worked></what_worked>"
                "<issues></issues>"
                "<outcome_assessment value=\"likely_success\">ok</outcome_assessment>"
                "</root_cause>"
            )

        vlm = MagicMock()
        vlm.chat.side_effect = _capture

        # Kit with realistic reviser_guidance so kit-supplied text is
        # also checked.
        analyzer = ReviserAnalyzer(vlm=vlm, kit=_StubKit(guidance=""), chunk_size=5)
        analyzer.analyze(
            traj_path=str(traj), instruction="do x", result_dir=str(tmp_path),
        )

        # Only check the *system* messages — the user messages legitimately
        # contain verbatim agent responses which may mention "skill" if the
        # tutorial was about a skill.
        system_text = ""
        for call in captured:
            for msg in call["messages"]:
                if msg.get("role") != "system":
                    continue
                content = msg["content"]
                if isinstance(content, str):
                    system_text += content
                else:
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_text += block["text"]

        # Catches "skill", "skills", "Skill", "Skills" — any word-boundary
        # occurrence. We exclude "skillful" / "skilled" which are not in
        # any current template but would be benign even if they were.
        skill_hits = _re.findall(r"\bskills?\b", system_text, flags=_re.IGNORECASE)
        assert skill_hits == [], (
            f"analyzer system prompt contains forbidden skill vocabulary "
            f"{skill_hits!r}; phase-1 must be trajectory-only"
        )

    def test_analyze_writes_audit_to_audit_dir_not_result_dir(self, tmp_path):
        """reviser_chunks/ must land under audit_dir, leaving result_dir
        untouched. This is what protects canonical/attempt_1 when a
        cross-bucket reviser reads its trajectory."""
        traj = self._make_traj(tmp_path, num_turns=2)
        result_dir = tmp_path           # where step_*_TS/ screenshots live
        audit_dir = tmp_path / "next_attempt"
        audit_dir.mkdir()

        vlm = MagicMock()
        vlm.chat.return_value = (
            "<root_cause>"
            "<trajectory_summary>ok</trajectory_summary>"
            "<what_worked></what_worked>"
            "<issues></issues>"
            "<outcome_assessment value=\"likely_success\">ok</outcome_assessment>"
            "</root_cause>"
        )
        analyzer = ReviserAnalyzer(vlm=vlm, kit=_StubKit(), chunk_size=10)
        analyzer.analyze(
            traj_path=str(traj),
            instruction="i",
            result_dir=str(result_dir),
            audit_dir=str(audit_dir),
        )

        # Audit lands in audit_dir, not in result_dir.
        assert (audit_dir / "reviser_chunks" / "chunk_1.json").is_file()
        assert not (result_dir / "reviser_chunks").exists()

    def test_analyze_injects_initial_obs_in_chunk_1_only(self, tmp_path):
        # obs_0 header must appear in chunk 1's user message and NOT in
        # later chunks — otherwise the analyzer would see duplicate
        # grounding images and waste context.
        traj = self._make_traj(tmp_path, num_turns=4)
        step0 = tmp_path / "step_0_initial"
        step0.mkdir()
        _make_png(step0 / "screenshot.png")

        captured: list[list[dict]] = []

        def _capture(messages, **_):
            captured.append(messages)
            # chunk_size=2 → 2 chunks: chunk 1 emits summary, chunk 2 emits root_cause
            if len(captured) < 2:
                return "<summary>mid</summary>"
            return (
                "<root_cause>"
                "<trajectory_summary>p</trajectory_summary>"
                "<what_worked></what_worked>"
                "<issues></issues>"
                "<outcome_assessment value=\"likely_success\">ok</outcome_assessment>"
                "</root_cause>"
            )

        vlm = MagicMock()
        vlm.chat.side_effect = _capture

        analyzer = ReviserAnalyzer(vlm=vlm, kit=_StubKit(), chunk_size=2)
        analyzer.analyze(
            traj_path=str(traj), instruction="do x", result_dir=str(tmp_path),
        )
        assert len(captured) == 2

        def _user_text(messages):
            for m in messages:
                if m.get("role") == "user":
                    return " ".join(
                        b["text"]
                        for b in m["content"]
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
            return ""

        chunk1_text = _user_text(captured[0])
        chunk2_text = _user_text(captured[1])
        assert "Initial observation (before Turn 1)" in chunk1_text
        assert "NOT a turn" in chunk1_text
        assert "Initial observation" not in chunk2_text

    def test_analyze_skips_initial_obs_when_step0_missing(self, tmp_path):
        # No step_0_initial/ → analyzer still runs, just without a
        # grounding block. The label text must not appear.
        traj = self._make_traj(tmp_path, num_turns=2)
        captured: list[list[dict]] = []

        def _capture(messages, **_):
            captured.append(messages)
            return (
                "<root_cause>"
                "<trajectory_summary>p</trajectory_summary>"
                "<what_worked></what_worked>"
                "<issues></issues>"
                "<outcome_assessment value=\"likely_success\">ok</outcome_assessment>"
                "</root_cause>"
            )

        vlm = MagicMock()
        vlm.chat.side_effect = _capture

        analyzer = ReviserAnalyzer(vlm=vlm, kit=_StubKit(), chunk_size=10)
        analyzer.analyze(
            traj_path=str(traj), instruction="do x", result_dir=str(tmp_path),
        )
        user_msg = next(m for m in captured[0] if m.get("role") == "user")
        texts = " ".join(
            b["text"] for b in user_msg["content"]
            if isinstance(b, dict) and b.get("type") == "text"
        )
        assert "Initial observation" not in texts

    def test_analyze_degrades_when_kit_returns_empty_obs(self, tmp_path):
        traj = self._make_traj(tmp_path, num_turns=2)
        kit = _StubKit()
        kit.load_saved_observation = lambda step_dir: []  # e.g. Minecraft without hook
        vlm = MagicMock()
        vlm.chat.return_value = (
            "<root_cause>"
            "<trajectory_summary>done</trajectory_summary>"
            "<what_worked></what_worked>"
            "<issues></issues>"
            "<outcome_assessment value=\"likely_success\">ok</outcome_assessment>"
            "</root_cause>"
        )
        analyzer = ReviserAnalyzer(vlm=vlm, kit=kit, chunk_size=10)
        rc = analyzer.analyze(
            traj_path=str(traj), instruction="do x", result_dir=str(tmp_path),
        )
        assert rc.trajectory_summary == "done"


# ----------------------------------------------------------------------
# Refiner — no-op, image_map merge, budget knobs
# ----------------------------------------------------------------------


class TestRefiner:
    def _skills(self) -> Skills:
        return Skills(
            task_id="t1",
            instruction="do x",
            skills=[
                Skill(name="s1", description="d1", content="body1", images=[]),
            ],
        )

    def _tutorial(self, tmp_path) -> TutorialMaterial:
        # A tutorial with two images.
        img_a = tmp_path / "img_a.png"
        img_b = tmp_path / "img_b.png"
        _make_png(img_a)
        _make_png(img_b)
        return TutorialMaterial(
            task_id="t1",
            instruction="do x",
            content_type="html",
            body="<p>tutorial</p>",
            image_paths=[str(img_a), str(img_b)],
        )

    def test_refiner_calls_vlm_on_empty_analysis(self):
        vlm = MagicMock()
        vlm.chat.return_value = "# s1\n> d1\n\nbody1\n"
        refiner = ReviserRefiner(vlm=vlm, kit=_StubKit())
        empty = RootCauseAnalysis(trajectory_summary="clean")
        result = refiner.refine(
            skills=self._skills(),
            root_cause=empty,
            tutorial=TutorialMaterial(
                task_id="t1", instruction="", content_type="html",
                body="", image_paths=[],
            ),
            instruction="do x",
        )
        vlm.chat.assert_called_once()
        assert result.skills[0].name == "s1"

    def test_refiner_image_map_includes_tutorial_images(self, tmp_path):
        captured = {}

        def _chat(messages, **_):
            captured["messages"] = messages
            return (
                "# s1\n> d1'\n\nbody1'\n"
            )

        vlm = MagicMock()
        vlm.chat.side_effect = _chat

        refiner = ReviserRefiner(
            vlm=vlm, kit=_StubKit(),
            tutorial_image_cap=10, include_tutorial_in_refine=True,
        )
        rc = RootCauseAnalysis(
            trajectory_summary="issue",
            issues=[{"where": "t1", "evidence": "e", "cause": "c"}],
            raw_xml="<root_cause></root_cause>",
        )
        refiner.refine(
            skills=self._skills(),
            root_cause=rc,
            tutorial=self._tutorial(tmp_path),
            instruction="do x",
        )

        # Find the image filenames mentioned in the user message.
        user_msg = captured["messages"][-1]
        texts = [b["text"] for b in user_msg["content"] if b.get("type") == "text"]
        joined_texts = " ".join(texts)
        assert "img_a.png" in joined_texts
        assert "img_b.png" in joined_texts

    def test_refiner_respects_tutorial_image_cap(self, tmp_path):
        captured = {}

        def _chat(messages, **_):
            captured["messages"] = messages
            return "# s1\n> d1\n\nbody\n"

        vlm = MagicMock()
        vlm.chat.side_effect = _chat

        refiner = ReviserRefiner(
            vlm=vlm, kit=_StubKit(),
            tutorial_image_cap=1,
            include_tutorial_in_refine=True,
        )
        rc = RootCauseAnalysis(
            trajectory_summary="p",
            issues=[{"where": "w", "evidence": "e", "cause": "c"}],
            raw_xml="<root_cause></root_cause>",
        )
        refiner.refine(
            skills=self._skills(),
            root_cause=rc,
            tutorial=self._tutorial(tmp_path),
            instruction="do x",
        )
        user_msg = captured["messages"][-1]
        img_blocks = [b for b in user_msg["content"] if b.get("type") == "image_url"]
        assert len(img_blocks) == 1  # capped

    def test_refiner_include_tutorial_false_sends_no_images(self, tmp_path):
        captured = {}

        def _chat(messages, **_):
            captured["messages"] = messages
            return "# s1\n> d1\n\nbody\n"

        vlm = MagicMock()
        vlm.chat.side_effect = _chat

        refiner = ReviserRefiner(
            vlm=vlm, kit=_StubKit(),
            tutorial_image_cap=10,
            include_tutorial_in_refine=False,
        )
        rc = RootCauseAnalysis(
            trajectory_summary="p",
            issues=[{"where": "w", "evidence": "e", "cause": "c"}],
            raw_xml="<root_cause></root_cause>",
        )
        refiner.refine(
            skills=self._skills(),
            root_cause=rc,
            tutorial=self._tutorial(tmp_path),
            instruction="do x",
        )
        user_msg = captured["messages"][-1]
        img_blocks = [b for b in user_msg["content"] if b.get("type") == "image_url"]
        assert img_blocks == []

    def test_refiner_history_appears_in_user_prompt(self, tmp_path):
        """When history is provided, each attempt's raw_xml must show up
        in the user message under an '### Attempt N' header."""
        captured = {}

        def _chat(messages, **_):
            captured["messages"] = messages
            return "# s1\n> d1\n\nbody\n"

        vlm = MagicMock()
        vlm.chat.side_effect = _chat

        refiner = ReviserRefiner(vlm=vlm, kit=_StubKit())
        prior_rc = RootCauseAnalysis(
            trajectory_summary="prior attempt",
            raw_xml="<root_cause>prior-marker-xyz</root_cause>",
        )
        current_rc = RootCauseAnalysis(
            trajectory_summary="current attempt",
            issues=[{"where": "t1", "evidence": "e", "cause": "c"}],
            raw_xml="<root_cause>current-marker</root_cause>",
        )
        refiner.refine(
            skills=self._skills(),
            root_cause=current_rc,
            tutorial=self._tutorial(tmp_path),
            instruction="do x",
            history=[prior_rc],
        )
        user_msg = captured["messages"][-1]
        texts = " ".join(
            b["text"] for b in user_msg["content"]
            if isinstance(b, dict) and b.get("type") == "text"
        )
        assert "### Attempt 1" in texts
        assert "prior-marker-xyz" in texts
        # The current attempt's trajectory_summary also surfaces as its own section.
        assert "Trajectory Summary (current attempt)" in texts
        assert "current attempt" in texts

    def test_refiner_system_prompt_includes_outcome_gating(self, tmp_path):
        """System prompt must describe the likely_success minimal-edit rule
        so the model actually sees the guidance regardless of outcome value."""
        captured = {}

        def _chat(messages, **_):
            captured["messages"] = messages
            return "# s1\n> d1\n\nbody\n"

        vlm = MagicMock()
        vlm.chat.side_effect = _chat

        refiner = ReviserRefiner(vlm=vlm, kit=_StubKit())
        # Refiner requires at least one issue to run — pair likely_success
        # with a single minor issue so the gating text still has to be
        # present in the system prompt.
        rc = RootCauseAnalysis(
            trajectory_summary="ok",
            what_worked=["good step at turn 1"],
            issues=[{"where": "turn 4", "evidence": "e", "cause": "c"}],
            outcome_assessment="likely_success",
            raw_xml="<root_cause>ok</root_cause>",
        )
        refiner.refine(
            skills=self._skills(),
            root_cause=rc,
            tutorial=self._tutorial(tmp_path),
            instruction="do x",
        )
        system_msg = next(
            m for m in captured["messages"] if m.get("role") == "system"
        )
        content = system_msg["content"]
        text = content if isinstance(content, str) else " ".join(
            b["text"] for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
        assert "likely_success" in text
        # Gating tier surface text. "minimal" is the word we use for the
        # narrowest tier; the specific capitalisation / surrounding words
        # may evolve but the gate name should stay stable.
        assert "minimal" in text.lower()

    def test_refiner_reinforces_on_likely_success_with_no_issues(self, tmp_path):
        """likely_success + no issues → refiner still calls VLM to absorb
        <what_worked> into silent skills (reinforce mode).

        Also pins the system prompt: L237 now spells out reinforce mode
        rather than telling the VLM to pass skills through unchanged.
        """
        captured = {}

        def _chat(messages, **_):
            captured["messages"] = messages
            return "# s1\n> d1\n\nbody1\n"

        vlm = MagicMock()
        vlm.chat.side_effect = _chat
        refiner = ReviserRefiner(vlm=vlm, kit=_StubKit())
        rc = RootCauseAnalysis(
            trajectory_summary="clean run",
            what_worked=["opened terminal", "verified zip"],
            issues=[],
            outcome_assessment="likely_success",
            raw_xml="<root_cause/>",
        )
        result = refiner.refine(
            skills=self._skills(),
            root_cause=rc,
            tutorial=self._tutorial(tmp_path),
            instruction="do x",
        )
        vlm.chat.assert_called_once()
        assert result.skills[0].name == "s1"
        system = captured["messages"][0]["content"]
        assert "reinforce mode" in system
        assert "pass every skill through unchanged" not in system


# ----------------------------------------------------------------------
# skills_io — load_skills_from_dir uses YAML frontmatter parser
# ----------------------------------------------------------------------


class TestSkillsIo:
    def test_roundtrip_absolute_paths_multi_tutorial(self, tmp_path):
        img_a = tmp_path / "tutA" / "a.png"
        img_b = tmp_path / "tutB" / "b.png"
        _make_png(img_a)
        _make_png(img_b)
        skills = Skills(
            task_id="t1",
            instruction="do x",
            skills=[
                Skill(
                    name="multi",
                    description="d",
                    content="body",
                    images=[str(img_a), str(img_b)],
                ),
            ],
        )
        save_skills_to_dir(skills, tmp_path / "attempt_1" / "skills")
        reloaded = load_skills_from_dir(
            tmp_path / "attempt_1" / "skills",
            task_id="t1", instruction="do x",
        )
        assert reloaded is not None
        assert len(reloaded.skills) == 1
        images = reloaded.skills[0].images
        assert str(img_a.resolve()) in images
        assert str(img_b.resolve()) in images

    def test_missing_file_returns_none(self, tmp_path):
        assert load_skills_from_dir(tmp_path / "nope", "t", "") is None


# ----------------------------------------------------------------------
# scan_attempts + cleanup + meta / root_cause persistence
# ----------------------------------------------------------------------


class TestScanAndCleanup:
    def _write(self, path: Path, text: str = "") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_scan_attempt_dirs_mixed(self, tmp_path):
        canonical = tmp_path / "canonical"
        reviser = tmp_path / "reviser"

        # canonical attempt_1: fully complete
        self._write(canonical / "attempt_1" / "result.txt", "0.8")
        self._write(
            canonical / "attempt_1" / "meta.json",
            '{"score":0.8,"steps":3,"skills_count":2,"early_stop_triggered":false}',
        )
        self._write(canonical / "attempt_1" / "traj.jsonl", "{}")
        (canonical / "attempt_1" / "skills").mkdir(parents=True)
        (canonical / "attempt_1" / "skills" / "foo").mkdir()
        (canonical / "attempt_1" / "skills" / "foo" / "SKILL.md").write_text("---\nname: foo\ndescription: d\n---\nbody")

        # reviser attempt_2: skills + root_cause present, not completed (no result.txt / meta.json)
        (reviser / "attempt_2" / "skills").mkdir(parents=True)
        (reviser / "attempt_2" / "skills" / "bar").mkdir()
        (reviser / "attempt_2" / "skills" / "bar" / "SKILL.md").write_text("---\nname: bar\ndescription: d\n---\n")
        self._write(reviser / "attempt_2" / "root_cause.xml", "<root_cause/>")

        scanned = scan_attempts(canonical, reviser)
        assert set(scanned.keys()) == {1, 2}
        assert scanned[1]["completed"] is True
        assert scanned[1]["has_skills"] is True
        assert scanned[2]["completed"] is False
        assert scanned[2]["has_skills"] is True
        assert scanned[2]["has_root_cause"] is True

    def test_cleanup_runtime_artifacts_preserves_persistent(self, tmp_path):
        a = tmp_path / "attempt_2"
        a.mkdir()
        # Persistent artifacts we must KEEP
        (a / "skills").mkdir()
        (a / "skills" / "s").mkdir()
        (a / "skills" / "s" / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\n")
        (a / "root_cause.xml").write_text("<root_cause/>")
        (a / "meta.json").write_text('{"score":0.1}')
        # Execution artifacts we must REMOVE
        (a / "traj.jsonl").write_text("{}")
        (a / "result.txt").write_text("0.0")
        (a / "response.log").write_text("boo")
        (a / "runtime.log").write_text("l")
        (a / "reviser_chunks").mkdir()
        (a / "reviser_chunks" / "chunk_1.json").write_text("{}")
        (a / "step_1_TS").mkdir()
        (a / "step_1_TS" / "screenshot.png").write_text("bin")

        _cleanup_runtime_artifacts(a)

        assert (a / "skills" / "s" / "SKILL.md").is_file()
        assert (a / "root_cause.xml").is_file()
        assert (a / "meta.json").is_file()
        assert not (a / "traj.jsonl").exists()
        assert not (a / "result.txt").exists()
        assert not (a / "response.log").exists()
        assert not (a / "runtime.log").exists()
        assert not (a / "reviser_chunks").exists()
        assert not (a / "step_1_TS").exists()

    def test_save_attempt_meta_is_atomic(self, tmp_path):
        a = tmp_path / "attempt_1"
        a.mkdir()
        _save_attempt_meta(
            a, score=0.75, steps=5, skills_count=2,
            early_stop_triggered=False,
        )
        payload = json.loads((a / "meta.json").read_text(encoding="utf-8"))
        assert payload == {
            "score": 0.75, "steps": 5, "skills_count": 2,
            "early_stop_triggered": False,
        }

    def test_save_root_cause_writes_raw_xml(self, tmp_path):
        a = tmp_path / "attempt_2"
        a.mkdir()
        rc = RootCauseAnalysis(
            trajectory_summary="p",
            raw_xml=(
                "<root_cause><trajectory_summary>p</trajectory_summary>"
                "</root_cause>"
            ),
        )
        _save_root_cause(a, rc)
        assert (a / "root_cause.xml").read_text(encoding="utf-8") == rc.raw_xml

    def test_latest_attempt_scores_picks_highest_numbered(self, tmp_path):
        canonical = tmp_path / "canonical"
        reviser = tmp_path / "reviser"
        # attempt_1 score 0.8 (higher, but earlier)
        (canonical / "attempt_1").mkdir(parents=True)
        (canonical / "attempt_1" / "result.txt").write_text("0.8")
        (canonical / "attempt_1" / "meta.json").write_text(
            '{"score":0.8,"steps":9,"skills_count":1,'
            '"early_stop_triggered":false}',
        )
        # attempt_2 score 0.3 (lower, but latest)
        (reviser / "attempt_2").mkdir(parents=True)
        (reviser / "attempt_2" / "result.txt").write_text("0.3")
        (reviser / "attempt_2" / "meta.json").write_text(
            '{"score":0.3,"steps":4,"skills_count":2,'
            '"early_stop_triggered":false}',
        )
        scanned = scan_attempts(canonical, reviser)
        score, steps = _latest_attempt_scores(scanned)
        # "latest", not "best" — attempt_2 wins even with lower score.
        assert score == 0.3
        assert steps == 4


# ----------------------------------------------------------------------
# ReviserRunner resume with mocks
# ----------------------------------------------------------------------


class _FakeEnv:
    def reset(self, task):
        return {}
    def step(self, action, sleep):
        return {}, 0.0, True, {}
    def evaluate(self):
        return 1.0
    def close(self):
        pass


class TestReviserRunnerResume:
    @pytest.fixture
    def setup_dirs(self, tmp_path, monkeypatch):
        canonical = tmp_path / "canonical"
        reviser = tmp_path / "reviser"
        canonical.mkdir()
        reviser.mkdir()
        return canonical, reviser

    def _skills(self) -> Skills:
        return Skills(
            task_id="t1",
            instruction="instr",
            skills=[Skill(name="s", description="d", content="body")],
        )

    def test_resume_from_last_complete(self, tmp_path, monkeypatch, setup_dirs):
        canonical, reviser = setup_dirs
        # Pre-populate attempt_1 as completed
        (canonical / "attempt_1").mkdir()
        (canonical / "attempt_1" / "result.txt").write_text("0.5")
        (canonical / "attempt_1" / "meta.json").write_text(
            '{"score":0.5,"steps":3,"skills_count":1,'
            '"early_stop_triggered":false}',
        )
        save_skills_to_dir(self._skills(), canonical / "attempt_1" / "skills")

        call_counter = {"n": 0}

        def fake_run_single(**kw):
            call_counter["n"] += 1
            # Create traj.jsonl + result.txt inside the result_dir so resume
            # advances next time.
            rd = Path(kw["result_dir"])
            (rd / "traj.jsonl").write_text('{"step_num":1,"predict_num":1,"action":"a","response":"r","done":true,"info":{},"action_timestamp":"T1"}')
            (rd / "result.txt").write_text("0.9")
            return 0.9, 2

        monkeypatch.setattr("anything2skill.runner.run_single_task", fake_run_single)

        # Stub analyzer so refine() sees empty issues → early break.
        runner = ReviserRunner(
            kit=_StubKit(), reviser_cfg={"max_attempts": 2, "chunk_size": 5},
            vlm_factory=lambda: MagicMock(),
        )
        monkeypatch.setattr(
            runner, "_get_analyzer",
            lambda: MagicMock(analyze=MagicMock(return_value=RootCauseAnalysis(
                trajectory_summary="",
                issues=[{"where": "1", "evidence": "e", "cause": "r"}],
                what_worked=[],
                outcome_assessment="likely_failure",
                raw_xml="<root_cause/>",
            ))),
        )
        fake_refiner = MagicMock()
        fake_refiner.refine.return_value = self._skills()
        monkeypatch.setattr(runner, "_get_refiner", lambda: fake_refiner)

        score, steps = runner.run_with_reviser(
            agent_factory=lambda s: MagicMock(result_dir=""),
            env=_FakeEnv(),
            task=_FakeTask(),
            skills=self._skills(),
            tutorial=TutorialMaterial(
                task_id="t1", instruction="", content_type="html", body="",
                image_paths=[],
            ),
            max_steps=5,
            sleep_after_execution=0.0,
            canonical_task_dir=canonical,
            reviser_task_dir=reviser,
            max_attempts=2,
        )
        # attempt_1 was already complete — only attempt_2 should run.
        assert call_counter["n"] == 1
        assert (reviser / "attempt_2" / "result.txt").is_file()
        assert score == 0.9  # latest attempt's score (attempt_2)

    def test_task_json_written_to_both_buckets(
        self, tmp_path, monkeypatch, setup_dirs,
    ):
        """Cross-bucket mode must land task.json in BOTH canonical and
        reviser task dirs so each bucket's rebuild can find the task."""
        canonical, reviser = setup_dirs

        def _runner(**kw):
            rd = Path(kw["result_dir"])
            (rd / "traj.jsonl").write_text("{}")
            (rd / "result.txt").write_text("0.5")
            return 0.5, 2

        monkeypatch.setattr("anything2skill.runner.run_single_task", _runner)

        runner = ReviserRunner(
            kit=_StubKit(),
            reviser_cfg={"max_attempts": 1, "chunk_size": 5},
            vlm_factory=lambda: MagicMock(),
        )
        runner.run_with_reviser(
            agent_factory=lambda s: MagicMock(result_dir=""),
            env=_FakeEnv(),
            task=_FakeTask(instruction="dual-write"),
            skills=self._skills(),
            tutorial=None,
            max_steps=5,
            sleep_after_execution=0.0,
            canonical_task_dir=canonical,
            reviser_task_dir=reviser,
            max_attempts=1,
        )
        # Both dirs must have task.json, both with the same instruction.
        for d in (canonical, reviser):
            tj = d / "task.json"
            assert tj.is_file(), f"{d} missing task.json"
            assert json.loads(tj.read_text())["instruction"] == "dual-write"

    def test_max_attempts_1_no_refine(self, tmp_path, monkeypatch, setup_dirs):
        canonical, reviser = setup_dirs
        fake_run_single = MagicMock(return_value=(0.3, 4))
        fake_run_single.side_effect = lambda **kw: (
            (Path(kw["result_dir"]) / "traj.jsonl").write_text("{}"),
            (Path(kw["result_dir"]) / "result.txt").write_text("0.3"),
        ) and (0.3, 4) or (0.3, 4)
        # Simpler version that actually writes and returns
        def _runner(**kw):
            rd = Path(kw["result_dir"])
            (rd / "traj.jsonl").write_text("{}")
            (rd / "result.txt").write_text("0.3")
            return 0.3, 4
        monkeypatch.setattr("anything2skill.runner.run_single_task", _runner)

        analyzer_called = {"n": 0}

        def fake_vlm_factory():
            analyzer_called["n"] += 1
            return MagicMock()

        runner = ReviserRunner(
            kit=_StubKit(), reviser_cfg={"max_attempts": 1, "chunk_size": 5},
            vlm_factory=fake_vlm_factory,
        )
        runner.run_with_reviser(
            agent_factory=lambda s: MagicMock(result_dir=""),
            env=_FakeEnv(),
            task=_FakeTask(),
            skills=self._skills(),
            tutorial=None,
            max_steps=5,
            sleep_after_execution=0.0,
            canonical_task_dir=canonical,
            reviser_task_dir=reviser,
            max_attempts=1,
        )
        # VLM factory must never be invoked when max_attempts=1.
        assert analyzer_called["n"] == 0
        assert (canonical / "attempt_1" / "result.txt").is_file()
        assert (canonical / "attempt_1" / "meta.json").is_file()

    def test_all_complete_is_idempotent(self, tmp_path, monkeypatch, setup_dirs):
        canonical, reviser = setup_dirs
        (canonical / "attempt_1").mkdir()
        (canonical / "attempt_1" / "result.txt").write_text("0.4")
        (canonical / "attempt_1" / "meta.json").write_text(
            '{"score":0.4,"steps":2,"skills_count":1,'
            '"early_stop_triggered":false}',
        )

        run_called = MagicMock()
        monkeypatch.setattr("anything2skill.runner.run_single_task", run_called)

        runner = ReviserRunner(
            kit=_StubKit(),
            reviser_cfg={"max_attempts": 1, "chunk_size": 5},
            vlm_factory=lambda: MagicMock(),
        )
        score, steps = runner.run_with_reviser(
            agent_factory=lambda s: MagicMock(result_dir=""),
            env=_FakeEnv(),
            task=_FakeTask(),
            skills=self._skills(),
            tutorial=None,
            max_steps=5,
            sleep_after_execution=0.0,
            canonical_task_dir=canonical,
            reviser_task_dir=reviser,
            max_attempts=1,
        )
        run_called.assert_not_called()
        assert score == 0.4
        assert steps == 2

    def test_immediate_refresh_after_each_attempt(
        self, tmp_path, monkeypatch, setup_dirs,
    ):
        """experiment_results.json must grow as each attempt lands on disk."""
        canonical_root = tmp_path / "canon_root"
        reviser_root = tmp_path / "rev_root"
        task_subdir = Path("os") / "task1"
        canonical = canonical_root / task_subdir
        reviser = reviser_root / task_subdir
        canonical.mkdir(parents=True)
        reviser.mkdir(parents=True)

        snapshots: list[dict] = []
        exp_path = reviser_root / "experiment_results.json"

        def fake_run_single(**kw):
            rd = Path(kw["result_dir"])
            (rd / "traj.jsonl").write_text(
                '{"step_num":1,"predict_num":1,"action":"a","response":"r",'
                '"done":true,"info":{},"action_timestamp":"T"}'
            )
            (rd / "result.txt").write_text("0.5")
            return 0.5, 3

        monkeypatch.setattr(
            "anything2skill.runner.run_single_task", fake_run_single,
        )

        runner = ReviserRunner(
            kit=_StubKit(),
            reviser_cfg={"max_attempts": 2, "chunk_size": 5},
            vlm_factory=lambda: MagicMock(),
            canonical_result_base=str(canonical_root),
            reviser_result_base=str(reviser_root),
        )
        monkeypatch.setattr(
            runner, "_get_analyzer",
            lambda: MagicMock(analyze=MagicMock(return_value=RootCauseAnalysis(
                trajectory_summary="",
                issues=[{"where": "1", "evidence": "e", "cause": "r"}],
                what_worked=[],
                outcome_assessment="likely_failure",
                raw_xml="<root_cause/>",
            ))),
        )
        monkeypatch.setattr(
            runner, "_get_refiner",
            lambda: MagicMock(refine=MagicMock(return_value=self._skills())),
        )

        # Patch _save_attempt_meta to capture the JSON state *after* each
        # attempt's meta lands (the rebuild fires on the next line inside
        # the loop). Easier: patch _refresh_experiment_results to snapshot
        # after each call.
        original_refresh = runner._refresh_experiment_results

        def capture_refresh():
            original_refresh()
            if exp_path.is_file():
                snapshots.append(json.loads(exp_path.read_text()))

        monkeypatch.setattr(runner, "_refresh_experiment_results", capture_refresh)

        runner.run_with_reviser(
            agent_factory=lambda s: MagicMock(result_dir=""),
            env=_FakeEnv(),
            task=_FakeTask(),
            skills=self._skills(),
            tutorial=TutorialMaterial(
                task_id="t1", instruction="", content_type="html", body="",
                image_paths=[],
            ),
            max_steps=5,
            sleep_after_execution=0.0,
            canonical_task_dir=canonical,
            reviser_task_dir=reviser,
            max_attempts=2,
        )

        # Two attempts → at least two refreshes, and the attempt-2 snapshot
        # must include both "1" and "2" entries with the flag serialized.
        assert len(snapshots) >= 2
        first_attempts = snapshots[0]["results"][0]["attempts"]
        assert list(first_attempts.keys()) == ["1"]
        assert first_attempts["1"]["early_stop_triggered"] is False
        final_attempts = snapshots[-1]["results"][0]["attempts"]
        assert set(final_attempts.keys()) == {"1", "2"}
        for entry in final_attempts.values():
            assert entry["early_stop_triggered"] is False

    def test_refiner_receives_history_from_prior_attempts(
        self, tmp_path, monkeypatch, setup_dirs,
    ):
        """With max_attempts=3 the refiner call producing attempt_3 must
        receive attempt_2/root_cause.xml in its `history` kwarg."""
        canonical, reviser = setup_dirs

        def _runner(**kw):
            rd = Path(kw["result_dir"])
            (rd / "traj.jsonl").write_text(
                '{"step_num":1,"predict_num":1,"action":"a","response":"r",'
                '"done":true,"info":{},"action_timestamp":"T"}'
            )
            (rd / "result.txt").write_text("0.4")
            return 0.4, 2
        monkeypatch.setattr("anything2skill.runner.run_single_task", _runner)

        # Analyzer produces a deterministic rc with a recognisable raw_xml
        # so we can later assert it shows up in history.
        def _make_rc(attempt_tag: str) -> RootCauseAnalysis:
            return RootCauseAnalysis(
                trajectory_summary=f"summary for {attempt_tag}",
                issues=[{"where": "t1", "evidence": "e", "cause": "c"}],
                what_worked=[],
                outcome_assessment="likely_failure",
                raw_xml=f"<root_cause>rc-for-{attempt_tag}</root_cause>",
            )

        analyze_call = {"n": 0}

        def fake_analyze(*args, **kwargs):
            analyze_call["n"] += 1
            # analyze is called once per post-attempt review; tag by attempt index.
            return _make_rc(f"attempt_{analyze_call['n']}")

        runner = ReviserRunner(
            kit=_StubKit(),
            reviser_cfg={"max_attempts": 3, "chunk_size": 5},
            vlm_factory=lambda: MagicMock(),
        )
        monkeypatch.setattr(
            runner, "_get_analyzer",
            lambda: MagicMock(analyze=MagicMock(side_effect=fake_analyze)),
        )

        refine_history_snapshots: list[list[RootCauseAnalysis]] = []
        fake_refiner = MagicMock()

        def fake_refine(**kw):
            refine_history_snapshots.append(list(kw.get("history") or []))
            return self._skills()

        fake_refiner.refine.side_effect = fake_refine
        monkeypatch.setattr(runner, "_get_refiner", lambda: fake_refiner)

        runner.run_with_reviser(
            agent_factory=lambda s: MagicMock(result_dir=""),
            env=_FakeEnv(),
            task=_FakeTask(),
            skills=self._skills(),
            tutorial=TutorialMaterial(
                task_id="t1", instruction="", content_type="html", body="",
                image_paths=[],
            ),
            max_steps=5,
            sleep_after_execution=0.0,
            canonical_task_dir=canonical,
            reviser_task_dir=reviser,
            max_attempts=3,
        )

        # Two refines total (after attempt_1, after attempt_2).
        assert len(refine_history_snapshots) == 2
        # Refining for attempt_2 (= first refine): history empty — no prior rc.
        assert refine_history_snapshots[0] == []
        # Refining for attempt_3 (= second refine): history contains rc from
        # attempt_2, which was saved by the loop between the two refines.
        assert len(refine_history_snapshots[1]) == 1
        assert (
            "rc-for-attempt_1"
            in refine_history_snapshots[1][0].raw_xml
        )

    def test_recovery_backfills_tag_from_seeded_root_cause(
        self, tmp_path, monkeypatch, setup_dirs,
    ):
        """Crash after saving attempt_2/root_cause.xml but before tagging
        attempt_1 → recovery path 2 reads the rc, sees empty issues, and
        backfills attempt_1's meta with early_stop_triggered=True.

        Without the backfill the main loop would re-tag a later attempt as
        the "first" early-stop."""
        canonical, reviser = setup_dirs
        # attempt_1 completed with False (crash happened before tag flip).
        (canonical / "attempt_1").mkdir()
        (canonical / "attempt_1" / "result.txt").write_text("0.5")
        (canonical / "attempt_1" / "meta.json").write_text(
            '{"score":0.5,"steps":3,"skills_count":1,'
            '"early_stop_triggered":false}',
        )
        (canonical / "attempt_1" / "traj.jsonl").write_text(
            '{"step_num":1,"predict_num":1,"action":"a","response":"r",'
            '"done":true,"info":{},"action_timestamp":"T"}'
        )
        save_skills_to_dir(self._skills(), canonical / "attempt_1" / "skills")
        # attempt_2/root_cause.xml landed with empty issues; skills dir
        # was never written (crash window).
        (reviser / "attempt_2").mkdir()
        (reviser / "attempt_2" / "root_cause.xml").write_text(
            "<root_cause>"
            "<trajectory_summary>ok</trajectory_summary>"
            "<what_worked><item>opened</item></what_worked>"
            "<issues></issues>"
            '<outcome_assessment value="likely_success"></outcome_assessment>'
            "</root_cause>"
        )

        _install_fake_run(monkeypatch)

        runner = ReviserRunner(
            kit=_StubKit(),
            reviser_cfg={"max_attempts": 3, "chunk_size": 5},
            vlm_factory=lambda: MagicMock(),
        )
        monkeypatch.setattr(
            runner, "_get_analyzer",
            lambda: MagicMock(analyze=MagicMock(return_value=_empty_rc())),
        )
        fake_refiner = MagicMock()
        fake_refiner.refine.return_value = self._skills()
        monkeypatch.setattr(runner, "_get_refiner", lambda: fake_refiner)

        runner.run_with_reviser(
            agent_factory=lambda s: MagicMock(result_dir=""),
            env=_FakeEnv(),
            task=_FakeTask(),
            skills=self._skills(),
            tutorial=TutorialMaterial(
                task_id="t1", instruction="", content_type="html", body="",
                image_paths=[],
            ),
            max_steps=5,
            sleep_after_execution=0.0,
            canonical_task_dir=canonical,
            reviser_task_dir=reviser,
            max_attempts=3,
        )

        # attempt_1 was backfilled during recovery; attempts 2/3 stay False.
        m1 = json.loads((canonical / "attempt_1" / "meta.json").read_text())
        m2 = json.loads((reviser / "attempt_2" / "meta.json").read_text())
        m3 = json.loads((reviser / "attempt_3" / "meta.json").read_text())
        assert m1["early_stop_triggered"] is True
        assert m2["early_stop_triggered"] is False
        assert m3["early_stop_triggered"] is False

    def test_recovery_backfills_likely_success_with_issues(
        self, tmp_path, monkeypatch, setup_dirs,
    ):
        canonical, reviser = setup_dirs
        (canonical / "attempt_1").mkdir()
        (canonical / "attempt_1" / "result.txt").write_text("0.5")
        (canonical / "attempt_1" / "meta.json").write_text(
            '{"score":0.5,"steps":3,"skills_count":1,'
            '"early_stop_triggered":false}',
        )
        save_skills_to_dir(self._skills(), canonical / "attempt_1" / "skills")

        (reviser / "attempt_2").mkdir()
        (reviser / "attempt_2" / "root_cause.xml").write_text(
            "<root_cause>"
            "<trajectory_summary>ok</trajectory_summary>"
            "<what_worked><item>opened</item></what_worked>"
            "<issues><issue>"
            "<where>turn 1</where>"
            "<evidence>minor</evidence>"
            "<cause>minor</cause>"
            "</issue></issues>"
            '<outcome_assessment value="likely_success"></outcome_assessment>'
            "</root_cause>"
        )

        _install_fake_run(monkeypatch)
        runner = ReviserRunner(
            kit=_StubKit(),
            reviser_cfg={
                "max_attempts": 2,
                "chunk_size": 5,
                "early_stop_signal": "likely_success",
            },
            vlm_factory=lambda: MagicMock(),
        )
        fake_refiner = MagicMock()
        fake_refiner.refine.return_value = self._skills()
        monkeypatch.setattr(runner, "_get_refiner", lambda: fake_refiner)

        runner.run_with_reviser(
            agent_factory=lambda s: MagicMock(result_dir=""),
            env=_FakeEnv(),
            task=_FakeTask(),
            skills=self._skills(),
            tutorial=TutorialMaterial(
                task_id="t1", instruction="", content_type="html", body="",
                image_paths=[],
            ),
            max_steps=5,
            sleep_after_execution=0.0,
            canonical_task_dir=canonical,
            reviser_task_dir=reviser,
            max_attempts=2,
        )

        m1 = json.loads((canonical / "attempt_1" / "meta.json").read_text())
        m2 = json.loads((reviser / "attempt_2" / "meta.json").read_text())
        assert m1["early_stop_triggered"] is True
        assert m2["early_stop_triggered"] is False

    def test_recovery_likely_success_ignores_empty_likely_failure(
        self, tmp_path, monkeypatch, setup_dirs,
    ):
        canonical, reviser = setup_dirs
        (canonical / "attempt_1").mkdir()
        (canonical / "attempt_1" / "result.txt").write_text("0.5")
        (canonical / "attempt_1" / "meta.json").write_text(
            '{"score":0.5,"steps":3,"skills_count":1,'
            '"early_stop_triggered":false}',
        )
        save_skills_to_dir(self._skills(), canonical / "attempt_1" / "skills")

        (reviser / "attempt_2").mkdir()
        (reviser / "attempt_2" / "root_cause.xml").write_text(
            "<root_cause>"
            "<trajectory_summary>failed</trajectory_summary>"
            "<what_worked></what_worked>"
            "<issues></issues>"
            '<outcome_assessment value="likely_failure"></outcome_assessment>'
            "</root_cause>"
        )

        _install_fake_run(monkeypatch)
        runner = ReviserRunner(
            kit=_StubKit(),
            reviser_cfg={
                "max_attempts": 2,
                "chunk_size": 5,
                "early_stop_signal": "likely_success",
            },
            vlm_factory=lambda: MagicMock(),
        )
        fake_refiner = MagicMock()
        fake_refiner.refine.return_value = self._skills()
        monkeypatch.setattr(runner, "_get_refiner", lambda: fake_refiner)

        runner.run_with_reviser(
            agent_factory=lambda s: MagicMock(result_dir=""),
            env=_FakeEnv(),
            task=_FakeTask(),
            skills=self._skills(),
            tutorial=TutorialMaterial(
                task_id="t1", instruction="", content_type="html", body="",
                image_paths=[],
            ),
            max_steps=5,
            sleep_after_execution=0.0,
            canonical_task_dir=canonical,
            reviser_task_dir=reviser,
            max_attempts=2,
        )

        m1 = json.loads((canonical / "attempt_1" / "meta.json").read_text())
        m2 = json.loads((reviser / "attempt_2" / "meta.json").read_text())
        assert m1["early_stop_triggered"] is False
        assert m2["early_stop_triggered"] is False

    def test_greedy_resume_does_not_retag(
        self, tmp_path, monkeypatch, setup_dirs,
    ):
        """Restart after a tagged attempt: further empty-issues events in
        the resumed run do NOT get re-tagged."""
        canonical, reviser = setup_dirs
        (canonical / "attempt_1").mkdir()
        (canonical / "attempt_1" / "result.txt").write_text("0.5")
        (canonical / "attempt_1" / "meta.json").write_text(
            '{"score":0.5,"steps":3,"skills_count":1,'
            '"early_stop_triggered":true}',
        )
        save_skills_to_dir(self._skills(), canonical / "attempt_1" / "skills")

        _install_fake_run(monkeypatch)

        runner = ReviserRunner(
            kit=_StubKit(),
            reviser_cfg={"max_attempts": 3, "chunk_size": 5},
            vlm_factory=lambda: MagicMock(),
        )
        monkeypatch.setattr(
            runner, "_get_analyzer",
            lambda: MagicMock(analyze=MagicMock(return_value=_empty_rc())),
        )
        fake_refiner = MagicMock()
        fake_refiner.refine.return_value = self._skills()
        monkeypatch.setattr(runner, "_get_refiner", lambda: fake_refiner)

        runner.run_with_reviser(
            agent_factory=lambda s: MagicMock(result_dir=""),
            env=_FakeEnv(),
            task=_FakeTask(),
            skills=self._skills(),
            tutorial=TutorialMaterial(
                task_id="t1", instruction="", content_type="html", body="",
                image_paths=[],
            ),
            max_steps=5,
            sleep_after_execution=0.0,
            canonical_task_dir=canonical,
            reviser_task_dir=reviser,
            max_attempts=3,
        )

        m1 = json.loads((canonical / "attempt_1" / "meta.json").read_text())
        m2 = json.loads((reviser / "attempt_2" / "meta.json").read_text())
        m3 = json.loads((reviser / "attempt_3" / "meta.json").read_text())
        assert m1["early_stop_triggered"] is True
        assert m2["early_stop_triggered"] is False
        assert m3["early_stop_triggered"] is False


# ----------------------------------------------------------------------
# Greedy / force-continue: empty issues no longer short-circuits the loop
# ----------------------------------------------------------------------


def _empty_rc(raw_xml: str = "<root_cause/>") -> RootCauseAnalysis:
    return RootCauseAnalysis(
        trajectory_summary="clean",
        issues=[],
        what_worked=["opened app"],
        outcome_assessment="likely_success",
        raw_xml=raw_xml,
    )


def _nonempty_rc(raw_xml: str = "<root_cause/>") -> RootCauseAnalysis:
    return RootCauseAnalysis(
        trajectory_summary="bad",
        issues=[{"where": "1", "evidence": "e", "cause": "c"}],
        what_worked=[],
        outcome_assessment="likely_failure",
        raw_xml=raw_xml,
    )


class TestEarlyStopSignal:
    def test_normalize_defaults_to_no_issue(self):
        assert _normalize_early_stop_signal(None) == "no_issue"

    def test_normalize_accepts_supported_values(self):
        assert _normalize_early_stop_signal("no_issue") == "no_issue"
        assert _normalize_early_stop_signal("likely_success") == "likely_success"

    def test_normalize_rejects_unknown_values(self):
        with pytest.raises(ValueError):
            _normalize_early_stop_signal("empty_issues")

    def test_no_issue_signal_matches_empty_issues_only(self):
        assert _matches_early_stop_signal(_empty_rc(), "no_issue") is True
        assert _matches_early_stop_signal(_nonempty_rc(), "no_issue") is False

    def test_likely_success_signal_matches_outcome_only(self):
        likely_success_with_issue = RootCauseAnalysis(
            issues=[{"where": "1", "evidence": "e", "cause": "c"}],
            outcome_assessment="likely_success",
        )
        empty_likely_failure = RootCauseAnalysis(
            issues=[],
            outcome_assessment="likely_failure",
        )

        assert (
            _matches_early_stop_signal(
                likely_success_with_issue, "likely_success",
            )
            is True
        )
        assert (
            _matches_early_stop_signal(empty_likely_failure, "likely_success")
            is False
        )


def _install_fake_run(monkeypatch, score: float = 0.5, steps: int = 2):
    def _runner(**kw):
        rd = Path(kw["result_dir"])
        (rd / "traj.jsonl").write_text(
            '{"step_num":1,"predict_num":1,"action":"a","response":"r",'
            '"done":true,"info":{},"action_timestamp":"T"}'
        )
        (rd / "result.txt").write_text(str(score))
        return score, steps
    monkeypatch.setattr(
        "anything2skill.runner.run_single_task", _runner,
    )


class TestReviserRunnerGreedy:
    @pytest.fixture
    def setup_dirs(self, tmp_path):
        canonical = tmp_path / "canonical"
        reviser = tmp_path / "reviser"
        canonical.mkdir()
        reviser.mkdir()
        return canonical, reviser

    def _skills(self) -> Skills:
        return Skills(
            task_id="t1",
            instruction="instr",
            skills=[Skill(name="s", description="d", content="body")],
        )

    def _make_runner(
        self, monkeypatch, *, analyzer_rcs, max_attempts, reviser_cfg=None,
    ) -> tuple[ReviserRunner, MagicMock]:
        _install_fake_run(monkeypatch)
        cfg = {"max_attempts": max_attempts, "chunk_size": 5}
        cfg.update(reviser_cfg or {})
        runner = ReviserRunner(
            kit=_StubKit(),
            reviser_cfg=cfg,
            vlm_factory=lambda: MagicMock(),
        )
        if isinstance(analyzer_rcs, list):
            it = iter(analyzer_rcs)
            analyze = MagicMock(side_effect=lambda *a, **kw: next(it))
        else:
            analyze = MagicMock(return_value=analyzer_rcs)
        monkeypatch.setattr(
            runner, "_get_analyzer",
            lambda: MagicMock(analyze=analyze),
        )
        fake_refiner = MagicMock()
        fake_refiner.refine.return_value = self._skills()
        monkeypatch.setattr(runner, "_get_refiner", lambda: fake_refiner)
        return runner, fake_refiner

    def _invoke(
        self, runner, canonical, reviser, *, tutorial, max_attempts,
    ) -> None:
        runner.run_with_reviser(
            agent_factory=lambda s: MagicMock(result_dir=""),
            env=_FakeEnv(),
            task=_FakeTask(),
            skills=self._skills(),
            tutorial=tutorial,
            max_steps=5,
            sleep_after_execution=0.0,
            canonical_task_dir=canonical,
            reviser_task_dir=reviser,
            max_attempts=max_attempts,
        )

    def test_continues_past_empty_issues(self, monkeypatch, setup_dirs):
        canonical, reviser = setup_dirs
        runner, fake_refiner = self._make_runner(
            monkeypatch, analyzer_rcs=_empty_rc(), max_attempts=3,
        )
        tutorial = TutorialMaterial(
            task_id="t1", instruction="", content_type="html", body="",
            image_paths=[],
        )
        self._invoke(
            runner, canonical, reviser, tutorial=tutorial, max_attempts=3,
        )

        assert (canonical / "attempt_1" / "meta.json").is_file()
        assert (reviser / "attempt_2" / "meta.json").is_file()
        assert (reviser / "attempt_3" / "meta.json").is_file()

        m1 = json.loads((canonical / "attempt_1" / "meta.json").read_text())
        m2 = json.loads((reviser / "attempt_2" / "meta.json").read_text())
        m3 = json.loads((reviser / "attempt_3" / "meta.json").read_text())
        assert m1["early_stop_triggered"] is True
        assert m2["early_stop_triggered"] is False
        assert m3["early_stop_triggered"] is False

        # Refiner runs between consecutive attempts: 1→2 and 2→3.
        assert fake_refiner.refine.call_count == 2

    def test_mixed_rc_only_first_empty_marked(self, monkeypatch, setup_dirs):
        canonical, reviser = setup_dirs
        # attempt_1 → non-empty; attempt_2 → empty (mark); attempt_3 → empty.
        runner, _ = self._make_runner(
            monkeypatch,
            analyzer_rcs=[_nonempty_rc(), _empty_rc(), _empty_rc()],
            max_attempts=4,
        )
        tutorial = TutorialMaterial(
            task_id="t1", instruction="", content_type="html", body="",
            image_paths=[],
        )
        self._invoke(
            runner, canonical, reviser, tutorial=tutorial, max_attempts=4,
        )

        m1 = json.loads((canonical / "attempt_1" / "meta.json").read_text())
        m2 = json.loads((reviser / "attempt_2" / "meta.json").read_text())
        m3 = json.loads((reviser / "attempt_3" / "meta.json").read_text())
        m4 = json.loads((reviser / "attempt_4" / "meta.json").read_text())
        assert m1["early_stop_triggered"] is False
        assert m2["early_stop_triggered"] is True
        assert m3["early_stop_triggered"] is False
        # attempt_4 is the last round — no post-analyze fires.
        assert m4["early_stop_triggered"] is False

    def test_tutorial_none_still_breaks_on_empty_issues(
        self, monkeypatch, setup_dirs,
    ):
        # Vanilla baseline: no tutorial means no refine material, so the
        # loop preserves its early exit even though the attempt is tagged.
        canonical, reviser = setup_dirs
        runner, _ = self._make_runner(
            monkeypatch, analyzer_rcs=_empty_rc(), max_attempts=3,
        )
        self._invoke(
            runner, canonical, reviser, tutorial=None, max_attempts=3,
        )

        assert (canonical / "attempt_1" / "meta.json").is_file()
        assert not (reviser / "attempt_2" / "meta.json").exists()
        m1 = json.loads((canonical / "attempt_1" / "meta.json").read_text())
        assert m1["early_stop_triggered"] is True

    def test_likely_success_mode_tags_nonempty_issues(
        self, monkeypatch, setup_dirs,
    ):
        canonical, reviser = setup_dirs
        runner, _ = self._make_runner(
            monkeypatch,
            analyzer_rcs=RootCauseAnalysis(
                trajectory_summary="mostly ok",
                issues=[{"where": "1", "evidence": "minor", "cause": "minor"}],
                what_worked=["opened app"],
                outcome_assessment="likely_success",
                raw_xml="<root_cause/>",
            ),
            max_attempts=2,
            reviser_cfg={"early_stop_signal": "likely_success"},
        )
        tutorial = TutorialMaterial(
            task_id="t1", instruction="", content_type="html", body="",
            image_paths=[],
        )
        self._invoke(
            runner, canonical, reviser, tutorial=tutorial, max_attempts=2,
        )

        m1 = json.loads((canonical / "attempt_1" / "meta.json").read_text())
        m2 = json.loads((reviser / "attempt_2" / "meta.json").read_text())
        assert m1["early_stop_triggered"] is True
        assert m2["early_stop_triggered"] is False

    def test_likely_success_mode_ignores_empty_likely_failure(
        self, monkeypatch, setup_dirs,
    ):
        canonical, reviser = setup_dirs
        runner, _ = self._make_runner(
            monkeypatch,
            analyzer_rcs=RootCauseAnalysis(
                trajectory_summary="clean issue list but failed",
                issues=[],
                what_worked=[],
                outcome_assessment="likely_failure",
                raw_xml="<root_cause/>",
            ),
            max_attempts=2,
            reviser_cfg={"early_stop_signal": "likely_success"},
        )
        tutorial = TutorialMaterial(
            task_id="t1", instruction="", content_type="html", body="",
            image_paths=[],
        )
        self._invoke(
            runner, canonical, reviser, tutorial=tutorial, max_attempts=2,
        )

        m1 = json.loads((canonical / "attempt_1" / "meta.json").read_text())
        m2 = json.loads((reviser / "attempt_2" / "meta.json").read_text())
        assert m1["early_stop_triggered"] is False
        assert m2["early_stop_triggered"] is False


# ----------------------------------------------------------------------
# runner.py integration: response.log, run_name, rebuild_experiment_results
# ----------------------------------------------------------------------


class TestRunnerGlue:
    def test_compute_run_names_bare_run_no_suffix(self):
        """max_attempts=1 = pure agent, no reviser → both names coincide."""
        from anything2skill.runner import _compute_run_names
        c, r = _compute_run_names(
            {"agent_mode": "simple", "model": "gpt-4o"},
            {"model": None},
            max_attempts=1,
        )
        assert c == "a2s-simple-gpt-4o"
        assert c == r

    def test_compute_run_names_bare_run_ignores_reviser_model(self):
        """Even if reviser.model is set, max_attempts=1 keeps one bucket —
        no reviser activity actually happens."""
        from anything2skill.runner import _compute_run_names
        c, r = _compute_run_names(
            {"agent_mode": "simple", "model": "gpt-4o"},
            {"model": "gpt-4o-mini"},
            max_attempts=1,
        )
        assert c == r == "a2s-simple-gpt-4o"

    def test_compute_run_names_reviser_run_same_model_still_has_suffix(self):
        """max_attempts>1 with reviser_model == agent_model must still
        carve out its own bucket — the bare-run baseline at canonical
        must not be overwritten by the reviser experiment."""
        from anything2skill.runner import _compute_run_names
        c, r = _compute_run_names(
            {"agent_mode": "simple", "model": "gpt-4o"},
            {"model": "gpt-4o"},
            max_attempts=2,
        )
        assert c == "a2s-simple-gpt-4o"
        assert r == "a2s-simple-gpt-4o-r_gpt-4o"

    def test_compute_run_names_reviser_run_different_model_has_suffix(self):
        from anything2skill.runner import _compute_run_names
        c, r = _compute_run_names(
            {"agent_mode": "simple", "model": "gpt-4o"},
            {"model": "gpt-4o-mini"},
            max_attempts=2,
        )
        assert c == "a2s-simple-gpt-4o"
        assert r == "a2s-simple-gpt-4o-r_gpt-4o-mini"

    def test_compute_run_names_reviser_model_defaults_to_agent(self):
        """reviser.model unset + max_attempts>1 → suffix uses agent model."""
        from anything2skill.runner import _compute_run_names
        c, r = _compute_run_names(
            {"agent_mode": "phased", "model": "gpt-4o"},
            {"model": None},
            max_attempts=3,
        )
        assert c == "a2s-phased-gpt-4o"
        assert r == "a2s-phased-gpt-4o-r_gpt-4o"

    def test_effective_max_attempts_clamped_for_vanilla(self, caplog):
        from anything2skill.runner import _effective_max_attempts
        with caplog.at_level(logging.WARNING, logger="anything2skill.runner"):
            assert _effective_max_attempts({"max_attempts": 5}, "vanilla") == 1
        assert any("clamping" in r.message for r in caplog.records)

    def test_effective_max_attempts_passes_through_non_vanilla(self):
        from anything2skill.runner import _effective_max_attempts
        assert _effective_max_attempts({"max_attempts": 3}, "simple") == 3
        assert _effective_max_attempts({}, "phased") == 1

    def test_append_response_log_format(self, tmp_path):
        from anything2skill.runner import _append_response_log
        _append_response_log(
            str(tmp_path), step_num=1, predict_num=1, phase="act",
            response="hello world",
        )
        _append_response_log(
            str(tmp_path), step_num=2, predict_num=2, phase="plan",
            response="another",
        )
        text = (tmp_path / "response.log").read_text(encoding="utf-8")
        assert "===== Step 1 (predict_num=1, phase=act) =====" in text
        assert "===== Step 2 (predict_num=2, phase=plan) =====" in text
        assert "hello world" in text
        assert "another" in text

    def test_rebuild_experiment_results_records_each_attempt(self, tmp_path):
        """Both attempts surface in the ``attempts`` dict, no best-of-N."""
        from anything2skill.runner import _rebuild_experiment_results
        task_dir = tmp_path / "os" / "task1"
        (task_dir / "attempt_1").mkdir(parents=True)
        (task_dir / "attempt_1" / "meta.json").write_text(
            '{"score":0.3,"steps":10,"skills_count":2,'
            '"early_stop_triggered":true}'
        )
        (task_dir / "attempt_2").mkdir(parents=True)
        (task_dir / "attempt_2" / "meta.json").write_text(
            '{"score":0.7,"steps":5,"skills_count":3,'
            '"early_stop_triggered":false}'
        )
        (task_dir / "task.json").write_text('{"instruction": "Open Bluetooth"}')

        _rebuild_experiment_results(str(tmp_path))
        data = json.loads((tmp_path / "experiment_results.json").read_text())
        results = data["results"]
        assert len(results) == 1
        entry = results[0]
        assert entry["task_id"] == "task1"
        assert entry["domain"] == "os"
        assert entry["instruction"] == "Open Bluetooth"
        assert entry["attempts"]["1"] == {
            "score": 0.3, "steps_taken": 10, "skills_count": 2,
            "early_stop_triggered": True,
        }
        assert entry["attempts"]["2"] == {
            "score": 0.7, "steps_taken": 5, "skills_count": 3,
            "early_stop_triggered": False,
        }
        # And summary reports per-attempt stats.
        per_attempt = data["summary"]["per_attempt"]
        assert per_attempt["1"]["avg_score"] == 0.3
        assert per_attempt["2"]["avg_score"] == 0.7
        assert data["summary"]["max_attempt_seen"] == 2

    def test_rebuild_experiment_results_skips_task_without_task_json(self, tmp_path):
        from anything2skill.runner import _rebuild_experiment_results
        task_dir = tmp_path / "os" / "task1"
        (task_dir / "attempt_1").mkdir(parents=True)
        (task_dir / "attempt_1" / "meta.json").write_text(
            '{"score":0.3,"steps":10,"skills_count":2}'
        )
        # No task.json → skipped.

        _rebuild_experiment_results(str(tmp_path))
        data = json.loads((tmp_path / "experiment_results.json").read_text())
        assert data["results"] == []

    def test_rebuild_experiment_results_cross_bucket_merges_both(self, tmp_path):
        """Cross-bucket: canonical's attempt_1 and reviser's attempt_2 both
        appear as separate entries — no collapsing to a single best."""
        from anything2skill.runner import _rebuild_experiment_results
        canonical = tmp_path / "canonical" / "os" / "task1"
        reviser = tmp_path / "reviser" / "os" / "task1"
        (canonical / "attempt_1").mkdir(parents=True)
        (canonical / "attempt_1" / "meta.json").write_text(
            '{"score":0.9,"steps":99,"skills_count":5,'
            '"early_stop_triggered":false}'
        )
        (canonical / "task.json").write_text('{"instruction": "cross"}')
        (reviser / "attempt_2").mkdir(parents=True)
        (reviser / "attempt_2" / "meta.json").write_text(
            '{"score":0.2,"steps":4,"skills_count":1,'
            '"early_stop_triggered":false}'
        )
        (reviser / "task.json").write_text('{"instruction": "cross"}')

        _rebuild_experiment_results(
            str(tmp_path / "reviser"),
            canonical_base=str(tmp_path / "canonical"),
        )
        data = json.loads(
            (tmp_path / "reviser" / "experiment_results.json").read_text()
        )
        entry = data["results"][0]
        assert entry["instruction"] == "cross"
        assert entry["attempts"]["1"] == {
            "score": 0.9, "steps_taken": 99, "skills_count": 5,
            "early_stop_triggered": False,
        }
        assert entry["attempts"]["2"] == {
            "score": 0.2, "steps_taken": 4, "skills_count": 1,
            "early_stop_triggered": False,
        }

    def test_rebuild_experiment_results_canonical_bucket_only_attempt1(self, tmp_path):
        """Canonical-bucket rebuild (no canonical_base arg) keeps only
        attempt_1 — the 'bare baseline' view alongside the reviser JSON."""
        from anything2skill.runner import _rebuild_experiment_results
        canonical = tmp_path / "canonical" / "os" / "task1"
        (canonical / "attempt_1").mkdir(parents=True)
        (canonical / "attempt_1" / "meta.json").write_text(
            '{"score":0.4,"steps":7,"skills_count":2,'
            '"early_stop_triggered":false}'
        )
        (canonical / "task.json").write_text('{"instruction": "baseline"}')

        _rebuild_experiment_results(str(tmp_path / "canonical"))
        data = json.loads(
            (tmp_path / "canonical" / "experiment_results.json").read_text()
        )
        entry = data["results"][0]
        assert list(entry["attempts"].keys()) == ["1"]
        assert entry["attempts"]["1"]["score"] == 0.4
        assert data["summary"]["max_attempt_seen"] == 1

    def test_rebuild_experiment_results_summary_by_domain_per_attempt(self, tmp_path):
        """summary.by_domain.<d>.per_attempt reports per-attempt stats."""
        from anything2skill.runner import _rebuild_experiment_results
        # Two tasks in the same domain, two attempts each.
        for i in range(2):
            td = tmp_path / "os" / f"task{i}"
            (td / "attempt_1").mkdir(parents=True)
            (td / "attempt_1" / "meta.json").write_text(
                json.dumps({
                    "score": 0.0, "steps": 5, "skills_count": 1,
                    "early_stop_triggered": False,
                })
            )
            (td / "attempt_2").mkdir(parents=True)
            (td / "attempt_2" / "meta.json").write_text(
                json.dumps({
                    "score": 1.0 if i == 0 else 0.0,
                    "steps": 3, "skills_count": 1,
                    "early_stop_triggered": False,
                })
            )
            (td / "task.json").write_text('{"instruction": "x"}')

        _rebuild_experiment_results(str(tmp_path))
        data = json.loads((tmp_path / "experiment_results.json").read_text())
        by_domain = data["summary"]["by_domain"]["os"]
        assert by_domain["count"] == 2
        assert by_domain["per_attempt"]["1"]["success_rate"] == 0.0
        assert by_domain["per_attempt"]["2"]["success_rate"] == 0.5

    def test_rebuild_experiment_results_summary_early_stop(self, tmp_path):
        """summary.early_stop aggregates tagged-attempt vs full-run views,
        with per-domain slicing."""
        from anything2skill.runner import _rebuild_experiment_results

        def _write_task(domain: str, name: str, attempts: list[dict]) -> None:
            td = tmp_path / domain / name
            for j, payload in enumerate(attempts, start=1):
                a = td / f"attempt_{j}"
                a.mkdir(parents=True)
                (a / "meta.json").write_text(json.dumps(payload))
            (td / "task.json").write_text('{"instruction":"x"}')

        # os/task_a: tagged at attempt_2 (score 1.0 there, full-run = attempt_3 = 0.5).
        _write_task("os", "task_a", [
            {"score": 0.0, "steps": 4, "skills_count": 1, "early_stop_triggered": False},
            {"score": 1.0, "steps": 3, "skills_count": 1, "early_stop_triggered": True},
            {"score": 0.5, "steps": 2, "skills_count": 1, "early_stop_triggered": False},
        ])
        # os/task_b: never tagged. full-run = attempt_1 = 0.0.
        _write_task("os", "task_b", [
            {"score": 0.0, "steps": 6, "skills_count": 1, "early_stop_triggered": False},
        ])
        # mc/task_c: tagged at attempt_1 (score 1.0, also the only attempt → both views the same).
        _write_task("mc", "task_c", [
            {"score": 1.0, "steps": 1, "skills_count": 1, "early_stop_triggered": True},
        ])

        _rebuild_experiment_results(str(tmp_path))
        data = json.loads((tmp_path / "experiment_results.json").read_text())
        es = data["summary"]["early_stop"]

        assert es["tagged_tasks"] == 2          # task_a + task_c
        assert es["tagged_rate"] == 0.6667      # 2 / 3
        assert es["by_attempt"] == {"1": 1, "2": 1}

        # early_stop_view = tagged-attempt score where present, last-attempt
        # score where absent: task_a=1.0 (tagged a2), task_b=0.0 (no tag → last),
        # task_c=1.0 (tagged a1).
        assert es["early_stop_view"]["count"] == 3
        assert es["early_stop_view"]["success_rate"] == round(2 / 3, 4)
        assert es["early_stop_view"]["avg_score"] == round(2 / 3, 4)
        # full_run_view = last-attempt scores: [0.5, 0.0, 1.0]
        assert es["full_run_view"]["count"] == 3
        assert es["full_run_view"]["success_rate"] == round(2 / 3, 4)
        assert es["full_run_view"]["avg_score"] == 0.5

        # by_domain slicing.
        os_block = es["by_domain"]["os"]
        assert os_block["tagged_tasks"] == 1
        assert os_block["tagged_rate"] == 0.5
        assert os_block["by_attempt"] == {"2": 1}
        # os early_stop_view = [1.0 (task_a tagged), 0.0 (task_b last)]
        assert os_block["early_stop_view"]["count"] == 2
        assert os_block["early_stop_view"]["avg_score"] == 0.5
        # os full_run = [0.5, 0.0]
        assert os_block["full_run_view"]["avg_score"] == 0.25

        mc_block = es["by_domain"]["mc"]
        assert mc_block["tagged_tasks"] == 1
        assert mc_block["tagged_rate"] == 1.0
        assert mc_block["by_attempt"] == {"1": 1}
        assert mc_block["early_stop_view"]["count"] == 1
        assert mc_block["early_stop_view"]["avg_score"] == 1.0
        assert mc_block["full_run_view"]["avg_score"] == 1.0


class TestTaskInProgressLock:
    def test_acquire_writes_pid_and_release_unlinks(self, tmp_path):
        from anything2skill.runner import (
            _release_task_lock, _try_acquire_task_lock,
        )
        lock = _try_acquire_task_lock(tmp_path)
        assert lock is not None and lock.exists()
        assert lock.read_text(encoding="utf-8").strip() == str(os.getpid())
        _release_task_lock(lock)
        assert not lock.exists()

    def test_acquire_returns_none_when_live_pid_holds_lock(self, tmp_path):
        from anything2skill.runner import _try_acquire_task_lock
        lock_path = tmp_path / ".in_progress.lock"
        lock_path.write_text(f"{os.getpid()}\n")  # current process is alive
        assert _try_acquire_task_lock(tmp_path) is None
        # original lock content untouched
        assert lock_path.read_text().strip() == str(os.getpid())

    def test_acquire_evicts_stale_lock_from_dead_pid(self, tmp_path, caplog):
        """Worker crash leaves a lock with a dead PID — acquire must reclaim."""
        from anything2skill.runner import _try_acquire_task_lock
        # Pick a PID overwhelmingly unlikely to exist (well above default
        # Linux pid_max of 32k / 4M).
        dead_pid = 4_194_304
        lock_path = tmp_path / ".in_progress.lock"
        lock_path.write_text(f"{dead_pid}\n")

        with caplog.at_level(logging.WARNING, logger="anything2skill.runner"):
            lock = _try_acquire_task_lock(tmp_path)

        assert lock is not None
        assert lock.read_text(encoding="utf-8").strip() == str(os.getpid())
        assert any(
            "Removed stale lock" in r.message for r in caplog.records
        )

    def test_acquire_treats_malformed_lock_as_stale(self, tmp_path):
        """A pre-PID lockfile (empty / non-digit) is reclaimable."""
        from anything2skill.runner import _try_acquire_task_lock
        lock_path = tmp_path / ".in_progress.lock"
        lock_path.write_text("")  # legacy lock from before PID was written
        lock = _try_acquire_task_lock(tmp_path)
        assert lock is not None
        assert lock.read_text(encoding="utf-8").strip() == str(os.getpid())

