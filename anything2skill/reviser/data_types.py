"""Data types for the reviser module."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RootCauseAnalysis:
    """Result of reviser phase-1 trajectory analysis.

    Mirrors the fields of the ``<root_cause>`` XML produced by the VLM. The
    raw XML is preserved verbatim so the refiner can paste it back into its
    prompt without round-tripping through a different serialization.
    """

    # Phase-1 final rolling summary. Also carries the "overall behaviour"
    # description — a separate <pattern> field turned out to duplicate it.
    trajectory_summary: str = ""
    # Local steps the analyzer judged correct. Refiner must preserve these
    # when outcome_assessment is likely_success.
    what_worked: list[str] = field(default_factory=list)
    # Each issue is a dict of {"where", "evidence", "cause"}. No skill name
    # — the refiner decides which skill to adjust.
    issues: list[dict] = field(default_factory=list)
    # Self-assessed outcome: "likely_success" | "uncertain" | "likely_failure"
    # | "". Used by the refiner to decide how aggressive to edit.
    outcome_assessment: str = ""
    outcome_rationale: str = ""
    raw_xml: str = ""
    # Preserved LLM response text for debugging when outer <root_cause>
    # block could not be isolated. Never injected into refiner prompts.
    raw_response: str = ""
