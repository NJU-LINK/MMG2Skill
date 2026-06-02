"""Reviser module: two-phase trajectory analysis + tutorial-aware skill refinement.

Phase 1 (:class:`anything2skill.reviser.analyzer.ReviserAnalyzer`):
    Roll through the trajectory in chunks of N predict-turns. Every chunk
    injects the detailed content (response, action(s), saved observation)
    of that chunk plus a rolling summary of all earlier chunks. The final
    chunk produces a ``<root_cause>`` XML analysis.

Phase 2 (:class:`anything2skill.reviser.refiner.ReviserRefiner`):
    Takes the raw ``<root_cause>`` XML, the current skills, and the full
    tutorial (body + images) and asks the VLM to produce improved skills.

:class:`anything2skill.reviser.reviser_runner.ReviserRunner` orchestrates
multiple attempts with a dual-bucket layout (attempt_1 shared across
reviser models, attempt_2+ isolated per reviser model).
"""

from __future__ import annotations

from anything2skill.reviser.analyzer import ReviserAnalyzer
from anything2skill.reviser.data_types import RootCauseAnalysis
from anything2skill.reviser.refiner import ReviserRefiner
from anything2skill.reviser.reviser_runner import ReviserRunner

__all__ = [
    "ReviserAnalyzer",
    "ReviserRefiner",
    "ReviserRunner",
    "RootCauseAnalysis",
]
