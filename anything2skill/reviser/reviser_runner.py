"""ReviserRunner: dual-bucket attempt orchestrator + phase-1/2 dispatch.

Responsibilities (high level):

- Attempt loop over ``1..max_attempts``.
- Resume: scan existing ``attempt_*`` dirs (across the canonical +
  reviser bucket split, see §5 of the plan) and continue from the first
  incomplete attempt.
- Clean up **only execution artifacts** (``traj.jsonl``, ``step_*``,
  ``result.txt``, ``response.log``, ``reviser_chunks/``) before a fresh
  run of attempt j; preserve ``skills/``, ``root_cause.xml``,
  ``meta.json``.
- Protect the canonical ``attempt_1`` from being cleaned when resuming
  into a later attempt (``start_from > 1``).
- Lazily construct the reviser VLMClient — when ``max_attempts == 1``,
  no phase-1/2 calls ever happen so we don't pay the constructor cost.
- Write ``attempt_N/meta.json`` after every attempt and
  ``task.json`` at the task-level for experiment tracking. The
  ``experiment_results.json`` under the bucket root is refreshed
  immediately after each attempt's meta is persisted.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from anything2skill.reviser.data_types import RootCauseAnalysis
from anything2skill.reviser.skills_io import (
    _atomic_write_text,
    load_skills_from_dir,
    save_skills_to_dir,
)

if TYPE_CHECKING:
    from anything2skill.agent.base import BaseAgent
    from anything2skill.benchmark_kit import BenchmarkKit, TaskDescriptor
    from anything2skill.env_base import EnvironmentInterface
    from anything2skill.parser.data_types import Skills, TutorialMaterial
    from anything2skill.reviser.analyzer import ReviserAnalyzer
    from anything2skill.reviser.refiner import ReviserRefiner
    from anything2skill.vlm.client import VLMClient

logger = logging.getLogger("anything2skill.reviser")


# File names used for execution vs. persistent artifacts. Execution
# artifacts are cleaned on retry; persistent artifacts survive.
_EXEC_FILES = ("traj.jsonl", "result.txt", "response.log", "runtime.log")
_EXEC_DIRS = ("reviser_chunks",)


_EARLY_STOP_SIGNALS = ("no_issue", "likely_success")


def _opt_float(v) -> float | None:
    """Coerce to float, preserving None/null for reasoning-endpoint compat."""
    return None if v is None else float(v)


def _normalize_early_stop_signal(value) -> str:
    if value is None:
        return "no_issue"
    signal = str(value).strip()
    if signal not in _EARLY_STOP_SIGNALS:
        allowed = ", ".join(_EARLY_STOP_SIGNALS)
        raise ValueError(
            f"Unsupported reviser.early_stop_signal={value!r}; expected one of: {allowed}"
        )
    return signal


def _matches_early_stop_signal(rc: RootCauseAnalysis, signal: str) -> bool:
    if signal == "no_issue":
        return not rc.issues
    if signal == "likely_success":
        outcome = (rc.outcome_assessment or "").strip().lower()
        return outcome == "likely_success"
    raise ValueError(f"Unsupported early-stop signal: {signal!r}")


class ReviserRunner:
    """Dual-bucket attempt orchestrator.

    Construction is cheap: no VLMClient is instantiated until the first
    attempt that actually needs phase-1/2. Callers can therefore pass a
    zero-cost factory for ``vlm_factory``.
    """

    def __init__(
        self,
        kit: BenchmarkKit,
        reviser_cfg: dict | None = None,
        vlm_factory: Callable[[], VLMClient] | None = None,
        canonical_result_base: str | os.PathLike | None = None,
        reviser_result_base: str | os.PathLike | None = None,
    ):
        self.kit = kit
        self.cfg = dict(reviser_cfg or {})
        self._early_stop_signal = _normalize_early_stop_signal(
            self.cfg.get("early_stop_signal", "no_issue")
        )
        self._vlm_factory = vlm_factory
        self._vlm: VLMClient | None = None
        self._analyzer: ReviserAnalyzer | None = None
        self._refiner: ReviserRefiner | None = None
        # Bucket roots used to rebuild experiment_results.json after every
        # attempt. Optional so unit tests can construct a Runner without
        # plumbing full paths; when unset, immediate rebuild is skipped.
        self._canonical_result_base = (
            str(canonical_result_base) if canonical_result_base else None
        )
        self._reviser_result_base = (
            str(reviser_result_base) if reviser_result_base else None
        )

    # ------------------------------------------------------------------
    # Lazy VLM construction
    # ------------------------------------------------------------------

    def _get_analyzer(self) -> ReviserAnalyzer:
        from anything2skill.reviser.analyzer import ReviserAnalyzer

        if self._analyzer is None:
            self._analyzer = ReviserAnalyzer(
                vlm=self._get_vlm(),
                kit=self.kit,
                chunk_size=int(self.cfg.get("chunk_size", 15)),
                max_tokens=int(self.cfg.get("analysis_max_tokens", 2000)),
                temperature=_opt_float(self.cfg.get("analysis_temperature", 0.1)),
                response_char_limit=int(self.cfg.get("response_char_limit", 4000)),
                rolling_summary_char_limit=int(
                    self.cfg.get("rolling_summary_char_limit", 1500)
                ),
            )
        return self._analyzer

    def _get_refiner(self) -> ReviserRefiner:
        from anything2skill.reviser.refiner import ReviserRefiner

        if self._refiner is None:
            self._refiner = ReviserRefiner(
                vlm=self._get_vlm(),
                kit=self.kit,
                max_tokens=int(self.cfg.get("refine_max_tokens", 4000)),
                temperature=_opt_float(self.cfg.get("refine_temperature", 0.1)),
                tutorial_image_cap=int(self.cfg.get("tutorial_image_cap", 8)),
                include_tutorial_in_refine=bool(
                    self.cfg.get("include_tutorial_in_refine", True)
                ),
            )
        return self._refiner

    def _get_vlm(self) -> VLMClient:
        if self._vlm is None:
            if self._vlm_factory is None:
                raise RuntimeError(
                    "ReviserRunner: vlm_factory is required once phase-1/2 fires"
                )
            self._vlm = self._vlm_factory()
        return self._vlm

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_with_reviser(
        self,
        agent_factory: Callable[[Skills], BaseAgent],
        env: EnvironmentInterface,
        task: TaskDescriptor,
        skills: Skills,
        tutorial: TutorialMaterial | None,
        max_steps: int,
        sleep_after_execution: float,
        canonical_task_dir: str | os.PathLike,
        reviser_task_dir: str | os.PathLike,
        max_attempts: int,
    ) -> tuple[float, int]:
        """Run attempts 1..max_attempts, refining between them.

        Returns the ``(score, steps_taken)`` of the most recently
        completed attempt (either the last one this call ran, or the
        latest already on disk if nothing else needed to run). Each
        attempt is recorded separately in ``experiment_results.json``.
        """
        from anything2skill.runner import run_single_task

        canonical_dir = Path(canonical_task_dir)
        reviser_dir = Path(reviser_task_dir)
        canonical_dir.mkdir(parents=True, exist_ok=True)
        reviser_dir.mkdir(parents=True, exist_ok=True)

        def attempt_dir(j: int) -> Path:
            return (canonical_dir if j == 1 else reviser_dir) / f"attempt_{j}"

        # Resume planning
        scanned = scan_attempts(canonical_dir, reviser_dir)
        last_complete = max(
            (n for n, meta in scanned.items() if meta["completed"]),
            default=0,
        )
        start_from = last_complete + 1

        # Delay task.json writes until *after* an attempt's meta.json
        # lands.  Writing task.json before the attempt completes creates
        # a window where other workers' _rebuild_experiment_results sees
        # a dir with task.json but no attempt_*/meta.json and logs a
        # spurious warning.  The only exception is the "already
        # satisfied" early-return, where data is already on disk.

        if start_from > max_attempts:
            logger.info(
                "[Reviser] %s already satisfied (%d/%d attempts complete)",
                task.task_id, last_complete, max_attempts,
            )
            _ensure_task_json(canonical_dir, task.instruction)
            if reviser_dir != canonical_dir:
                _ensure_task_json(reviser_dir, task.instruction)
            last_score, last_steps = _latest_attempt_scores(scanned)
            self._refresh_experiment_results()
            return last_score, last_steps

        if last_complete == 0:
            logger.info(
                "[Reviser] %s fresh start attempts 1..%d",
                task.task_id, max_attempts,
            )
        else:
            logger.info(
                "[Reviser] %s resume from attempt_%d/%d",
                task.task_id, start_from, max_attempts,
            )

        # Recover the skills we'll use for the next attempt.
        current_skills = self._recover_skills_for_attempt(
            start_from,
            canonical_dir=canonical_dir,
            reviser_dir=reviser_dir,
            original_skills=skills,
            tutorial=tutorial,
            instruction=task.instruction,
        )

        last_score, last_steps = _latest_attempt_scores(scanned)

        # Recompute after recovery — it may have backfilled a tag onto a
        # prior attempt whose flag write was lost to a crash.
        already_marked = any(
            meta["early_stop_triggered"]
            for meta in scan_attempts(canonical_dir, reviser_dir).values()
            if meta["completed"]
        )

        for j in range(start_from, max_attempts + 1):
            a_dir = attempt_dir(j)
            a_dir.mkdir(parents=True, exist_ok=True)

            # Canonical attempt_1 is read-only once we're resuming past it.
            if j == 1 and start_from == 1:
                _cleanup_runtime_artifacts(a_dir)
            elif j != 1:
                _cleanup_runtime_artifacts(a_dir)

            save_skills_to_dir(current_skills, a_dir / "skills")

            agent = agent_factory(current_skills)
            agent.result_dir = str(a_dir)

            score, steps = run_single_task(
                agent=agent,
                env=env,
                task=task,
                max_steps=max_steps,
                sleep_after_execution=sleep_after_execution,
                result_dir=str(a_dir),
            )

            last_score, last_steps = score, steps

            _save_attempt_meta(
                a_dir, score=score, steps=steps,
                skills_count=len(current_skills.skills),
                early_stop_triggered=False,
            )
            _ensure_task_json(canonical_dir, task.instruction)
            if reviser_dir != canonical_dir:
                _ensure_task_json(reviser_dir, task.instruction)
            # Immediate refresh so experiment_results.json reflects this
            # attempt the moment its meta.json lands.
            self._refresh_experiment_results()

            if j >= max_attempts:
                break

            # Phase 1: analyze traj for attempt j
            traj_path = a_dir / "traj.jsonl"
            if not traj_path.is_file():
                logger.warning(
                    "[Reviser] No traj.jsonl for attempt_%d; stopping reviser loop",
                    j,
                )
                break

            # Audit dumps belong to the attempt we're *producing* (j+1),
            # not the one we're reading from (j). This keeps canonical
            # attempt_1 read-only when a cross-bucket reviser inspects it.
            next_dir = attempt_dir(j + 1)
            next_dir.mkdir(parents=True, exist_ok=True)

            root_cause = self._get_analyzer().analyze(
                traj_path=str(traj_path),
                instruction=task.instruction,
                result_dir=str(a_dir),
                audit_dir=str(next_dir),
            )

            _save_root_cause(next_dir, root_cause)

            if (
                _matches_early_stop_signal(root_cause, self._early_stop_signal)
                and not already_marked
            ):
                logger.info(
                    "[Reviser] First early-stop signal at attempt_%d "
                    "(signal=%s, issues=%d, outcome=%s); "
                    "tagging and force-continuing to max_attempts",
                    j,
                    self._early_stop_signal,
                    len(root_cause.issues),
                    root_cause.outcome_assessment or "unset",
                )
                _update_attempt_meta(a_dir, early_stop_triggered=True)
                already_marked = True
                self._refresh_experiment_results()

            # Phase 2: refine
            if tutorial is None:
                logger.info(
                    "[Reviser] tutorial=None; cannot refine (likely vanilla). Stopping."
                )
                break

            history = self._collect_history(
                target_attempt=j + 1,
                canonical_dir=canonical_dir,
                reviser_dir=reviser_dir,
            )

            current_skills = self._get_refiner().refine(
                skills=current_skills,
                root_cause=root_cause,
                tutorial=tutorial,
                instruction=task.instruction,
                history=history,
            )

        return last_score, last_steps

    # ------------------------------------------------------------------
    # experiment_results.json refresh
    # ------------------------------------------------------------------

    def _refresh_experiment_results(self) -> None:
        """Rebuild experiment_results.json in each configured bucket.

        No-op when bucket roots were not plumbed through (unit tests
        often skip this to avoid touching disk outside the tmp dirs).
        """
        if not self._reviser_result_base:
            return
        from anything2skill.runner import _rebuild_experiment_results

        _rebuild_experiment_results(
            self._reviser_result_base, self._canonical_result_base,
        )
        if (
            self._canonical_result_base
            and self._canonical_result_base != self._reviser_result_base
        ):
            _rebuild_experiment_results(self._canonical_result_base)

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def _recover_skills_for_attempt(
        self,
        j: int,
        *,
        canonical_dir: Path,
        reviser_dir: Path,
        original_skills: Skills,
        tutorial: TutorialMaterial | None,
        instruction: str,
    ) -> Skills:
        """Produce the Skills to run attempt *j* with.

        Priority:
        1. ``attempt_j/skills/`` already on disk → use it.
        2. ``attempt_j/root_cause.xml`` exists (but skills missing) →
           skip analyze, run refine directly.
        3. Otherwise: read ``attempt_{j-1}/traj.jsonl`` from the right
           bucket, run analyze + refine, then persist.

        If the rc consumed on this path matches the configured early-stop
        signal and no prior attempt is already tagged, backfill
        ``early_stop_triggered=True`` onto ``attempt_{j-1}/meta.json`` — the
        previous run crashed between saving the attempt's meta and flipping
        the flag.
        """
        if j == 1:
            return original_skills

        attempt_j_dir = (canonical_dir if j == 1 else reviser_dir) / f"attempt_{j}"
        existing_skills_dir = attempt_j_dir / "skills"
        existing_rc = attempt_j_dir / "root_cause.xml"

        loaded = load_skills_from_dir(
            existing_skills_dir,
            task_id=original_skills.task_id,
            instruction=instruction,
            image_dir=original_skills.image_dir,
        )
        if loaded is not None and loaded.skills:
            logger.info(
                "[Reviser] reusing refined skills for attempt_%d from disk", j,
            )
            return loaded

        # 2) root_cause.xml present but no skills
        prev_skills = self._load_prev_attempt_skills(
            j - 1,
            canonical_dir=canonical_dir,
            reviser_dir=reviser_dir,
            original_skills=original_skills,
            instruction=instruction,
        )

        history = self._collect_history(
            target_attempt=j,
            canonical_dir=canonical_dir,
            reviser_dir=reviser_dir,
        )

        prev_dir = (canonical_dir if j - 1 == 1 else reviser_dir) / f"attempt_{j - 1}"

        if existing_rc.is_file() and tutorial is not None:
            logger.info(
                "[Reviser] reusing root_cause for attempt_%d; only running refine", j,
            )
            rc = _load_root_cause(existing_rc)
            self._backfill_early_stop_if_first(
                rc, prev_dir,
                canonical_dir=canonical_dir, reviser_dir=reviser_dir,
            )
            return self._get_refiner().refine(
                skills=prev_skills,
                root_cause=rc,
                tutorial=tutorial,
                instruction=instruction,
                history=history,
            )

        # 3) full analyze + refine using the previous attempt's trajectory.
        prev_traj = prev_dir / "traj.jsonl"
        if not prev_traj.is_file():
            logger.warning(
                "[Reviser] prev traj %s missing; falling back to original skills",
                prev_traj,
            )
            return prev_skills

        attempt_j_dir.mkdir(parents=True, exist_ok=True)

        rc = self._get_analyzer().analyze(
            traj_path=str(prev_traj),
            instruction=instruction,
            result_dir=str(prev_dir),
            audit_dir=str(attempt_j_dir),
        )
        _save_root_cause(attempt_j_dir, rc)
        self._backfill_early_stop_if_first(
            rc, prev_dir,
            canonical_dir=canonical_dir, reviser_dir=reviser_dir,
        )

        if tutorial is None:
            logger.info("[Reviser] tutorial=None; skipping refine during recovery")
            return prev_skills

        return self._get_refiner().refine(
            skills=prev_skills,
            root_cause=rc,
            tutorial=tutorial,
            instruction=instruction,
            history=history,
        )

    def _backfill_early_stop_if_first(
        self,
        rc: RootCauseAnalysis,
        prev_attempt_dir: Path,
        *,
        canonical_dir: Path,
        reviser_dir: Path,
    ) -> None:
        """Tag ``prev_attempt_dir/meta.json`` on the first early-stop signal.

        Covers the resume window where the original run wrote the attempt's
        meta (``early_stop_triggered=False``) and then crashed before
        flipping it — without backfill the main loop below would re-tag a
        later attempt instead.
        """
        if not _matches_early_stop_signal(rc, self._early_stop_signal):
            return
        if not (prev_attempt_dir / "meta.json").is_file():
            return
        scanned = scan_attempts(canonical_dir, reviser_dir)
        if any(
            meta["early_stop_triggered"]
            for meta in scanned.values()
            if meta["completed"]
        ):
            return
        logger.info(
            "[Reviser] Recovery backfill: tagging %s as first early-stop",
            prev_attempt_dir,
        )
        _update_attempt_meta(prev_attempt_dir, early_stop_triggered=True)

    # ------------------------------------------------------------------
    # History collection
    # ------------------------------------------------------------------

    def _collect_history(
        self,
        target_attempt: int,
        *,
        canonical_dir: Path,
        reviser_dir: Path,
    ) -> list[RootCauseAnalysis]:
        """Return root_cause XMLs from attempt_2..attempt_{target_attempt-1}.

        When refining to produce ``attempt_{target_attempt}``, these are the
        prior-round root_causes the refiner should see so it doesn't undo
        earlier fixes. attempt_1 has no root_cause (nothing preceded it),
        and the current attempt's root_cause is passed separately.
        """
        if not bool(self.cfg.get("history_in_refine", True)):
            return []
        if target_attempt <= 2:
            return []
        # attempt_1 has no root_cause.xml (nothing preceded it), so the
        # loop starts at 2 and every path lives in the reviser bucket.
        out: list[RootCauseAnalysis] = []
        for k in range(2, target_attempt):
            rc_path = reviser_dir / f"attempt_{k}" / "root_cause.xml"
            if not rc_path.is_file():
                continue
            try:
                out.append(_load_root_cause(rc_path))
            except Exception as e:
                logger.warning(
                    "[Reviser] failed to load history rc %s: %s", rc_path, e,
                )
        return out

    @staticmethod
    def _load_prev_attempt_skills(
        prev_j: int,
        *,
        canonical_dir: Path,
        reviser_dir: Path,
        original_skills: Skills,
        instruction: str,
    ) -> Skills:
        """Load the skills used for completed attempt *prev_j*."""
        base = canonical_dir if prev_j == 1 else reviser_dir
        loaded = load_skills_from_dir(
            base / f"attempt_{prev_j}" / "skills",
            task_id=original_skills.task_id,
            instruction=instruction,
            image_dir=original_skills.image_dir,
        )
        if loaded is not None and loaded.skills:
            return loaded
        logger.warning(
            "[Reviser] attempt_%d/skills missing; reusing original skills",
            prev_j,
        )
        return original_skills


# ── Attempt scanning + persistence helpers ──────────────────────────────


def scan_attempts(
    canonical_task_dir: str | Path,
    reviser_task_dir: str | Path,
) -> dict[int, dict]:
    """Scan both buckets for ``attempt_N`` directories.

    Returns ``{n: {"dir": Path, "completed": bool, "has_skills": bool,
    "has_root_cause": bool, "has_traj": bool, "has_meta": bool,
    "early_stop_triggered": bool}}``. The early-stop flag is read from
    persisted meta, not inferred from root_cause.xml.
    """
    canonical_dir = Path(canonical_task_dir)
    reviser_dir = Path(reviser_task_dir)
    out: dict[int, dict] = {}

    def _scan_one(p: Path) -> dict:
        meta = _load_attempt_meta(p)
        has_meta = meta is not None
        return {
            "dir": p,
            "has_skills": (p / "skills").is_dir()
            and any((p / "skills").iterdir())
            if (p / "skills").is_dir()
            else False,
            "has_root_cause": (p / "root_cause.xml").is_file(),
            "has_traj": (p / "traj.jsonl").is_file(),
            "has_meta": has_meta,
            "completed": (p / "result.txt").is_file() and has_meta,
            "early_stop_triggered": (
                bool(meta["early_stop_triggered"]) if has_meta else False
            ),
        }

    a1 = canonical_dir / "attempt_1"
    if a1.is_dir():
        out[1] = _scan_one(a1)

    if reviser_dir.is_dir():
        for entry in reviser_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("attempt_"):
                continue
            try:
                n = int(entry.name.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            if n <= 1 and reviser_dir != canonical_dir:
                # attempt_1 is canonical-only when buckets differ.
                continue
            out[n] = _scan_one(entry)

    return out


def _cleanup_runtime_artifacts(attempt_dir: Path) -> None:
    """Remove execution outputs while preserving persistent artifacts."""
    if not attempt_dir.is_dir():
        return

    for name in _EXEC_FILES:
        p = attempt_dir / name
        if p.is_file():
            p.unlink()

    for name in _EXEC_DIRS:
        p = attempt_dir / name
        if p.is_dir():
            shutil.rmtree(p)

    # step_*_* dirs are execution products too
    for child in attempt_dir.iterdir():
        if child.is_dir() and child.name.startswith("step_"):
            shutil.rmtree(child)


def _load_attempt_meta(attempt_dir: Path) -> dict | None:
    """Return parsed ``meta.json`` payload, or ``None`` if missing/corrupt."""
    path = attempt_dir / "meta.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save_attempt_meta(
    attempt_dir: Path,
    *,
    score: float,
    steps: int,
    skills_count: int,
    early_stop_triggered: bool,
) -> None:
    """Write ``meta.json`` atomically."""
    payload = {
        "score": float(score),
        "steps": int(steps),
        "skills_count": int(skills_count),
        "early_stop_triggered": bool(early_stop_triggered),
    }
    _atomic_write_text(
        attempt_dir / "meta.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _update_attempt_meta(attempt_dir: Path, **fields) -> None:
    """Merge ``fields`` into existing ``meta.json`` atomically."""
    payload = _load_attempt_meta(attempt_dir)
    if payload is None:
        raise FileNotFoundError(f"{attempt_dir / 'meta.json'} missing")
    payload.update(fields)
    _atomic_write_text(
        attempt_dir / "meta.json",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _save_root_cause(attempt_dir: Path, rc: RootCauseAnalysis) -> None:
    """Persist the raw XML (or a reconstruction) alongside the attempt dir."""
    from anything2skill.reviser.refiner import _serialize_root_cause_fallback

    xml = rc.raw_xml.strip() or _serialize_root_cause_fallback(rc)
    _atomic_write_text(attempt_dir / "root_cause.xml", xml)


def _load_root_cause(path: Path) -> RootCauseAnalysis:
    from anything2skill.reviser.analyzer import _parse_root_cause

    text = path.read_text(encoding="utf-8")
    return _parse_root_cause(text)


def _ensure_task_json(task_dir: Path, instruction: str) -> None:
    """Write ``task.json`` once — ignored on subsequent calls."""
    p = task_dir / "task.json"
    if p.is_file():
        return
    _atomic_write_text(
        p, json.dumps({"instruction": instruction}, ensure_ascii=False, indent=2)
    )


def _latest_attempt_scores(scanned: dict[int, dict]) -> tuple[float, int]:
    """Return (score, steps) of the highest-numbered completed attempt."""
    completed = [n for n, m in scanned.items() if m["completed"]]
    if not completed:
        return 0.0, 0
    n = max(completed)
    payload = json.loads(
        (scanned[n]["dir"] / "meta.json").read_text(encoding="utf-8"),
    )
    return float(payload.get("score", 0.0)), int(payload.get("steps", 0))
