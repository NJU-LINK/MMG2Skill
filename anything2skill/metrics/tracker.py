"""Result tracking and aggregation for Anything2Skill experiments.

Per-attempt schema: every task carries an ``attempts`` dict keyed by the
attempt number as a string. Each attempt holds its own
``score`` / ``steps_taken`` / ``skills_count``. The ``summary`` section
aggregates per attempt so round-1 vs round-N comparison is readable from
the JSON directly.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("anything2skill.metrics")


@dataclass
class AttemptResult:
    """Per-attempt metrics for a single task."""

    score: float
    steps_taken: int = 0
    skills_count: int = 0
    early_stop_triggered: bool = False


@dataclass
class TaskResult:
    """Result for a single task across (possibly) multiple attempts."""

    task_id: str
    domain: str
    instruction: str = ""
    attempts: dict[str, AttemptResult] = field(default_factory=dict)


@dataclass
class ExperimentTracker:
    """Tracks results across multiple tasks and attempts."""

    results: dict[str, TaskResult] = field(default_factory=dict)

    def add_attempt(
        self,
        *,
        task_id: str,
        domain: str,
        instruction: str,
        attempt_n: int,
        score: float,
        steps_taken: int,
        skills_count: int,
        early_stop_triggered: bool,
    ) -> None:
        """Upsert a single attempt's data for a task."""
        task = self.results.get(task_id)
        if task is None:
            task = TaskResult(
                task_id=task_id, domain=domain, instruction=instruction,
            )
            self.results[task_id] = task
        else:
            # Keep the latest non-empty instruction / domain.
            if instruction and not task.instruction:
                task.instruction = instruction
            if domain and not task.domain:
                task.domain = domain
        task.attempts[str(int(attempt_n))] = AttemptResult(
            score=float(score),
            steps_taken=int(steps_taken),
            skills_count=int(skills_count),
            early_stop_triggered=bool(early_stop_triggered),
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _all_attempt_keys(self) -> list[str]:
        keys = set()
        for t in self.results.values():
            keys.update(t.attempts.keys())
        return sorted(keys, key=lambda s: int(s))

    def _per_attempt_global(self) -> dict:
        out: dict[str, dict] = {}
        for k in self._all_attempt_keys():
            attempts = [
                t.attempts[k] for t in self.results.values() if k in t.attempts
            ]
            if not attempts:
                continue
            n = len(attempts)
            out[k] = {
                "count": n,
                "success_rate": round(
                    sum(1 for a in attempts if a.score > 0) / n, 4,
                ),
                "avg_score": round(sum(a.score for a in attempts) / n, 4),
                "avg_steps": round(sum(a.steps_taken for a in attempts) / n, 2),
            }
        return out

    def _by_domain(self) -> dict:
        domains: dict[str, list[TaskResult]] = {}
        for t in self.results.values():
            domains.setdefault(t.domain, []).append(t)

        out: dict[str, dict] = {}
        for domain, tasks in domains.items():
            per_attempt: dict[str, dict] = {}
            keys = sorted(
                {k for t in tasks for k in t.attempts.keys()},
                key=lambda s: int(s),
            )
            for k in keys:
                attempts = [t.attempts[k] for t in tasks if k in t.attempts]
                if not attempts:
                    continue
                n = len(attempts)
                per_attempt[k] = {
                    "count": n,
                    "success_rate": round(
                        sum(1 for a in attempts if a.score > 0) / n, 4,
                    ),
                    "avg_score": round(
                        sum(a.score for a in attempts) / n, 4,
                    ),
                }
            out[domain] = {"count": len(tasks), "per_attempt": per_attempt}
        return out

    def _early_stop_block(self, tasks: list[TaskResult]) -> dict:
        """Aggregate early-stop info over a task subset.

        ``early_stop_view`` takes the score of the tagged attempt when one
        exists, falling back to the last attempt otherwise — matching what
        the old break-on-empty-issues policy would have returned.
        ``full_run_view`` always takes the last attempt (greedy mode's
        final answer). Both views share the same task set so counts and
        success rates are directly comparable.
        """
        total = len(tasks)
        tagged_pairs: list[tuple[TaskResult, str]] = []
        by_attempt: dict[str, int] = {}
        tagged_keys: dict[str, str] = {}
        for t in tasks:
            for k, a in t.attempts.items():
                if a.early_stop_triggered:
                    tagged_pairs.append((t, k))
                    by_attempt[k] = by_attempt.get(k, 0) + 1
                    tagged_keys[t.task_id] = k
                    break

        def _stat(records: list[AttemptResult]) -> dict:
            n = len(records)
            if n == 0:
                return {
                    "count": 0, "success_rate": 0.0,
                    "avg_score": 0.0, "avg_steps": 0.0,
                }
            return {
                "count": n,
                "success_rate": round(
                    sum(1 for a in records if a.score > 0) / n, 4,
                ),
                "avg_score": round(sum(a.score for a in records) / n, 4),
                "avg_steps": round(sum(a.steps_taken for a in records) / n, 2),
            }

        early_stop_records: list[AttemptResult] = []
        full_run_records: list[AttemptResult] = []
        for t in tasks:
            if not t.attempts:
                continue
            last_k = max(t.attempts.keys(), key=lambda s: int(s))
            full_run_records.append(t.attempts[last_k])
            tagged_k = tagged_keys.get(t.task_id, last_k)
            early_stop_records.append(t.attempts[tagged_k])

        return {
            "tagged_tasks": len(tagged_pairs),
            "tagged_rate": round(len(tagged_pairs) / total, 4) if total else 0.0,
            "by_attempt": dict(sorted(by_attempt.items(), key=lambda kv: int(kv[0]))),
            "early_stop_view": _stat(early_stop_records),
            "full_run_view": _stat(full_run_records),
        }

    def _early_stop_summary(self) -> dict:
        block = self._early_stop_block(list(self.results.values()))
        domains: dict[str, list[TaskResult]] = {}
        for t in self.results.values():
            domains.setdefault(t.domain, []).append(t)
        block["by_domain"] = {
            domain: self._early_stop_block(tasks)
            for domain, tasks in domains.items()
        }
        return block

    def summary(self) -> dict:
        keys = self._all_attempt_keys()
        return {
            "total_tasks": len(self.results),
            "max_attempt_seen": int(keys[-1]) if keys else 0,
            "per_attempt": self._per_attempt_global(),
            "by_domain": self._by_domain(),
            "early_stop": self._early_stop_summary(),
        }

    def save(self, path: str):
        """Save results to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "results": [
                {
                    "task_id": t.task_id,
                    "domain": t.domain,
                    "instruction": t.instruction,
                    "attempts": {
                        k: {
                            "score": t.attempts[k].score,
                            "steps_taken": t.attempts[k].steps_taken,
                            "skills_count": t.attempts[k].skills_count,
                            "early_stop_triggered":
                                t.attempts[k].early_stop_triggered,
                        }
                        for k in sorted(t.attempts.keys(), key=lambda s: int(s))
                    },
                }
                for t in self.results.values()
            ],
        }
        # Atomic write so N parallel workers can rebuild concurrently
        # without readers ever seeing a torn / half-written JSON file.
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        fd, tmp = tempfile.mkstemp(
            prefix=".experiment_results.", suffix=".json", dir=str(out.parent),
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, out)
        logger.info("Saved experiment results to %s", out)
