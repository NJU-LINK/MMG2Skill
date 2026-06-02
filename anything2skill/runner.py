"""Core task runner for all benchmarks.

Provides:
- :func:`run_single_task` — benchmark-agnostic execution loop (writes
  ``traj.jsonl`` + per-step screenshots + ``response.log`` +
  ``result.txt``).
- :func:`run_parallel` — generic multi-env runner (also used for
  ``num_envs=1``; there is no separate single-env code path).

All non-vanilla paths funnel through :class:`ReviserRunner`, which owns
the ``attempt_N`` orchestration (resume, cleanup, reviser analyze+refine).
The runner here only resolves config, builds paths, and wires things up.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager
from multiprocessing import Manager, Process, current_process
from pathlib import Path
from typing import Any, Callable

from anything2skill.agent.base import BaseAgent
from anything2skill.agent_factory import create_agent
from anything2skill.benchmark_kit import BenchmarkKit, TaskDescriptor
from anything2skill.benchmarks.registry import get_kit
from anything2skill.env_base import EnvironmentInterface
from anything2skill.metrics.tracker import ExperimentTracker
from anything2skill.parser.data_types import Skills, TutorialMaterial
from anything2skill.parser.skill_store import get_or_extract_skills
from anything2skill.parser.tutorial_loader import load_tutorials
from anything2skill.reviser import ReviserRunner
from anything2skill.reviser.reviser_runner import scan_attempts
from anything2skill.vlm.client import VLMClient

logger = logging.getLogger("anything2skill.runner")

# ---------------------------------------------------------------------------
# Globals for signal handling in parallel mode
# ---------------------------------------------------------------------------
_active_envs: list = []
_processes: list[Process] = []
_is_terminating = False


# ---------------------------------------------------------------------------
# Per-task logging + response log
# ---------------------------------------------------------------------------


@contextmanager
def _task_logging(result_dir: str | None):
    """Attach a FileHandler to root logger for per-task runtime.log."""
    if not result_dir:
        yield
        return
    handler = logging.FileHandler(os.path.join(result_dir, "runtime.log"))
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"),
    )
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


def _append_response_log(
    result_dir: str | None,
    step_num: int,
    predict_num: Any,
    phase: str,
    response: str,
) -> None:
    """Append the raw model response for this predict turn.

    Written directly with ``open("a")`` to keep it separate from the
    logging handlers — runtime.log captures logger output, response.log
    captures the unfiltered VLM responses.
    """
    if not result_dir:
        return
    try:
        with open(os.path.join(result_dir, "response.log"), "a", encoding="utf-8") as f:
            f.write(
                f"===== Step {step_num} (predict_num={predict_num}, phase={phase}) =====\n"
            )
            f.write(response or "")
            if not (response or "").endswith("\n"):
                f.write("\n")
            f.write("\n")
    except Exception as e:
        logger.warning("Failed to append response.log: %s", e)


# ---------------------------------------------------------------------------
# Core step loop (benchmark-agnostic)
# ---------------------------------------------------------------------------


def run_single_task(
    agent: BaseAgent,
    env: EnvironmentInterface,
    task: TaskDescriptor,
    max_steps: int = 15,
    sleep_after_execution: float = 2.0,
    result_dir: str | None = None,
) -> tuple[float, int]:
    """Run a single task with an Anything2Skill agent.

    Returns (score, steps_taken).
    """
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

    with _task_logging(result_dir):
        logger.info("[Task %s] Instruction: %s", task.task_id, task.instruction)
        obs = env.reset(task)

        if result_dir:
            step0_dir = Path(result_dir) / "step_0_initial"
            step0_dir.mkdir(parents=True, exist_ok=True)
            agent.kit.save_observation(obs, step0_dir)

        done = False
        step_idx = 0

        while not done and step_idx < max_steps:
            response, actions = agent.predict(task.instruction, obs)
            predict_info = agent.last_predict_info

            _append_response_log(
                result_dir,
                step_num=step_idx + 1,
                predict_num=predict_info.get("predict_num"),
                phase=predict_info.get("phase", ""),
                response=response,
            )

            for action in actions:
                action_timestamp = datetime.datetime.now().strftime(
                    "%Y%m%d@%H%M%S%f",
                )
                logger.info("Step %d: %s", step_idx + 1, action[:100])

                obs, reward, done, info = env.step(action, sleep_after_execution)

                step_data = {
                    "step_num": step_idx + 1,
                    "action_timestamp": action_timestamp,
                    "action": action,
                    "response": response,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    **predict_info,
                }

                if result_dir:
                    result_path = Path(result_dir)
                    step_obs_dir = (
                        result_path
                        / f"step_{step_idx + 1}_{action_timestamp}"
                    )
                    step_obs_dir.mkdir(parents=True, exist_ok=True)
                    agent.kit.save_observation(obs, step_obs_dir)

                    with open(result_path / "traj.jsonl", "a") as f:
                        writable = {
                            k: v for k, v in step_data.items() if k != "screenshot"
                        }
                        f.write(json.dumps(writable, default=str))
                        f.write("\n")

                if done:
                    logger.info("Episode done at step %d", step_idx + 1)
                    break

            step_idx += 1

        # Stop recording if env supports it (e.g. OSWorldEnvWrapper)
        if hasattr(env, "stop_recording") and result_dir:
            env.stop_recording(result_dir)

        score = env.evaluate()
        logger.info("Score: %.2f (steps: %d)", score, step_idx)

        if result_dir:
            with open(os.path.join(result_dir, "result.txt"), "w") as f:
                f.write(f"{score}\n")

    return score, step_idx


# ---------------------------------------------------------------------------
# Config resolution + path helpers
# ---------------------------------------------------------------------------


def _resolve_config(cfg) -> dict:
    """Resolve Hydra DictConfig to plain dict."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(cfg, resolve=True)


def _get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build_llm_params(agent_cfg: dict) -> dict:
    params = {
        "max_tokens": agent_cfg.get("max_tokens", 1500),
    }
    if agent_cfg.get("temperature") is not None:
        params["temperature"] = agent_cfg["temperature"]
    return params


def _compute_run_names(
    agent_cfg: dict, reviser_cfg: dict, max_attempts: int,
) -> tuple[str, str]:
    """Return ``(canonical_run_name, reviser_run_name)``.

    The two names partition two *semantically distinct* experiment
    families and must therefore live in different directories:

    - canonical bucket ``a2s-{mode}-{agent_model}`` = pure-agent bare
      runs (``max_attempts == 1``). Always contains ``attempt_1``.
      Read-only once written; shared across every reviser variant that
      later reuses that attempt_1.

    - reviser bucket ``a2s-{mode}-{agent_model}-r_{reviser_model}`` =
      any run that actually invokes the reviser (``max_attempts > 1``),
      **even when reviser_model equals agent_model**. That run's
      attempt_2+ plus its own task-level ``task.json`` / per-attempt
      ``meta.json`` live here so they never overwrite the bare-run
      baseline.

    When ``max_attempts == 1`` the two names coincide — there is no
    reviser activity, so no dedicated bucket is needed.
    """
    agent_mode = agent_cfg.get("agent_mode", "simple")
    agent_model = agent_cfg.get("model", "gpt-4o")
    canonical = f"a2s-{agent_mode}-{agent_model}"

    if max_attempts <= 1:
        return canonical, canonical

    reviser_model = reviser_cfg.get("model") or agent_model
    return canonical, f"{canonical}-r_{reviser_model}"


_NO_REVISER_MODES = {"vanilla", "vanilla_tutorial"}


def _effective_max_attempts(reviser_cfg: dict, agent_mode: str) -> int:
    requested = int(reviser_cfg.get("max_attempts", 1))
    if agent_mode in _NO_REVISER_MODES and requested > 1:
        logger.warning(
            "%s mode does not support reviser refinement; "
            "clamping max_attempts from %d to 1",
            agent_mode,
            requested,
        )
        return 1
    return max(1, requested)


def _make_reviser_vlm_factory(
    reviser_cfg: dict,
    agent_cfg: dict,
    agent_vlm: VLMClient,
) -> Callable[[], VLMClient]:
    """Lazy VLMClient factory: reuses agent_vlm when reviser == agent."""
    r_model = reviser_cfg.get("model") or agent_cfg.get("model", "gpt-4o")
    r_base_url = reviser_cfg.get("base_url") or agent_cfg.get("base_url")
    r_api_key = reviser_cfg.get("api_key") or agent_cfg.get("api_key")
    r_use_mct = reviser_cfg.get("use_max_completion_tokens")
    if r_use_mct is None:
        r_use_mct = agent_cfg.get("use_max_completion_tokens")
    needs_separate = (
        r_model != agent_cfg.get("model", "gpt-4o")
        or r_base_url != agent_cfg.get("base_url")
        or r_api_key != agent_cfg.get("api_key")
    )
    if not needs_separate:
        return lambda: agent_vlm
    return lambda: VLMClient(
        model=r_model, base_url=r_base_url, api_key=r_api_key,
        use_max_completion_tokens=r_use_mct,
    )


# ---------------------------------------------------------------------------
# Experiment results rebuild
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_attempt_metas(task_dir: Path) -> dict[int, dict]:
    """Return ``{n: meta_dict}`` for every parseable ``attempt_N/meta.json``."""
    out: dict[int, dict] = {}
    if not task_dir.is_dir():
        return out
    for entry in sorted(task_dir.glob("attempt_*")):
        if not entry.is_dir():
            continue
        try:
            n = int(entry.name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        meta = _read_json(entry / "meta.json")
        if meta is None:
            continue
        out[n] = meta
    return out


def _rebuild_experiment_results(
    bucket_base: str,
    canonical_base: str | None = None,
) -> None:
    """Rebuild ``experiment_results.json`` under *bucket_base*.

    Each task carries every attempt separately (no best-of-N collapsing).
    A task is included if ``task.json`` exists **and** at least one
    ``attempt_*/meta.json`` exists locally — or in the canonical
    counterpart when ``canonical_base`` differs from ``bucket_base``.

    - ``bucket_base == canonical_base`` (bare run): emit attempt_1 only.
    - Cross-bucket (reviser run): attempt_1 comes from canonical, attempt_2+
      from reviser; both are merged into one per-task ``attempts`` map.
    - Canonical-side rebuild with ``canonical_base=None``: attempt_1 only,
      as the "bare baseline" view parallel to the reviser bucket JSON.
    """
    if not os.path.isdir(bucket_base):
        return
    base = Path(bucket_base)
    canonical_path = Path(canonical_base) if canonical_base else None
    cross_bucket = canonical_path is not None and canonical_path != base

    tracker = ExperimentTracker()

    # Union of candidate task dirs from both buckets so reviser-only
    # attempts and canonical-only attempts both surface.
    seen: set[tuple[str, ...]] = set()

    def _scan(root: Path) -> None:
        if not root.is_dir():
            return
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith("attempt_")]
            if "task.json" not in filenames:
                continue
            rel = Path(os.path.relpath(dirpath, root)).parts
            seen.add(rel)

    _scan(base)
    if cross_bucket:
        _scan(canonical_path)

    for rel_parts in sorted(seen):
        task_id = rel_parts[-1] if rel_parts else ""
        domain = rel_parts[0] if len(rel_parts) > 1 else ""

        local_dir = base.joinpath(*rel_parts) if rel_parts else base
        canonical_dir = (
            canonical_path.joinpath(*rel_parts)
            if cross_bucket and canonical_path is not None
            else None
        )

        # Instruction: prefer local task.json, fall back to canonical.
        task_meta = _read_json(local_dir / "task.json") or (
            _read_json(canonical_dir / "task.json") if canonical_dir else None
        ) or {}
        instruction = str(task_meta.get("instruction", ""))

        metas: dict[int, dict] = {}
        if canonical_dir is not None:
            # Only attempt_1 is authoritative from canonical.
            canon_metas = _collect_attempt_metas(canonical_dir)
            if 1 in canon_metas:
                metas[1] = canon_metas[1]
        metas.update(_collect_attempt_metas(local_dir))

        if not metas:
            logger.debug(
                "%s has task.json but no attempt_*/meta.json; "
                "skipping in experiment_results",
                local_dir,
            )
            continue

        for n, meta in sorted(metas.items()):
            tracker.add_attempt(
                task_id=task_id,
                domain=domain,
                instruction=instruction,
                attempt_n=n,
                score=meta.get("score", 0.0),
                steps_taken=meta.get("steps", 0),
                skills_count=meta.get("skills_count", 0),
                early_stop_triggered=meta["early_stop_triggered"],
            )

    tracker.save(os.path.join(bucket_base, "experiment_results.json"))


# ---------------------------------------------------------------------------
# Resume filter (skip-only, no deletions)
# ---------------------------------------------------------------------------


def _filter_pending_tasks(
    all_tasks: list[TaskDescriptor],
    kit: BenchmarkKit,
    canonical_result_base: str,
    reviser_result_base: str,
    max_attempts: int,
) -> list[TaskDescriptor]:
    """Return tasks that still need attempts.

    A task is skipped when its completed-attempt count across both
    buckets is ``>= max_attempts``. This never deletes directories —
    cleanup of stale execution artifacts happens inside
    :class:`ReviserRunner`.
    """
    pending: list[TaskDescriptor] = []
    skipped = 0
    for task in all_tasks:
        subdir = kit.get_result_subdir(task)
        canonical_dir = Path(canonical_result_base) / subdir
        reviser_dir = Path(reviser_result_base) / subdir
        scanned = scan_attempts(canonical_dir, reviser_dir)
        completed = sum(1 for m in scanned.values() if m["completed"])
        if completed >= max_attempts:
            skipped += 1
            continue
        pending.append(task)
    if skipped:
        logger.info(
            "Resumed: %d tasks already satisfy max_attempts=%d (skipped), "
            "%d remaining",
            skipped, max_attempts, len(pending),
        )
    return pending


# ---------------------------------------------------------------------------
# In-progress lock (belt-and-braces for manual parallel launches)
# ---------------------------------------------------------------------------


def _try_acquire_task_lock(canonical_task_dir: Path) -> Path | None:
    """Atomically create ``.in_progress.lock`` under the canonical bucket.

    Returns the lock path on success, or ``None`` if another process
    already holds it. Primary uniqueness guarantee is ``Manager().Queue()``;
    this lock only defends against manual parallel launches against the
    same ``result_base``. Lock lives under the canonical bucket so both
    reviser_model variants serialize on it.

    Stale-lock recovery: the holder's PID is written into the file. On
    conflict we read it and check ``os.kill(pid, 0)`` — if the holder is
    gone (e.g. previous worker SIGKILL'd by ``_signal_handler``), we unlink
    the stale lock and retry once.
    """
    canonical_task_dir.mkdir(parents=True, exist_ok=True)
    lock_path = canonical_task_dir / ".in_progress.lock"
    if _try_create_lock(lock_path):
        return lock_path
    if _is_lock_stale(lock_path):
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        else:
            logger.warning(
                "Removed stale lock %s (previous holder no longer running)",
                lock_path,
            )
        if _try_create_lock(lock_path):
            return lock_path
    return None


def _try_create_lock(lock_path: Path) -> bool:
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
    finally:
        os.close(fd)
    return True


def _is_lock_stale(lock_path: Path) -> bool:
    """A lock is stale iff it carries a PID that is no longer alive."""
    try:
        text = lock_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        # Vanished while we were checking — treat as not-stale; the next
        # acquire attempt will sort it out.
        return False
    except OSError:
        return False
    if not text.isdigit():
        # Empty or malformed lock — left over from a pre-PID build; safe
        # to consider stale.
        return True
    pid = int(text)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        # PID is alive but owned by a different user; not ours to evict.
        return False
    return False


def _release_task_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Per-task runner (shared by worker processes)
# ---------------------------------------------------------------------------


def _load_task_skills(
    *,
    task: TaskDescriptor,
    kit: BenchmarkKit,
    agent_mode: str,
    tutorial_dir: str,
    tutorial_type: str,
    skills_cache_dir: str,
    benchmark: str,
    model_name: str,
    force_regenerate: bool,
    max_images: int | None,
    vlm: VLMClient,
    llm_params: dict,
) -> tuple[Skills, TutorialMaterial | None] | None:
    """Return (skills, tutorial) for a task. Returns None if tutorial missing."""
    if agent_mode == "vanilla":
        return (
            Skills(task_id=task.task_id, instruction=task.instruction),
            None,
        )

    tutorial_ids = kit.tutorial_ids_for(task)
    try:
        tutorial = load_tutorials(
            tutorial_ids, tutorial_dir, tutorial_type, task_id=task.task_id,
        )
    except FileNotFoundError:
        logger.warning(
            "Tutorials %s not found (task %s), skipping",
            tutorial_ids, task.task_id,
        )
        return None

    if agent_mode == "vanilla_tutorial":
        return (
            Skills(task_id=task.task_id, instruction=task.instruction),
            tutorial,
        )

    skills = get_or_extract_skills(
        tutorial=tutorial,
        vlm=vlm,
        instruction=task.instruction,
        cache_dir=skills_cache_dir,
        tutorial_type=tutorial_type,
        benchmark=benchmark,
        model=model_name,
        force_regenerate=force_regenerate,
        kit=kit,
        llm_params=llm_params,
        max_images=max_images,
    )
    return skills, tutorial


def _run_one_task(
    *,
    task: TaskDescriptor,
    kit: BenchmarkKit,
    env: EnvironmentInterface,
    agent_cfg: dict,
    reviser_cfg: dict,
    canonical_result_base: str,
    reviser_result_base: str,
    max_attempts: int,
    agent_vlm: VLMClient,
    tutorial_dir: str,
    tutorial_type: str,
    skills_cache_dir: str,
    benchmark: str,
    force_regenerate: bool,
    max_images: int | None,
    llm_params: dict,
    logger_prefix: str = "",
) -> tuple[float, int] | None:
    """Run ReviserRunner for a single task. Returns (score, steps) or None."""
    subdir = kit.get_result_subdir(task)
    canonical_task_dir = Path(canonical_result_base) / subdir
    reviser_task_dir = Path(reviser_result_base) / subdir
    canonical_task_dir.mkdir(parents=True, exist_ok=True)
    reviser_task_dir.mkdir(parents=True, exist_ok=True)

    agent_mode = agent_cfg.get("agent_mode", "simple")
    max_steps = agent_cfg.get("max_steps", 15)

    loaded = _load_task_skills(
        task=task,
        kit=kit,
        agent_mode=agent_mode,
        tutorial_dir=tutorial_dir,
        tutorial_type=tutorial_type,
        skills_cache_dir=skills_cache_dir,
        benchmark=benchmark,
        model_name=agent_cfg.get("model", "gpt-4o"),
        force_regenerate=force_regenerate,
        max_images=max_images,
        vlm=agent_vlm,
        llm_params=llm_params,
    )
    if loaded is None:
        return None
    skills, tutorial = loaded
    logger.info(
        "%sSkills: %d for task %s",
        logger_prefix, len(skills.skills), task.task_id,
    )

    reviser_vlm_factory = _make_reviser_vlm_factory(
        reviser_cfg, agent_cfg, agent_vlm,
    )
    runner = ReviserRunner(
        kit=kit,
        reviser_cfg=reviser_cfg,
        vlm_factory=reviser_vlm_factory,
        canonical_result_base=canonical_result_base,
        reviser_result_base=reviser_result_base,
    )

    def _agent_factory(current_skills: Skills) -> BaseAgent:
        return create_agent(
            agent_mode, agent_vlm, current_skills, kit, agent_cfg,
            str(reviser_task_dir),
            tutorial=tutorial,
            max_images=max_images,
        )

    lock_path = _try_acquire_task_lock(canonical_task_dir)
    if lock_path is None:
        logger.warning(
            "%sTask %s skipped: another process holds %s/.in_progress.lock",
            logger_prefix, task.task_id, canonical_task_dir,
        )
        return None
    try:
        return runner.run_with_reviser(
            agent_factory=_agent_factory,
            env=env,
            task=task,
            skills=skills,
            tutorial=tutorial,
            max_steps=max_steps,
            sleep_after_execution=agent_cfg.get("sleep_after_execution", 2.0),
            canonical_task_dir=canonical_task_dir,
            reviser_task_dir=reviser_task_dir,
            max_attempts=max_attempts,
        )
    finally:
        _release_task_lock(lock_path)


# ---------------------------------------------------------------------------
# Parallel runner (also handles num_envs=1)
# ---------------------------------------------------------------------------


def _signal_handler(signum, frame):
    global _is_terminating
    if _is_terminating:
        return
    _is_terminating = True
    logger.info("Received signal %s, shutting down...", signum)

    for env in _active_envs:
        try:
            env.close()
        except Exception:
            pass

    for p in _processes:
        if p.is_alive():
            try:
                p.terminate()
            except Exception:
                pass

    time.sleep(1)
    for p in _processes:
        if p.is_alive():
            try:
                os.kill(p.pid, signal.SIGKILL)
            except Exception:
                pass

    sys.exit(0)


def _worker(
    task_queue: Any,
    shared_scores: Any,
    benchmark: str,
    env_cfg: dict,
    agent_cfg: dict,
    skills_cfg: dict,
    data_cfg: dict,
    tasks_cfg: dict,
    canonical_result_base: str,
    reviser_result_base: str,
    reviser_cfg: dict,
    max_attempts: int,
    tutorial_type: str,
):
    """Worker process: create env + kit + pull tasks from queue."""
    proc_name = current_process().name
    env = None
    project_root = _get_project_root()

    try:
        agent_vlm = VLMClient(
            model=agent_cfg.get("model", "gpt-4o"),
            base_url=agent_cfg.get("base_url"),
            api_key=agent_cfg.get("api_key"),
            use_max_completion_tokens=agent_cfg.get("use_max_completion_tokens"),
        )

        worker_env_cfg = {**env_cfg, "agent_model": agent_cfg.get("model", "")}
        kit = get_kit(benchmark, env_cfg=worker_env_cfg)
        env = kit.create_env(worker_env_cfg)
        _active_envs.append(env)

        tutorial_dir = os.path.join(
            project_root,
            data_cfg.get("tutorial_dir", "data_tutorial"),
            tutorial_type,
            benchmark,
        )
        skills_cache_dir = os.path.join(
            project_root, skills_cfg.get("cache_dir", "skills_cache"),
        )
        force_regenerate = skills_cfg.get("force_regenerate", False)
        max_images = skills_cfg.get("max_images")
        llm_params = _build_llm_params(agent_cfg)

        logger.info("[%s] Worker started", proc_name)

        while True:
            try:
                task = task_queue.get(timeout=5)
            except Exception:
                break

            task_result_dir = ""
            try:
                logger.info(
                    "[%s][Task %s][Instruction] %s",
                    proc_name, task.task_id, task.instruction,
                )
                task_result_dir = os.path.join(
                    reviser_result_base, kit.get_result_subdir(task),
                )

                result = _run_one_task(
                    task=task,
                    kit=kit,
                    env=env,
                    agent_cfg=agent_cfg,
                    reviser_cfg=reviser_cfg,
                    canonical_result_base=canonical_result_base,
                    reviser_result_base=reviser_result_base,
                    max_attempts=max_attempts,
                    agent_vlm=agent_vlm,
                    tutorial_dir=tutorial_dir,
                    tutorial_type=tutorial_type,
                    skills_cache_dir=skills_cache_dir,
                    benchmark=benchmark,
                    force_regenerate=force_regenerate,
                    max_images=max_images,
                    llm_params=llm_params,
                    logger_prefix=f"[{proc_name}] ",
                )
                if result is None:
                    continue
                score, _steps = result
                shared_scores.append(score)
                # Per-attempt rebuild now happens inside ReviserRunner
                # right after each meta.json is persisted; no per-task
                # rebuild needed here.

            except Exception as e:
                import traceback
                logger.error("[%s] %s error: %s", proc_name, task.task_id, e)
                logger.error(traceback.format_exc())
                try:
                    if hasattr(env, "stop_recording"):
                        env.stop_recording(task_result_dir)
                except Exception:
                    pass
                if task_result_dir:
                    os.makedirs(task_result_dir, exist_ok=True)
                    with open(os.path.join(task_result_dir, "error.json"), "w") as f:
                        f.write(json.dumps({"Error": str(e)}))
                        f.write("\n")

    except Exception as e:
        import traceback
        logger.error("[%s] Process-level error: %s", proc_name, e)
        logger.error(traceback.format_exc())
    finally:
        logger.info("[%s] Cleaning up...", proc_name)
        if env is not None:
            try:
                env.close()
            except Exception as e:
                logger.error("[%s] Cleanup error: %s", proc_name, e)


def run_parallel(cfg) -> None:
    """Generic multi-env parallel runner for any benchmark."""
    global _processes

    config = _resolve_config(cfg)
    env_cfg = config.get("env", {})
    agent_cfg = config.get("agent", {})
    skills_cfg = config.get("skills", {})
    data_cfg = config.get("data", {})
    tasks_cfg = config.get("tasks", {})
    runner_cfg = config.get("runner", {})
    reviser_cfg = config.get("reviser", {})

    benchmark = config.get("benchmark", "osworld")
    project_root = _get_project_root()
    num_envs = runner_cfg.get("num_envs", 1)
    agent_mode = agent_cfg.get("agent_mode", "simple")
    max_attempts = _effective_max_attempts(reviser_cfg, agent_mode)
    tutorial_type = data_cfg.get("tutorial_type", "html")

    canonical_run_name, reviser_run_name = _compute_run_names(
        agent_cfg, reviser_cfg, max_attempts,
    )
    result_root = os.path.join(
        project_root,
        tasks_cfg.get("result_dir", "results"),
        tutorial_type,
    )
    canonical_result_base = os.path.join(result_root, canonical_run_name, benchmark)
    reviser_result_base = os.path.join(result_root, reviser_run_name, benchmark)

    kit = get_kit(benchmark, env_cfg=env_cfg)
    all_tasks = kit.collect_tasks(config)

    pending_tasks = _filter_pending_tasks(
        all_tasks, kit, canonical_result_base, reviser_result_base, max_attempts,
    )
    if not pending_tasks:
        logger.info("All tasks already completed, nothing to run")
        _rebuild_experiment_results(reviser_result_base, canonical_result_base)
        return

    num_envs = min(num_envs, len(pending_tasks))
    logger.info("Pending tasks: %d, workers: %d", len(pending_tasks), num_envs)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        for item in pending_tasks:
            task_queue.put(item)

        _processes = []
        worker_args = (
            task_queue, shared_scores,
            benchmark, env_cfg, agent_cfg, skills_cfg,
            data_cfg, tasks_cfg,
            canonical_result_base, reviser_result_base,
            reviser_cfg, max_attempts,
            tutorial_type,
        )
        for i in range(num_envs):
            p = Process(
                target=_worker,
                args=worker_args,
                name=f"A2S-Worker-{i+1}",
                daemon=True,
            )
            p.start()
            _processes.append(p)
            logger.info("Started %s (PID %d)", p.name, p.pid)

        try:
            while True:
                alive = sum(1 for p in _processes if p.is_alive())
                if task_queue.empty():
                    logger.info("All tasks dispatched, waiting for workers...")
                    break
                if alive == 0:
                    logger.error("All workers died")
                    break

                for idx, p in enumerate(_processes):
                    if not p.is_alive() and not task_queue.empty():
                        logger.warning("%s died, restarting...", p.name)
                        new_p = Process(
                            target=_worker,
                            args=worker_args,
                            name=f"A2S-Worker-Restart-{idx+1}",
                            daemon=True,
                        )
                        new_p.start()
                        _processes[idx] = new_p

                time.sleep(5)

            for p in _processes:
                p.join()

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt, shutting down...")
            _signal_handler(signal.SIGINT, None)

        scores = list(shared_scores)

    _rebuild_experiment_results(reviser_result_base, canonical_result_base)
    if canonical_result_base != reviser_result_base:
        _rebuild_experiment_results(canonical_result_base)
    if scores:
        logger.info(
            "Tasks: %d, Average: %.4f", len(scores), sum(scores) / len(scores),
        )
    else:
        logger.info("No tasks completed")
