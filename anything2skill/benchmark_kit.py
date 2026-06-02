"""BenchmarkKit: abstract base class for benchmark-specific behavior.

Each benchmark implements a BenchmarkKit subclass that provides:
- Observation encoding (raw env obs -> VLM content blocks)
- Domain system prompt (identity + action format rules)
- Action parsing (LLM response -> executable actions)
- Environment creation and task collection

All message orchestration (multi-turn history, skills formatting, planner
prompt construction) is handled by the agent layer's MessageBuilder.
Kit developers do **not** need to assemble VLM message lists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anything2skill.env_base import EnvironmentInterface


@dataclass
class TaskDescriptor:
    """Base task descriptor with only universally required fields.

    Benchmarks should subclass this to add benchmark-specific data
    (e.g., domain, task_config dict, snapshot info).
    The generic runner only accesses task_id and instruction.
    """

    task_id: str
    instruction: str


class BenchmarkKit(ABC):
    """Implement this to support a new benchmark domain.

    Required methods (must override):
        encode_observation, parse_actions, system_prompt,
        create_env, collect_tasks

    Optional methods (have defaults):
        bridging_text, reflection_guidance, planner_guidance,
        skill_extraction_guidance, tutorial_ids_for,
        save_observation, get_result_subdir
    """

    def __init__(self, env_cfg: dict | None = None):
        self.env_cfg = env_cfg or {}

    # ------------------------------------------------------------------
    # Required: observation encoding
    # ------------------------------------------------------------------

    @abstractmethod
    def encode_observation(self, obs: dict) -> list[dict]:
        """Convert raw environment observation into VLM content blocks.

        This is where env feedback is processed (e.g., error messages,
        screenshots, simulation state, board position).

        Returns:
            List of OpenAI-format content blocks.
        """

    # ------------------------------------------------------------------
    # Required: action parsing
    # ------------------------------------------------------------------

    @abstractmethod
    def parse_actions(self, response: str) -> list[str]:
        """Parse LLM response text into executable action strings.

        Should recognize framework-level tokens: DONE, FAIL, WAIT.
        """

    # ------------------------------------------------------------------
    # Required: domain system prompt
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Domain identity and action format rules.

        Should include:
        - What kind of environment this is
        - What actions the agent can take
        - How actions should be formatted in the response
        - Domain-specific constraints and special tokens

        Should NOT include:
        - Skills/SOP usage instructions (managed by MessageBuilder)
        - Planner format (managed by MessageBuilder)
        - Multi-turn history instructions (managed by MessageBuilder)
        """


    # ------------------------------------------------------------------
    # Optional: with sensible defaults
    # ------------------------------------------------------------------

    @property
    def bridging_text(self) -> str:
        """Text prefixed to each historical observation in multi-turn context.

        Default: generic prompt.
        """
        return "Given the current observation. What's the next step?"

    @property
    def reflection_guidance(self) -> str:
        """Extra domain-specific guidance for reflection / error recovery.

        Appended to the framework's reflection system prompt.
        Default: empty.
        """
        return ""

    @property
    def planner_guidance(self) -> str:
        """Extra domain-specific context for the planner prompt.

        Inserted into the framework's planner system prompt.
        Default: empty.
        """
        return ""

    @property
    def skill_extraction_guidance(self) -> str:
        """Domain-specific guidance for skill extraction from tutorials.

        Inserted into the framework's extraction system prompt.
        Default: empty.
        """
        return ""

    @property
    def reviser_guidance(self) -> str:
        """Domain context for reviser trajectory analysis and skill refinement.

        Describes the environment, action format, and common failure patterns
        the VLM should look for when reading a trajectory. Injected into the
        reviser analyzer / refiner system prompts. Default: empty.
        """
        return ""

    def save_observation(self, obs: dict, path: Path) -> None:
        """Save observation artifact to disk for logging."""

    def load_saved_observation(self, step_dir: Path) -> list[dict]:
        """Load a previously saved observation from disk as VLM content blocks.

        Counterpart to save_observation(). Returns the same format as
        encode_observation() — list of text/image_url dicts.
        Default: empty list.
        """
        return []

    # ------------------------------------------------------------------
    # Required: environment and task management
    # ------------------------------------------------------------------

    @abstractmethod
    def create_env(self, env_cfg: dict) -> EnvironmentInterface:
        """Create and return a benchmark environment instance.

        Called once per worker (single-env) or once per worker process
        (multi-env parallel). Each call should return a new instance.

        Args:
            env_cfg: Environment config dict from Hydra config.
        """

    @abstractmethod
    def collect_tasks(self, cfg: dict) -> list[TaskDescriptor]:
        """Collect the list of tasks to run.

        Receives the full resolved config dict. Benchmark implementations
        should extract whatever they need (task filters, data paths, etc.)
        and return TaskDescriptor subclass instances.

        Args:
            cfg: Full resolved Hydra config as a plain dict.
        """

    # ------------------------------------------------------------------
    # Optional: result directory customization
    # ------------------------------------------------------------------

    def get_result_subdir(self, task: TaskDescriptor) -> str:
        """Return the relative path for this task's result directory.

        Default: just the task_id. Override to add benchmark-specific
        hierarchy (e.g., ``f"{task.domain}/{task.task_id}"``).
        """
        return task.task_id

    # ------------------------------------------------------------------
    # Optional: tutorial source customization
    # ------------------------------------------------------------------

    def tutorial_ids_for(self, task: TaskDescriptor) -> list[str]:
        """Return the list of tutorial directory names to load for this task.

        Used by the runner to support per-task layering (e.g. a per-task
        recipe page combined with a shared mechanics guide). Tutorials are
        merged in order — later entries are concatenated after earlier ones
        and missing entries are skipped with a warning.

        Default: ``[task.task_id]`` (single tutorial named after the task),
        which matches the common 1:1 layout used by OSWorld-style benchmarks.
        """
        return [task.task_id]
