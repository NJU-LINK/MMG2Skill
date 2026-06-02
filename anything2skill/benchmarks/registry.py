"""BenchmarkKit registry: discover and instantiate benchmark kits by name."""

from __future__ import annotations

import importlib
import logging

from anything2skill.benchmark_kit import BenchmarkKit

logger = logging.getLogger("anything2skill.benchmarks.registry")

_REGISTRY: dict[str, type[BenchmarkKit]] = {}


def register_kit(name: str):
    """Class decorator to register a BenchmarkKit implementation.

    Usage::

        @register_kit("osworld")
        class OSWorldKit(BenchmarkKit):
            ...
    """

    def decorator(cls: type[BenchmarkKit]) -> type[BenchmarkKit]:
        if name in _REGISTRY:
            logger.warning(
                "Overwriting existing kit '%s' (%s) with %s",
                name,
                _REGISTRY[name].__name__,
                cls.__name__,
            )
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_kit(name: str, env_cfg: dict | None = None) -> BenchmarkKit:
    """Instantiate a registered BenchmarkKit by benchmark name.

    If the name is not yet registered, attempts to auto-import
    ``anything2skill.benchmarks.{name}`` which should trigger
    the ``@register_kit`` decorator.

    Args:
        name: Benchmark name (e.g. "osworld", "webarena").
        env_cfg: Environment config dict from Hydra config.

    Returns:
        A BenchmarkKit instance.

    Raises:
        KeyError: If the benchmark name is not registered.
    """
    if name not in _REGISTRY:
        try:
            importlib.import_module(f"anything2skill.benchmarks.{name}")
        except ModuleNotFoundError:
            pass

    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown benchmark: '{name}'. Available: {available}"
        )

    return _REGISTRY[name](env_cfg=env_cfg)


def list_kits() -> list[str]:
    """Return names of all registered kits."""
    return sorted(_REGISTRY)
