"""
Helpers for the trainable Bittle task environments.

This package groups the simulator rules for each task and the small helper
functions that let the sweep runner discover the matching task-specific JSON
files.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


TASK_ALIASES = {
    "walking": "walking",
    "locomotion": "walking",
    "dance": "dance",
}

TASK_CLI_CHOICES = ("walking", "dance", "locomotion")

TASK_MODULE_NAMES = {
    "walking": "locomotion.tasks.bittle_walk_env",
    "dance": "locomotion.tasks.bittle_dance_env",
}


def normalize_task_name(task_name: str) -> str:
    """Map user-facing task names and aliases onto one canonical task key."""
    try:
        return TASK_ALIASES[task_name]
    except KeyError as exc:
        valid = ", ".join(TASK_CLI_CHOICES)
        raise ValueError(f"Unknown task '{task_name}'. Expected one of: {valid}") from exc


def import_task_module(task_name: str) -> ModuleType:
    """
    Load the Python module that defines one task.

    The sweep runner uses this instead of hardcoding per-task import logic in
    multiple files.
    """
    canonical_task = normalize_task_name(task_name)
    module_name = TASK_MODULE_NAMES[canonical_task]

    return import_module(module_name)


__all__ = [
    "TASK_ALIASES",
    "TASK_CLI_CHOICES",
    "TASK_MODULE_NAMES",
    "normalize_task_name",
    "import_task_module",
]
