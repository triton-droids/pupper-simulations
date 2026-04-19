"""
Plain-English training settings for Bittle runs.

Think of this file as the preset shelf for training jobs:

- the class holds the main "how big/how long" knobs for a run
- the normal walking task keeps the large default values
- the dance task can swap in a smaller preset that is better for that job

The object stays editable on purpose so sweep scripts can take one of these
presets and then tweak a few values for experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from locomotion.tasks import normalize_task_name


TEST_MODE_OVERRIDES = {
    "num_timesteps": 10_000,
    "num_evals": 2,
    "episode_length": 100,
    "num_envs": 10,
    "batch_size": 100,
    "unroll_length": 5,
    "num_minibatches": 2,
    "num_updates_per_batch": 1,
}

# The first dance sweep improved most in the smallest batch/env setting, and
# that run was still getting better when the run ended. These presets start new
# dance runs near that "best so far" region instead of at the walking scale.
TASK_FULL_MODE_OVERRIDES = {
    "dance": {
        "num_timesteps": 200_000,
        "num_evals": 6,
        "episode_length": 200,
        "num_envs": 32,
        "batch_size": 32,
        "unroll_length": 8,
        "num_minibatches": 4,
        "num_updates_per_batch": 1,
    },
}

TASK_TEST_MODE_OVERRIDES = {
    "dance": {
        "num_timesteps": 20_000,
        "num_evals": 3,
        "episode_length": 200,
        "num_envs": 16,
        "batch_size": 16,
        "unroll_length": 8,
        "num_minibatches": 2,
        "num_updates_per_batch": 1,
    },
}


@dataclass(slots=True)
class TrainingConfig:
    """
    Store the size and duration settings for one training run.

    In everyday terms, this object answers questions like:

    - how long should we train?
    - how many simulated robots should we run at once?
    - how often should we stop and check progress?
    """

    test_mode: bool = False
    num_timesteps: int = 10_000_000
    num_evals: int = 10
    episode_length: int = 1000
    num_envs: int = 5000
    batch_size: int = 500
    unroll_length: int = 20
    num_minibatches: int = 10
    num_updates_per_batch: int = 1

    def __post_init__(self) -> None:
        """
        Shrink the run when someone asked for test mode.

        Test mode is the quick "make sure the plumbing still works" version of
        training, not the version you use to get a strong policy.
        """
        if not self.test_mode:
            return

        # Copy the smaller smoke-test values onto this config object.
        for field_name, value in TEST_MODE_OVERRIDES.items():
            setattr(self, field_name, value)

    @classmethod
    def for_task(cls, task_name: str, *, test_mode: bool = False) -> "TrainingConfig":
        """
        Build a config that already matches the requested task.

        This is the convenience constructor used by the trainer and the sweep
        runner so they both start from the same default assumptions.
        """
        config = cls(test_mode=test_mode)
        config.apply_task_preset(task_name)
        return config

    def apply_task_preset(self, task_name: str) -> None:
        """
        Layer a task-specific preset on top of the base settings.

        In practice, this means "if the task is dance, replace the general
        defaults with the smaller dance-friendly values."
        """
        task_name = normalize_task_name(task_name)
        task_overrides_by_mode = (
            TASK_TEST_MODE_OVERRIDES if self.test_mode else TASK_FULL_MODE_OVERRIDES
        )
        task_overrides = task_overrides_by_mode.get(task_name, {})
        # Apply any task-specific values one field at a time.
        for field_name, value in task_overrides.items():
            setattr(self, field_name, value)

    def to_dict(self) -> dict[str, Any]:
        """Turn the dataclass into plain JSON-friendly data for logs and files."""
        return asdict(self)
