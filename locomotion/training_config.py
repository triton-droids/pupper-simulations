"""
Training configuration for Bittle PPO runs.

The project currently uses two presets:

- full training: the default configuration used for normal experiments
- test mode: a smaller configuration meant for smoke tests and quick iteration

The object remains mutable on purpose because sweep utilities override fields
after construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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


@dataclass(slots=True)
class TrainingConfig:
    """Container for PPO hyperparameters and runtime mode selection."""

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
        """Apply the smaller test preset when requested."""
        if not self.test_mode:
            return

        for field_name, value in TEST_MODE_OVERRIDES.items():
            setattr(self, field_name, value)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation for logging and JSON output."""
        return asdict(self)
