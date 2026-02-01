from typing import Dict, Any

# ============================================================================
# Configuration
# ============================================================================


class TrainingConfig:
    """Training configuration with sensible defaults."""

    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode

        if test_mode:
            # Minimal config for fast iteration (~6 min on A100)
            self.num_timesteps = 10_000
            self.num_evals = 2
            self.episode_length = 100
            self.num_envs = 8
            self.batch_size = 128
            self.unroll_length = 5
            self.num_minibatches = 2
            self.num_updates_per_batch = 1
        else:
            # Full training config (~30 min on A100)
            self.num_timesteps = 10_000_000
            self.num_evals = 10
            self.episode_length = 1000
            self.num_envs = 4095
            self.batch_size = 512
            self.unroll_length = 20
            self.num_minibatches = 8
            self.num_updates_per_batch = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "test_mode": self.test_mode,
            "num_timesteps": self.num_timesteps,
            "num_evals": self.num_evals,
            "episode_length": self.episode_length,
            "num_envs": self.num_envs,
            "batch_size": self.batch_size,
            "unroll_length": self.unroll_length,
            "num_minibatches": self.num_minibatches,
            "num_updates_per_batch": self.num_updates_per_batch,
        }
