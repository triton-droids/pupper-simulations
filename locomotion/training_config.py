from typing import Dict, Any


class TrainingConfig:
    """Training configuration for Bittle locomotion policy."""

    def __init__(self):
        # PPO schedule
        self.num_timesteps = 100_000_000
        self.num_evals = 10
        self.episode_length = 1000
        self.num_envs = 8192
        self.batch_size = 256
        self.unroll_length = 20
        self.num_minibatches = 32
        self.num_updates_per_batch = 4

        # PPO hyperparameters
        self.policy_hidden_layer_sizes = (128, 128, 128, 128)
        self.reward_scaling = 1.0
        self.normalize_observations = False
        self.action_repeat = 1
        self.discounting = 0.97
        self.learning_rate = 3.0e-4
        self.entropy_cost = 1e-2
        self.seed = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for wandb logging."""
        return {
            "num_timesteps": self.num_timesteps,
            "num_evals": self.num_evals,
            "episode_length": self.episode_length,
            "num_envs": self.num_envs,
            "batch_size": self.batch_size,
            "unroll_length": self.unroll_length,
            "num_minibatches": self.num_minibatches,
            "num_updates_per_batch": self.num_updates_per_batch,
            "policy_hidden_layer_sizes": self.policy_hidden_layer_sizes,
            "reward_scaling": self.reward_scaling,
            "normalize_observations": self.normalize_observations,
            "action_repeat": self.action_repeat,
            "discounting": self.discounting,
            "learning_rate": self.learning_rate,
            "entropy_cost": self.entropy_cost,
            "seed": self.seed,
        }
