"""Tests for task-aware training presets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomotion.training.config import TrainingConfig


class TrainingConfigTests(unittest.TestCase):
    def test_walking_defaults_remain_unchanged(self) -> None:
        config = TrainingConfig.for_task("walking")

        self.assertEqual(config.num_timesteps, 10_000_000)
        self.assertEqual(config.num_evals, 10)
        self.assertEqual(config.episode_length, 1000)
        self.assertEqual(config.num_envs, 5000)
        self.assertEqual(config.batch_size, 500)
        self.assertEqual(config.unroll_length, 20)
        self.assertEqual(config.num_minibatches, 10)

    def test_locomotion_alias_maps_to_walking_defaults(self) -> None:
        config = TrainingConfig.for_task("locomotion")

        self.assertEqual(config.num_timesteps, 10_000_000)
        self.assertEqual(config.num_envs, 5000)

    def test_dance_full_preset_uses_refined_defaults(self) -> None:
        config = TrainingConfig.for_task("dance")

        self.assertEqual(config.num_timesteps, 200_000)
        self.assertEqual(config.num_evals, 6)
        self.assertEqual(config.episode_length, 200)
        self.assertEqual(config.num_envs, 32)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.unroll_length, 8)
        self.assertEqual(config.num_minibatches, 4)
        self.assertEqual(config.num_updates_per_batch, 1)

    def test_dance_test_preset_keeps_same_shape_with_smaller_budget(self) -> None:
        config = TrainingConfig.for_task("dance", test_mode=True)

        self.assertTrue(config.test_mode)
        self.assertEqual(config.num_timesteps, 20_000)
        self.assertEqual(config.num_evals, 3)
        self.assertEqual(config.episode_length, 200)
        self.assertEqual(config.num_envs, 16)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.unroll_length, 8)
        self.assertEqual(config.num_minibatches, 2)


if __name__ == "__main__":
    unittest.main()
