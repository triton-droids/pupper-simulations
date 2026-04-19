"""Tests for dance-task reward helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import jax.numpy as jp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomotion.tasks.bittle_dance_env import BittleDanceEnv, build_reward_config


class BittleDanceEnvRewardTests(unittest.TestCase):
    def test_termination_scale_is_large_enough_to_matter_after_dt_scaling(self) -> None:
        config = build_reward_config()
        self.assertEqual(config.rewards.scales.termination, -20.0)

    def test_termination_penalty_is_stronger_for_earlier_falls(self) -> None:
        env = BittleDanceEnv.__new__(BittleDanceEnv)
        env._cycle_steps = 100

        early_penalty = float(np.asarray(env._reward_termination(jp.array(True), jp.array(0))))
        late_penalty = float(np.asarray(env._reward_termination(jp.array(True), jp.array(90))))
        no_penalty = float(np.asarray(env._reward_termination(jp.array(False), jp.array(0))))

        self.assertAlmostEqual(early_penalty, 2.0)
        self.assertAlmostEqual(late_penalty, 1.1)
        self.assertEqual(no_penalty, 0.0)
        self.assertGreater(early_penalty, late_penalty)


if __name__ == "__main__":
    unittest.main()
