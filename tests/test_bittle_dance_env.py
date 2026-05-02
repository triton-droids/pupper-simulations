"""Tests for dance-task reward helpers."""

from __future__ import annotations

import sys
import unittest
from inspect import signature
from pathlib import Path

import numpy as np
import jax.numpy as jp
from brax.io import mjcf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomotion.paths import DEFAULT_SCENE_PATH
from locomotion.tasks.bittle_dance_env import (
    BittleDanceEnv,
    DEFAULT_ACTION_SCALE,
    DEFAULT_CYCLE_STEPS,
    DEFAULT_DANCE_AMPLITUDE,
    DEFAULT_ENABLE_KICKS,
    DEFAULT_KICK_VEL,
    DEFAULT_POSE,
    DEFAULT_POSE_TRACKING_SCALE,
    DEFAULT_UPRIGHT_SCALE,
    TERMINATION_MARGIN,
    _actuator_position_ranges,
    build_reward_config,
)


class BittleDanceEnvRewardTests(unittest.TestCase):
    def test_default_baseline_matches_latest_best_sweep(self) -> None:
        init_signature = signature(BittleDanceEnv.__init__)

        self.assertEqual(
            init_signature.parameters["action_scale"].default,
            DEFAULT_ACTION_SCALE,
        )
        self.assertEqual(
            init_signature.parameters["kick_vel"].default,
            DEFAULT_KICK_VEL,
        )
        self.assertEqual(
            init_signature.parameters["enable_kicks"].default,
            DEFAULT_ENABLE_KICKS,
        )
        self.assertEqual(
            init_signature.parameters["cycle_steps"].default,
            DEFAULT_CYCLE_STEPS,
        )
        self.assertEqual(
            init_signature.parameters["dance_amplitude"].default,
            DEFAULT_DANCE_AMPLITUDE,
        )

        config = build_reward_config()
        self.assertEqual(
            config.rewards.scales.pose_tracking,
            DEFAULT_POSE_TRACKING_SCALE,
        )
        self.assertEqual(config.rewards.scales.upright, DEFAULT_UPRIGHT_SCALE)

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

    def test_default_pose_is_inside_model_actuator_limits(self) -> None:
        sys = mjcf.load(str(DEFAULT_SCENE_PATH))
        lowers, uppers = _actuator_position_ranges(sys)

        self.assertTrue(
            np.all(np.asarray(DEFAULT_POSE) >= np.asarray(lowers) - TERMINATION_MARGIN)
        )
        self.assertTrue(
            np.all(np.asarray(DEFAULT_POSE) <= np.asarray(uppers) + TERMINATION_MARGIN)
        )


if __name__ == "__main__":
    unittest.main()
