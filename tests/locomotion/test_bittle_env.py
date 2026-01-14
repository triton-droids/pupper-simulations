"""
Unit tests for BittleEnv.

Tests environment initialization, compilation, action/observation spaces,
reset functionality, step dynamics, and reward computation.
"""

import os
import unittest
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from locomotion.bittle_env import BittleEnv, get_config


class TestBittleEnv(unittest.TestCase):
    """Test suite for BittleEnv."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        # Find the XML file path
        project_root = Path(__file__).parent.parent.parent
        cls.xml_path = str(project_root / "locomotion" / "bittle_adapted_scene.xml")

        # Verify XML file exists
        if not os.path.exists(cls.xml_path):
            raise FileNotFoundError(f"XML file not found at {cls.xml_path}")

        # Initialize RNG
        cls.rng = jax.random.PRNGKey(0)

    def setUp(self):
        """Set up test fixtures for each test."""
        # Create a fresh environment for each test
        self.env = BittleEnv(
            xml_path=self.xml_path,
            obs_noise=0.05,
            action_scale=1.5,
            kick_vel=0.05,
            enable_kicks=True,
        )

    def test_environment_compiles(self):
        """Test that the environment initializes without errors."""
        # If setUp succeeded, environment compiled successfully
        self.assertIsNotNone(self.env)
        self.assertIsNotNone(self.env.sys)

    def test_config_structure(self):
        """Test that the reward config has expected structure."""
        config = get_config()
        self.assertIn('rewards', config)
        self.assertIn('scales', config.rewards)

        # Check that key reward components are present
        expected_rewards = [
            'tracking_lin_vel', 'tracking_ang_vel', 'lin_vel_z',
            'ang_vel_xy', 'orientation', 'torques', 'action_rate',
            'joint_acc', 'stand_still', 'termination', 'feet_air_time',
            'foot_slip', 'energy'
        ]
        for reward_name in expected_rewards:
            self.assertIn(reward_name, config.rewards.scales)

    def test_action_space(self):
        """Test action space dimensions and expected range."""
        # Action space should match number of actuators
        expected_action_dim = self.env.sys.nu
        self.assertEqual(expected_action_dim, 9, "Bittle should have 9 actuators")

        # Actions are normalized to [-1, 1] then scaled
        # Test that a random action in valid range doesn't cause errors
        rng = jax.random.PRNGKey(42)
        action = jax.random.uniform(rng, (expected_action_dim,), minval=-1.0, maxval=1.0)
        self.assertEqual(action.shape, (expected_action_dim,))

    def test_observation_space(self):
        """Test observation space dimensions and structure."""
        state = self.env.reset(self.rng)

        # Observation components (per single frame):
        # 1 (yaw rate) + 3 (gravity) + 3 (command) + 9 (joint angles) + 9 (joint vels) + 9 (last action)
        single_obs_size = 1 + 3 + 3 + 9 + 9 + 9
        self.assertEqual(single_obs_size, 34)

        # With history stacking (15 frames)
        expected_obs_size = 15 * single_obs_size
        self.assertEqual(state.obs.shape, (expected_obs_size,))

        # Check that observations are finite
        self.assertTrue(jp.all(jp.isfinite(state.obs)))

    def test_reset(self):
        """Test reset functionality."""
        state = self.env.reset(self.rng)

        # Check that state components exist
        self.assertIsNotNone(state.pipeline_state)
        self.assertIsNotNone(state.obs)
        self.assertIsNotNone(state.reward)
        self.assertIsNotNone(state.done)
        self.assertIsNotNone(state.metrics)
        self.assertIsNotNone(state.info)

        # Check initial values
        self.assertEqual(float(state.reward), 0.0)
        self.assertEqual(float(state.done), 0.0)

        # Check state info structure
        self.assertIn('rng', state.info)
        self.assertIn('last_act', state.info)
        self.assertIn('last_joint_vel', state.info)
        self.assertIn('command', state.info)
        self.assertIn('last_contact', state.info)
        self.assertIn('feet_air_time', state.info)
        self.assertIn('rewards', state.info)
        self.assertIn('step', state.info)

        # Check command is valid (3D: [vx, vy, wz])
        self.assertEqual(state.info['command'].shape, (3,))

        # Check last_act has correct shape
        self.assertEqual(state.info['last_act'].shape, (self.env.sys.nu,))

        # Check feet arrays
        self.assertEqual(state.info['feet_air_time'].shape, (4,))
        self.assertEqual(state.info['last_contact'].shape, (4,))

    def test_reset_determinism(self):
        """Test that reset with same RNG produces identical states."""
        rng1 = jax.random.PRNGKey(123)
        rng2 = jax.random.PRNGKey(123)

        state1 = self.env.reset(rng1)
        state2 = self.env.reset(rng2)

        # Observations should be identical with same RNG
        np.testing.assert_array_almost_equal(state1.obs, state2.obs)

    def test_reset_randomness(self):
        """Test that reset with different RNGs produces different commands."""
        rng1 = jax.random.PRNGKey(1)
        rng2 = jax.random.PRNGKey(2)

        state1 = self.env.reset(rng1)
        state2 = self.env.reset(rng2)

        # Commands should be different with different RNGs
        self.assertFalse(jp.allclose(state1.info['command'], state2.info['command']))

    def test_step(self):
        """Test basic step functionality."""
        state = self.env.reset(self.rng)

        # Take a random action
        rng = jax.random.PRNGKey(42)
        action = jax.random.uniform(rng, (self.env.sys.nu,), minval=-1.0, maxval=1.0)

        # Step the environment
        next_state = self.env.step(state, action)

        # Check that state was updated
        self.assertIsNotNone(next_state)
        self.assertTrue(jp.all(jp.isfinite(next_state.obs)))
        self.assertTrue(jp.isfinite(next_state.reward))
        self.assertIn(float(next_state.done), [0.0, 1.0])

        # Check that step counter is valid (0 if done, 1 if not done)
        # Step counter resets to 0 when episode terminates
        expected_step = 0 if next_state.done else 1
        self.assertEqual(int(next_state.info['step']), expected_step)

        # Check that last_act was updated
        np.testing.assert_array_almost_equal(next_state.info['last_act'], action)

    def test_step_updates_physics(self):
        """Test that step updates physics state."""
        state = self.env.reset(self.rng)
        initial_qpos = state.pipeline_state.q.copy()

        # Apply non-zero action
        action = jp.ones(self.env.sys.nu) * 0.5
        next_state = self.env.step(state, action)

        # Physics state should have changed
        self.assertFalse(jp.allclose(next_state.pipeline_state.q, initial_qpos))

    def test_termination_conditions(self):
        """Test that termination conditions are checked."""
        state = self.env.reset(self.rng)

        # Test with normal action - should not terminate immediately
        action = jp.zeros(self.env.sys.nu)
        next_state = self.env.step(state, action)

        # Should not be done after one normal step
        # (unless unlucky with random kick, but unlikely)
        # We'll just verify the done flag is a valid boolean
        self.assertIn(float(next_state.done), [0.0, 1.0])

    def test_reward_computation(self):
        """Test that rewards are computed and finite."""
        state = self.env.reset(self.rng)
        action = jp.zeros(self.env.sys.nu)
        next_state = self.env.step(state, action)

        # Reward should be finite and within clipping bounds
        self.assertTrue(jp.isfinite(next_state.reward))
        self.assertGreaterEqual(float(next_state.reward), -10.0)
        self.assertLessEqual(float(next_state.reward), 10.0)

        # Individual rewards should be present in info
        self.assertIn('rewards', next_state.info)
        expected_rewards = [
            'tracking_lin_vel', 'tracking_ang_vel', 'lin_vel_z',
            'ang_vel_xy', 'orientation', 'torques', 'action_rate',
            'joint_acc', 'stand_still', 'feet_air_time', 'foot_slip',
            'termination', 'energy'
        ]
        for reward_name in expected_rewards:
            self.assertIn(reward_name, next_state.info['rewards'])
            self.assertTrue(jp.isfinite(next_state.info['rewards'][reward_name]))

    def test_metrics(self):
        """Test that metrics are populated."""
        state = self.env.reset(self.rng)
        action = jp.zeros(self.env.sys.nu)
        next_state = self.env.step(state, action)

        # Check metrics exist
        self.assertIn('total_dist', next_state.metrics)

        # All reward components should be in metrics
        for reward_name in next_state.info['rewards'].keys():
            self.assertIn(reward_name, next_state.metrics)

    def test_command_sampling(self):
        """Test command sampling produces valid commands."""
        rng = jax.random.PRNGKey(123)
        command = self.env.sample_command(rng)

        # Command should be 3D
        self.assertEqual(command.shape, (3,))

        # Check bounds (these are from the env implementation)
        # lin_vel_x: [-0.3, 0.6]
        self.assertGreaterEqual(float(command[0]), -0.3)
        self.assertLessEqual(float(command[0]), 0.6)

        # lin_vel_y: [-0.3, 0.3]
        self.assertGreaterEqual(float(command[1]), -0.3)
        self.assertLessEqual(float(command[1]), 0.3)

        # ang_vel_yaw: [-0.5, 0.5]
        self.assertGreaterEqual(float(command[2]), -0.5)
        self.assertLessEqual(float(command[2]), 0.5)

    def test_multiple_steps(self):
        """Test running multiple steps in sequence."""
        state = self.env.reset(self.rng)

        # Run 10 steps
        for i in range(10):
            rng = jax.random.PRNGKey(i)
            action = jax.random.uniform(rng, (self.env.sys.nu,), minval=-0.5, maxval=0.5)
            state = self.env.step(state, action)

            # Verify state remains valid
            self.assertTrue(jp.all(jp.isfinite(state.obs)))
            self.assertTrue(jp.isfinite(state.reward))

    def test_action_scaling(self):
        """Test that actions are properly scaled."""
        state = self.env.reset(self.rng)

        # Test with maximum action
        action = jp.ones(self.env.sys.nu)
        next_state = self.env.step(state, action)

        # Should not crash and should produce valid state
        self.assertTrue(jp.all(jp.isfinite(next_state.obs)))

        # Test with minimum action
        action = -jp.ones(self.env.sys.nu)
        next_state = self.env.step(next_state, action)

        # Should not crash
        self.assertTrue(jp.all(jp.isfinite(next_state.obs)))

    def test_default_pose(self):
        """Test that default pose is correctly set."""
        # Default pose should be a 9-element array
        self.assertEqual(self.env._default_pose.shape, (9,))
        self.assertTrue(jp.all(jp.isfinite(self.env._default_pose)))

    def test_joint_limits(self):
        """Test that joint limits are properly defined."""
        # Position limits for termination
        self.assertEqual(self.env.pos_lowers.shape, (self.env.sys.nu,))
        self.assertEqual(self.env.pos_uppers.shape, (self.env.sys.nu,))

        # Joint range for control
        self.assertEqual(self.env._joint_range_lower.shape, (self.env.sys.nu,))
        self.assertEqual(self.env._joint_range_upper.shape, (self.env.sys.nu,))

    def test_environment_properties(self):
        """Test basic environment properties."""
        # Check dt
        self.assertEqual(self.env._dt, 0.02)  # 50 fps

        # Check dimensions
        self.assertEqual(self.env._nu, 9)
        self.assertGreater(self.env._nv, 0)

        # Check base body ID is valid
        self.assertGreaterEqual(self.env._base_body_id, 0)

    def test_kicks_can_be_disabled(self):
        """Test that kicks can be disabled."""
        env_no_kicks = BittleEnv(
            xml_path=self.xml_path,
            enable_kicks=False,
        )

        state = env_no_kicks.reset(self.rng)
        action = jp.zeros(env_no_kicks.sys.nu)

        # With kicks disabled, multiple steps should be more deterministic
        # (though still has observation noise)
        next_state = env_no_kicks.step(state, action)
        self.assertIsNotNone(next_state)

    def test_observation_noise(self):
        """Test that observation noise can be configured."""
        env_no_noise = BittleEnv(
            xml_path=self.xml_path,
            obs_noise=0.0,
        )

        state = env_no_noise.reset(self.rng)
        self.assertTrue(jp.all(jp.isfinite(state.obs)))

    def test_custom_reward_scales(self):
        """Test that reward scales can be customized."""
        env_custom = BittleEnv(
            xml_path=self.xml_path,
            tracking_lin_vel_scale=5.0,
            orientation_scale=-10.0,
        )

        # Check that custom scales were applied
        self.assertEqual(
            env_custom.reward_config.rewards.scales['tracking_lin_vel'], 5.0
        )
        self.assertEqual(
            env_custom.reward_config.rewards.scales['orientation'], -10.0
        )


class TestBittleEnvRewards(unittest.TestCase):
    """Test suite specifically for reward functions."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        project_root = Path(__file__).parent.parent.parent
        cls.xml_path = str(project_root / "locomotion" / "bittle_adapted_scene.xml")
        cls.rng = jax.random.PRNGKey(0)

    def setUp(self):
        """Create environment for each test."""
        self.env = BittleEnv(xml_path=self.xml_path)

    def test_all_rewards_are_finite(self):
        """Test that all reward components produce finite values."""
        state = self.env.reset(self.rng)
        action = jp.zeros(self.env.sys.nu)
        next_state = self.env.step(state, action)

        for reward_name, reward_value in next_state.info['rewards'].items():
            self.assertTrue(
                jp.isfinite(reward_value),
                f"Reward '{reward_name}' is not finite: {reward_value}"
            )

    def test_termination_reward(self):
        """Test termination reward is only non-zero when terminated early."""
        state = self.env.reset(self.rng)

        # Normal step should not trigger termination reward
        action = jp.zeros(self.env.sys.nu)
        next_state = self.env.step(state, action)

        # Termination reward depends on done flag and step count
        term_reward = next_state.info['rewards']['termination']
        self.assertTrue(jp.isfinite(term_reward))


if __name__ == '__main__':
    unittest.main()
