#!/usr/bin/env python3
"""Tests that robot constants are consistent and used by both consumers."""

import ast
import os
import unittest

from locomotion.constants import (
    DEFAULT_POSE,
    NUM_ACTUATORS,
    OBS_SIZE,
    HISTORY_LEN,
    TOTAL_OBS,
    PHYSICS_TIMESTEP,
    CONTROL_DT,
    NSUBSTEPS,
    INIT_QPOS_BASE,
    Q_JOINT_START,
    QD_JOINT_START,
)


class TestConstantValues(unittest.TestCase):
    """Verify internal consistency of constants."""

    def test_default_pose_length(self):
        self.assertEqual(len(DEFAULT_POSE), NUM_ACTUATORS)

    def test_total_obs(self):
        self.assertEqual(TOTAL_OBS, OBS_SIZE * HISTORY_LEN)

    def test_physics_timing(self):
        self.assertAlmostEqual(PHYSICS_TIMESTEP * NSUBSTEPS, CONTROL_DT)

    def test_init_qpos_base_length(self):
        # 3 position + 4 quaternion = 7
        self.assertEqual(len(INIT_QPOS_BASE), Q_JOINT_START)

    def test_qd_joint_start(self):
        # freejoint: 6 velocity DOFs (3 translational + 3 rotational)
        self.assertEqual(QD_JOINT_START, 6)


class TestConsumersImportConstants(unittest.TestCase):
    """Verify that bittle_env.py and teleop.py import from locomotion.constants."""

    @staticmethod
    def _imports_from_constants(filepath):
        """Return True if the file has 'from locomotion.constants import ...'."""
        with open(filepath) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "locomotion.constants" in node.module:
                    return True
                # Also accept relative import '.constants'
                if node.module == "constants" and node.level == 1:
                    return True
        return False

    def test_bittle_env_imports_constants(self):
        path = os.path.join(os.path.dirname(__file__), "..", "locomotion", "bittle_env.py")
        self.assertTrue(
            os.path.exists(path), f"bittle_env.py not found at {path}"
        )
        self.assertTrue(
            self._imports_from_constants(path),
            "bittle_env.py does not import from locomotion.constants",
        )

    def test_teleop_imports_constants(self):
        path = os.path.join(os.path.dirname(__file__), "..", "locomotion", "teleop.py")
        self.assertTrue(
            os.path.exists(path), f"teleop.py not found at {path}"
        )
        self.assertTrue(
            self._imports_from_constants(path),
            "teleop.py does not import from locomotion.constants",
        )


if __name__ == "__main__":
    unittest.main()
