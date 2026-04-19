"""Tests for sweep output path defaults."""

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomotion.paths import OUTPUTS_ROOT, build_numbered_sweep_output_dir
from locomotion.sweeps import hparam_sweep


class HparamSweepPathTests(unittest.TestCase):
    def test_default_sweep_output_dir_uses_numbered_scripts_outputs_path(self) -> None:
        args = argparse.Namespace(base_output_dir=None)

        expected_dir = OUTPUTS_ROOT / "Sweep #42"
        with mock.patch.object(
            hparam_sweep,
            "build_numbered_sweep_output_dir",
            return_value=expected_dir,
        ):
            output_dir = hparam_sweep._build_sweep_output_dir(args)

        self.assertEqual(output_dir, expected_dir)

    def test_build_numbered_sweep_output_dir_increments_counter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            sweep_root = temp_root / "Scripts" / "Outputs"
            counter_path = temp_root / "Scripts" / ".sweep_counter"

            first_dir = build_numbered_sweep_output_dir(
                sweep_root,
                counter_path=counter_path,
            )
            second_dir = build_numbered_sweep_output_dir(
                sweep_root,
                counter_path=counter_path,
            )

        self.assertEqual(first_dir, sweep_root / "Sweep #0")
        self.assertEqual(second_dir, sweep_root / "Sweep #1")

    def test_sweep_combinations_use_trainer_outer_loop_and_task_inner_loop(self) -> None:
        training_trials = [{"batch_size": 32}, {"batch_size": 64}]
        task_trials = [{"action_scale": 0.35}, {"action_scale": 0.4}]

        combinations = hparam_sweep._build_sweep_combinations(training_trials, task_trials)

        self.assertEqual(len(combinations), 4)
        self.assertEqual(combinations[0].training_overrides, {"batch_size": 32})
        self.assertEqual(combinations[0].task_overrides, {"action_scale": 0.35})
        self.assertEqual(combinations[1].training_overrides, {"batch_size": 32})
        self.assertEqual(combinations[1].task_overrides, {"action_scale": 0.4})
        self.assertEqual(combinations[2].training_overrides, {"batch_size": 64})
        self.assertEqual(combinations[2].task_overrides, {"action_scale": 0.35})

    def test_load_task_hparam_trials_uses_default_task_json(self) -> None:
        task_path, task_trials = hparam_sweep._load_task_hparam_trials("dance", None)

        self.assertEqual(task_path.name, "bittle_dance_hparams.json")
        self.assertGreaterEqual(len(task_trials), 1)
        self.assertIsInstance(task_trials[0], dict)


if __name__ == "__main__":
    unittest.main()
