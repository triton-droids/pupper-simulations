"""Tests for sweep output path defaults."""

from __future__ import annotations

import argparse
import json
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

    def test_prepare_trial_dir_uses_short_name_and_writes_parameters_txt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            trial_dir = output_dir / hparam_sweep._build_trial_dir_name(1)

            hparam_sweep._prepare_trial_dir(
                trial_dir,
                {
                    "batch_size": 24,
                    "episode_length": 200,
                    "num_envs": 24,
                },
                {
                    "action_scale": 0.35,
                    "dance_amplitude": 0.85,
                    "enable_kicks": False,
                },
                overwrite=False,
            )

            parameters_path = trial_dir / hparam_sweep.PARAMETERS_FILENAME

            self.assertEqual(trial_dir.name, "trial_001")
            self.assertTrue(parameters_path.exists())

            contents = parameters_path.read_text(encoding="utf-8")
            self.assertIn("train parameters:", contents)
            self.assertIn("batch_size=24", contents)
            self.assertIn("episode_length=200", contents)
            self.assertIn("num_envs=24", contents)
            self.assertIn("task parameters:", contents)
            self.assertIn("action_scale=0.35", contents)
            self.assertIn("dance_amplitude=0.85", contents)
            self.assertIn("enable_kicks=False", contents)

    def test_build_trial_command_includes_require_gpu_flag(self) -> None:
        command = hparam_sweep._build_trial_command(
            trial_id=3,
            xml_path="scene.xml",
            task="dance",
            trial_dir=Path("trial_003"),
            test_mode=False,
            training_overrides={"batch_size": 24},
            task_overrides={"action_scale": 0.35},
            metric="best_reward",
            require_gpu=True,
        )

        self.assertIn("--metric", command)
        self.assertIn("best_reward", command)
        self.assertIn("--require_gpu", command)

    def test_validate_gpu_probe_rejects_cpu_only_fallback(self) -> None:
        probe = hparam_sweep.JaxDeviceProbe(
            probe_target="CUDA_VISIBLE_DEVICES=2",
            cuda_visible_devices="2",
            returncode=0,
            platforms=("cpu",),
            devices=("TFRT_CPU_0",),
            stderr_tail=("Falling back to cpu.",),
        )

        failure = hparam_sweep._validate_gpu_probe(probe)

        self.assertIsNotNone(failure)
        self.assertIn("did not expose a GPU-backed device", failure)
        self.assertIn("Falling back to cpu.", failure)

    def test_run_gpu_preflight_checks_probes_each_parallel_slot(self) -> None:
        with mock.patch.object(
            hparam_sweep,
            "_probe_jax_devices",
            side_effect=[
                hparam_sweep.JaxDeviceProbe(
                    probe_target="CUDA_VISIBLE_DEVICES=0",
                    cuda_visible_devices="0",
                    returncode=0,
                    platforms=("gpu",),
                    devices=("CudaDevice(id=0)",),
                    stderr_tail=(),
                ),
                hparam_sweep.JaxDeviceProbe(
                    probe_target="CUDA_VISIBLE_DEVICES=1",
                    cuda_visible_devices="1",
                    returncode=0,
                    platforms=("cpu",),
                    devices=("TFRT_CPU_0",),
                    stderr_tail=("Falling back to cpu.",),
                ),
            ],
        ):
            failures = hparam_sweep._run_gpu_preflight_checks(
                require_gpu=True,
                max_concurrent_trials=2,
                gpu_slots=["0", "1", "2"],
            )

        self.assertEqual(len(failures), 1)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", failures[0])

    def test_persist_trial_record_updates_results_leaderboard_and_best(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            first = hparam_sweep.TrialRecord(
                trial_id=2,
                trial_dir=str(output_dir / "trial_002"),
                training_overrides={"num_timesteps": 175000},
                task_overrides={"action_scale": 0.3},
                combined_overrides={"num_timesteps": 175000, "action_scale": 0.3},
                success=True,
                exit_code=0,
                metric="best_reward",
                score=16.7,
                summary={"best_reward": 16.7},
                error=None,
            )
            second = hparam_sweep.TrialRecord(
                trial_id=1,
                trial_dir=str(output_dir / "trial_001"),
                training_overrides={"num_timesteps": 175000},
                task_overrides={"action_scale": 0.32},
                combined_overrides={"num_timesteps": 175000, "action_scale": 0.32},
                success=True,
                exit_code=0,
                metric="best_reward",
                score=12.9,
                summary={"best_reward": 12.9},
                error=None,
            )

            hparam_sweep._initialize_progress_artifacts(output_dir)
            hparam_sweep._persist_trial_record(output_dir, first)
            hparam_sweep._persist_trial_record(output_dir, second)

            rows = hparam_sweep._read_jsonl(output_dir / hparam_sweep.RESULTS_JSONL_FILENAME)
            leaderboard = json.loads(
                (output_dir / hparam_sweep.LEADERBOARD_FILENAME).read_text(encoding="utf-8")
            )
            best = json.loads(
                (output_dir / hparam_sweep.BEST_TRIAL_FILENAME).read_text(encoding="utf-8")
            )

        self.assertEqual([row["trial_id"] for row in rows], [1, 2])
        self.assertEqual([row["trial_id"] for row in leaderboard], [2, 1])
        self.assertEqual(best["trial_id"], 2)


if __name__ == "__main__":
    unittest.main()
