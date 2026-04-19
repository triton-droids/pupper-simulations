"""Tests for final-only training artifact generation."""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomotion.training.monitor import FINAL_METRICS_FILENAME, FINAL_PLOT_FILENAME, TrainingMonitor


class TrainingMonitorTests(unittest.TestCase):
    def test_finalize_writes_single_final_plot_and_metrics(self) -> None:
        logger = logging.getLogger("training_monitor_test")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            monitor = TrainingMonitor(output_dir, num_timesteps=200_000, logger=logger)

            monitor(
                40_000,
                {
                    "eval/episode_reward": -0.25,
                    "eval/episode_reward_std": 0.05,
                    "eval/episode_reward/upright": 0.1,
                },
            )
            monitor.finalize()

            metrics_path = output_dir / "metrics" / FINAL_METRICS_FILENAME
            plot_path = output_dir / "plots" / FINAL_PLOT_FILENAME

            self.assertTrue(metrics_path.exists())
            self.assertTrue(plot_path.exists())
            self.assertEqual(list((output_dir / "metrics").glob("metrics_step_*.json")), [])
            self.assertEqual(list((output_dir / "plots").glob("progress_step_*.png")), [])
            self.assertEqual(list((output_dir / "videos").glob("*.mp4")), [])

            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics_payload["steps"], [40_000])
            self.assertEqual(metrics_payload["rewards"], [-0.25])
            self.assertEqual(metrics_payload["reward_stds"], [0.05])


if __name__ == "__main__":
    unittest.main()
