"""Tests for task-specific hyperparameter JSON helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomotion.tasks.bittle_dance_env import load_task_hparam_sweep as load_dance_hparams
from locomotion.tasks.bittle_walk_env import (
    load_task_hparam_sweep as load_walking_hparams,
)


class TaskHyperparameterJsonTests(unittest.TestCase):
    def test_dance_task_hparams_json_loads(self) -> None:
        entries = load_dance_hparams()

        self.assertGreaterEqual(len(entries), 1)
        self.assertIsInstance(entries[0], dict)

    def test_walking_task_hparams_json_loads(self) -> None:
        entries = load_walking_hparams()

        self.assertGreaterEqual(len(entries), 1)
        self.assertIsInstance(entries[0], dict)


if __name__ == "__main__":
    unittest.main()
