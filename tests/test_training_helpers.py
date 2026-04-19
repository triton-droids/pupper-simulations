"""Tests for training helper callbacks."""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomotion.training.helpers import policy_params_callback


class PolicyParamsCallbackTests(unittest.TestCase):
    def test_callback_caches_deterministic_policy_for_final_video(self) -> None:
        logger = logging.getLogger("training_helpers_test")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

        class MonitorStub:
            make_inference_fn_cached = None

        monitor = MonitorStub()
        make_policy_calls: list[tuple[object, bool]] = []

        def make_policy(params: object, deterministic: bool = False) -> dict[str, object]:
            make_policy_calls.append((params, deterministic))
            return {"params": params, "deterministic": deterministic}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_checkpointer = mock.Mock()

            with (
                mock.patch(
                    "locomotion.training.helpers.ocp.PyTreeCheckpointer",
                    return_value=mock_checkpointer,
                ),
                mock.patch(
                    "locomotion.training.helpers.orbax_utils.save_args_from_target",
                    return_value={"mock": "save_args"},
                ),
            ):
                callback = policy_params_callback(output_dir, logger, monitor=monitor)
                callback(123, make_policy, {"weights": [1, 2, 3]})

        self.assertEqual(make_policy_calls, [({"weights": [1, 2, 3]}, True)])
        self.assertEqual(
            monitor.make_inference_fn_cached,
            {"params": {"weights": [1, 2, 3]}, "deterministic": True},
        )
        mock_checkpointer.save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
