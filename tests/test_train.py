"""Tests for locomotion/train.py (1 test)."""

from unittest.mock import MagicMock, patch
import sys

import pytest

# Import train module (conftest pre-imports mujoco so MUJOCO_GL=egl is harmless)
import train as train_module


def test_main(monkeypatch, fake_policy_params):
    """main() registers env, calls ppo.train with test config, and exports ONNX."""
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--test", "--no-video", "--output", "/tmp/test_policy.onnx",
    ])

    mock_ppo_train = MagicMock(return_value=(MagicMock(), fake_policy_params, {}))
    mock_export = MagicMock()
    mock_register = MagicMock()
    mock_get_env = MagicMock(return_value=MagicMock())

    with patch.object(train_module.ppo, "train", mock_ppo_train), \
         patch.object(train_module, "export_policy_to_onnx", mock_export), \
         patch.object(train_module.envs, "register_environment", mock_register), \
         patch.object(train_module.envs, "get_environment", mock_get_env), \
         patch("os.makedirs"):

        train_module.main()

    mock_register.assert_called_once_with("bittle", train_module.BittleEnv)
    mock_ppo_train.assert_called_once()
    mock_export.assert_called_once_with(fake_policy_params, "/tmp/test_policy.onnx")
