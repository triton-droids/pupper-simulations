"""Shared fixtures and path setup for the test suite."""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

# Force JAX to use CPU (no GPU required for tests)
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Ensure MUJOCO_GL is valid for this platform before any module sets it to "egl".
# train.py sets MUJOCO_GL=egl at import time, which is invalid on macOS.
os.environ.pop("MUJOCO_GL", None)

# Pre-import mujoco so it caches with a valid GL setting.
# Later imports (even after train.py sets MUJOCO_GL=egl) use the cached module.
import mujoco  # noqa: E402,F401

# Add project paths so bare imports in locomotion/ work (e.g. train.py imports bittle_env)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_locomotion_dir = os.path.join(_project_root, "locomotion")
for _p in (_project_root, _locomotion_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def fake_policy_params():
    """Brax PPO params tuple with correct shapes for ONNX export.

    Network: 510 -> 256 -> 256 -> 256 -> 256 -> 18  (5 hidden layers)
    Output 18 = 9 action means + 9 log_stds, sliced to first 9.
    """
    normalizer = SimpleNamespace(
        mean=np.zeros(510, dtype=np.float32),
        std=np.ones(510, dtype=np.float32) * 2.0,
    )

    layer_sizes = [
        (510, 256),   # hidden_0
        (256, 256),   # hidden_1
        (256, 256),   # hidden_2
        (256, 256),   # hidden_3
        (256, 18),    # hidden_4 (output: 9 means + 9 log_stds)
    ]
    policy_params = {"params": {}}
    rng = np.random.RandomState(42)
    for i, (fan_in, fan_out) in enumerate(layer_sizes):
        policy_params["params"][f"hidden_{i}"] = {
            "kernel": rng.randn(fan_in, fan_out).astype(np.float32) * 0.01,
            "bias": np.zeros(fan_out, dtype=np.float32),
        }

    value_params = None  # not used in export

    return (normalizer, policy_params, value_params)
