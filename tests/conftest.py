import os
import sys

import numpy as np
import pytest

# Add locomotion/ and project root to sys.path so imports match train.py
# (train.py does `from bittle_env import BittleEnv` from within locomotion/)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_locomotion_dir = os.path.join(_project_root, "locomotion")

for p in (_project_root, _locomotion_dir):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def xml_path():
    return os.path.join(_locomotion_dir, "bittle_adapted_scene.xml")


@pytest.fixture(scope="session")
def bittle_env(xml_path):
    from bittle_env import BittleEnv
    return BittleEnv(xml_path=xml_path)


@pytest.fixture(scope="session")
def brax_sys(bittle_env):
    return bittle_env.sys


@pytest.fixture
def mock_policy_params():
    """Fake (normalizer_params, policy_params, value_params) with correct shapes.

    Network architecture: 510 -> 256 -> 256 -> 256 -> 128 -> 18
    """

    class _NormalizerParams:
        def __init__(self):
            self.mean = np.zeros(510, dtype=np.float32)
            self.std = np.ones(510, dtype=np.float32)

    layer_sizes = [
        (510, 256),  # hidden_0
        (256, 256),  # hidden_1
        (256, 256),  # hidden_2
        (256, 128),  # hidden_3
        (128, 18),   # hidden_4 (output: 9 mean + 9 log_std)
    ]

    policy_params = {"params": {}}
    for i, (fan_in, fan_out) in enumerate(layer_sizes):
        policy_params["params"][f"hidden_{i}"] = {
            "kernel": np.random.randn(fan_in, fan_out).astype(np.float32) * 0.01,
            "bias": np.zeros(fan_out, dtype=np.float32),
        }

    value_params = None
    return (_NormalizerParams(), policy_params, value_params)
