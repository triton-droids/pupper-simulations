import numpy as np
import mujoco

import sys
import os

# teleop_locomotion.py lives at project root, not in locomotion/
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from teleop_locomotion import (
    build_obs,
    DEFAULT_POSE,
    NUM_ACTUATORS,
    OBS_SIZE,
    TOTAL_OBS,
    INIT_QPOS_BASE,
)


def test_build_obs(xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = 0.004
    model.dof_damping[6:] = 5.0
    data = mujoco.MjData(model)

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    # Set initial pose
    data.qpos[:7] = INIT_QPOS_BASE
    data.qpos[7 : 7 + NUM_ACTUATORS] = DEFAULT_POSE
    mujoco.mj_forward(model, data)

    command = np.zeros(3, dtype=np.float32)
    last_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    obs_history = np.zeros(TOTAL_OBS, dtype=np.float32)

    result = build_obs(data, base_body_id, command, last_action, obs_history)

    assert result.shape == (TOTAL_OBS,)  # (510,)
    # First OBS_SIZE elements should be non-zero (projected gravity alone is non-zero)
    assert np.any(result[:OBS_SIZE] != 0.0)
    # Remaining history slots should still be zero (only one obs inserted)
    assert np.allclose(result[OBS_SIZE:], 0.0)
