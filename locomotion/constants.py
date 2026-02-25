"""
Single source of truth for Bittle robot parameters.

Shared between the Brax training environment (bittle_env.py) and the
MuJoCo teleop visualizer (teleop.py).  Pure Python only — no JAX or
numpy imports.  Each consumer converts to its preferred array type.
"""

# Default standing pose (radians) — matches the "home" keyframe in
# bittle_adapted_scene.xml.  Order: shrfs, shrft, shrrs, shrrt, neck,
# shlfs, shlft, shlrs, shlrt.
DEFAULT_POSE = [
    -0.6908,    # shrfs
    1.9782,     # shrft
    0.7222,     # shrrs
    1.9468,     # shrrt
    -0.596904,  # neck
    -0.6908,    # shlfs
    1.9782,     # shlft
    0.7222,     # shlrs
    1.9468,     # shlrt
]

NUM_ACTUATORS = 9
ACTION_SCALE = 0.5          # radians, ±π/2 position offset range

OBS_SIZE = 34               # 1 + 3 + 3 + 9 + 9 + 9
HISTORY_LEN = 15
TOTAL_OBS = OBS_SIZE * HISTORY_LEN  # 510

# Physics timing
PHYSICS_TIMESTEP = 0.004    # seconds
CONTROL_DT = 0.02           # seconds (50 Hz)
NSUBSTEPS = 5               # PHYSICS_TIMESTEP * NSUBSTEPS = CONTROL_DT

# Free-joint indices (freejoint = 7 qpos DOFs, 6 qvel DOFs)
Q_JOINT_START = 7           # first actuated joint in qpos
QD_JOINT_START = 6          # first actuated joint in qvel

# Damping applied to actuated joints
JOINT_DAMPING = 5.0

# Initial base position + quaternion [x, y, z, qw, qx, qy, qz]
INIT_QPOS_BASE = [0.0, 0.0, 0.075, 1.0, 0.0, 0.0, 0.0]
