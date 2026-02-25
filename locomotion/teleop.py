#!/usr/bin/env python3
"""
Interactive Bittle locomotion teleop with ONNX policy.

Controls:
  W / S          vx  +-key_value
  A / D          vy  +-key_value
  Left / Right   yaw +-key_value
  Space          zero commands
  R              reset
  Q              quit

Run:  mjpython locomotion/teleop.py
"""

import argparse
import os
import select
import sys
import termios
import time
import tty

import mujoco
import mujoco.viewer
import numpy as np

from .constants import (
    DEFAULT_POSE as _DEFAULT_POSE_LIST,
    ACTION_SCALE,
    OBS_SIZE,
    HISTORY_LEN,
    TOTAL_OBS,
    NUM_ACTUATORS,
    NSUBSTEPS,
    PHYSICS_TIMESTEP,
    INIT_QPOS_BASE as _INIT_QPOS_BASE_LIST,
    Q_JOINT_START,
    QD_JOINT_START,
    JOINT_DAMPING,
)

DEFAULT_POSE = np.array(_DEFAULT_POSE_LIST, dtype=np.float32)
INIT_QPOS_BASE = np.array(_INIT_QPOS_BASE_LIST, dtype=np.float64)

# ---------------------------------------------------------------------------
# Terminal input (non-blocking WASD + arrows)
# ---------------------------------------------------------------------------
class _TerminalInput:
    def __init__(self):
        self.enabled = False
        self._fd = None
        self._old_term = None

    def __enter__(self):
        if not sys.stdin.isatty():
            print("[warn] stdin is not a TTY; key controls disabled.")
            return self
        self._fd = sys.stdin.fileno()
        self._old_term = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self._fd is not None and self._old_term is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
        self.enabled = False

    def _read_arrow_tail(self) -> str:
        seq = ""
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not r:
            return seq
        seq += sys.stdin.read(1)
        if seq != "[":
            return seq
        r, _, _ = select.select([sys.stdin], [], [], 0.0)
        if r:
            seq += sys.stdin.read(1)
        return seq

    def read_keys(self) -> list[str]:
        if not self.enabled:
            return []
        keys: list[str] = []
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not r:
                break
            ch = sys.stdin.read(1)
            if not ch:
                continue
            if ch == "\x1b":
                tail = self._read_arrow_tail()
                if tail == "[D":
                    keys.append("LEFT")
                elif tail == "[C":
                    keys.append("RIGHT")
                continue
            keys.append(ch)
        return keys


# ---------------------------------------------------------------------------
# Observation builder (matches BittleEnv._get_obs, no noise)
# ---------------------------------------------------------------------------
def build_obs(data, base_body_id, command, last_action, obs_history):
    """Build observation vector matching BittleEnv._get_obs (without noise)."""
    # Rotation matrix for base body (3x3, column-major in MuJoCo)
    R = data.xmat[base_body_id].reshape(3, 3)

    # Local angular velocity: R^T @ world_angular_vel
    ang_vel_world = data.cvel[base_body_id][0:3]  # (rotational part of cvel)
    local_ang_vel = R.T @ ang_vel_world

    # Projected gravity: R^T @ [0, 0, -1]
    proj_gravity = R.T @ np.array([0.0, 0.0, -1.0])

    # Joint states (skip freejoint qpos/qvel)
    joint_angles = data.qpos[Q_JOINT_START:Q_JOINT_START + NUM_ACTUATORS].astype(np.float32)
    joint_vels = data.qvel[QD_JOINT_START:QD_JOINT_START + NUM_ACTUATORS].astype(np.float32)

    obs = np.concatenate([
        np.array([local_ang_vel[2]], dtype=np.float32) * 0.25,       # yaw rate (1)
        proj_gravity.astype(np.float32),                              # projected gravity (3)
        command * np.array([2.0, 2.0, 0.25], dtype=np.float32),      # scaled command (3)
        (joint_angles - DEFAULT_POSE),                                # joint offsets (9)
        joint_vels * 0.05,                                            # scaled joint vels (9)
        last_action.astype(np.float32),                               # last action (9)
    ])

    obs = np.clip(obs, -100.0, 100.0)

    # History stacking: roll right by OBS_SIZE, set first OBS_SIZE to new obs
    obs_history = np.roll(obs_history, OBS_SIZE)
    obs_history[:OBS_SIZE] = obs
    return obs_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Bittle ONNX teleop")
    parser.add_argument("--policy", default="locomotion/outputs/policy.onnx")
    parser.add_argument("--xml-path", default="locomotion/bittle_adapted_scene.xml")
    parser.add_argument("--key-value", type=float, default=0.25)
    parser.add_argument("--key-hold", type=float, default=0.12)
    parser.add_argument("--no-policy", action="store_true")
    args = parser.parse_args()

    # Load MuJoCo model and set physics to match training
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    model.opt.timestep = PHYSICS_TIMESTEP
    model.dof_damping[QD_JOINT_START:] = JOINT_DAMPING
    data = mujoco.MjData(model)

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    # Load ONNX policy
    session = None
    if not args.no_policy:
        import onnxruntime as ort
        if not os.path.exists(args.policy):
            print(f"Policy not found: {args.policy}")
            sys.exit(1)
        session = ort.InferenceSession(args.policy)
        print(f"Loaded policy: {args.policy}")
    else:
        print("Running without policy (zero actions).")

    # State
    command = np.zeros(3, dtype=np.float32)
    last_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
    obs_history = np.zeros(TOTAL_OBS, dtype=np.float32)

    active_until = {"w": 0.0, "s": 0.0, "a": 0.0, "d": 0.0, "LEFT": 0.0, "RIGHT": 0.0}

    def reset():
        nonlocal last_action, obs_history
        mujoco.mj_resetData(model, data)
        data.qpos[:Q_JOINT_START] = INIT_QPOS_BASE
        data.qpos[Q_JOINT_START:Q_JOINT_START + NUM_ACTUATORS] = DEFAULT_POSE
        mujoco.mj_forward(model, data)
        last_action = np.zeros(NUM_ACTUATORS, dtype=np.float32)
        obs_history = np.zeros(TOTAL_OBS, dtype=np.float32)
        # Prime history with initial observation
        build_obs(data, base_body_id, command, last_action, obs_history)

    def fmt(cmd):
        return f"vx={cmd[0]:+.3f} vy={cmd[1]:+.3f} yaw={cmd[2]:+.3f}"

    reset()

    print("Controls: W/S=vx  A/D=vy  Left/Right=yaw  Space=zero  R=reset  Q=quit")
    print(f"[cmd] {fmt(command)}")

    with _TerminalInput() as term:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            loop_dt = 1.0 / 50.0
            next_tick = time.perf_counter()
            quit_requested = False

            while viewer.is_running() and not quit_requested:
                now = time.perf_counter()

                # Read keys
                for key in term.read_keys():
                    k = key.lower() if len(key) == 1 else key
                    if k in active_until:
                        active_until[k] = now + args.key_hold
                    elif k == " ":
                        command[:] = 0.0
                        for ak in active_until:
                            active_until[ak] = 0.0
                        print(f"[cmd] {fmt(command)}")
                    elif k == "r":
                        reset()
                        print(f"[reset] {fmt(command)}")
                    elif k == "q":
                        quit_requested = True

                # Update command from active keys
                new_cmd = np.array([
                    args.key_value * (float(active_until["w"] > now) - float(active_until["s"] > now)),
                    args.key_value * (float(active_until["a"] > now) - float(active_until["d"] > now)),
                    args.key_value * (float(active_until["LEFT"] > now) - float(active_until["RIGHT"] > now)),
                ], dtype=np.float32)
                if not np.array_equal(new_cmd, command):
                    command[:] = new_cmd
                    print(f"[cmd] {fmt(command)}")

                # Wait for next control tick
                if now < next_tick:
                    time.sleep(min(0.001, next_tick - now))
                    continue

                # Build observation
                obs_history = build_obs(data, base_body_id, command, last_action, obs_history)

                # Run policy
                if session is not None:
                    obs_input = obs_history.reshape(1, -1)
                    action = session.run(None, {"observation": obs_input})[0].squeeze(0)
                    action = np.clip(action, -1.0, 1.0)
                else:
                    action = np.zeros(NUM_ACTUATORS, dtype=np.float32)

                last_action = action.copy()

                # Set control: default pose + action * scale
                data.ctrl[:] = DEFAULT_POSE + action * ACTION_SCALE

                # Step physics
                mujoco.mj_step(model, data, nstep=NSUBSTEPS)
                viewer.sync()

                next_tick += loop_dt
                if now - next_tick > loop_dt:
                    next_tick = now + loop_dt


if __name__ == "__main__":
    main()
