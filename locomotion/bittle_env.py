"""
Bittle quadruped environment with relative position control.

The policy outputs joint position offsets relative to a default pose.
"""

from __future__ import annotations

import logging
from typing import Any, List, Sequence

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict

from brax import base
from brax import math
from brax.base import Motion, Transform
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


logger = logging.getLogger(__name__)


def get_config():
    """Return reward configuration for the Bittle quadruped environment."""

    def get_default_rewards_config():
        return config_dict.ConfigDict(
            dict(
                scales=config_dict.ConfigDict(
                    dict(
                        tracking_lin_vel=2.5,
                        tracking_ang_vel=1.5,
                        lin_vel_z=-2.0,
                        ang_vel_xy=-0.05,
                        orientation=-5.0,
                        torques=-0.0002,
                        action_rate=-0.001,
                        joint_acc=-0.0025,
                        stand_still=-0.5,
                        termination=-1.0,
                        feet_air_time=1.0,
                        foot_slip=-0.04,
                        energy=-0.002,
                    )
                ),
                tracking_sigma=0.25,
            )
        )

    return config_dict.ConfigDict(
        dict(
            rewards=get_default_rewards_config(),
        )
    )


class BittleEnv(PipelineEnv):
    """Environment for Bittle quadruped locomotion."""

    def __init__(
        self,
        xml_path: str,
        obs_noise: float = 0.05,
        action_scale: float = 0.5,
        kick_vel: float = 0.05,
        enable_kicks: bool = True,
        log_init_summary: bool = False,
        **kwargs,
    ):
        sys = mjcf.load(xml_path)
        self._dt = 0.02
        sys = sys.tree_replace({"opt.timestep": 0.004})

        sys = sys.replace(
            dof_damping=sys.dof_damping.at[6:].set(5),
        )

        n_frames = kwargs.pop("n_frames", int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self.reward_config = get_config()
        for key, value in kwargs.items():
            if key.endswith("_scale"):
                self.reward_config.rewards.scales[key[:-6]] = value

        self._base_body_id = mujoco.mj_name2id(
            sys.mj_model,
            mujoco.mjtObj.mjOBJ_BODY.value,
            "base",
        )

        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._enable_kicks = enable_kicks
        self._nu = sys.nu

        self._q_joint_start = 7
        self._qd_joint_start = 6

        self._default_pose = jp.array(
            [
                -0.6908,
                1.9782,
                0.7222,
                1.9468,
                -0.596904,
                -0.6908,
                1.9782,
                0.7222,
                1.9468,
            ]
        )

        self.pos_lowers = jp.array([-1.5] * sys.nu)
        self.pos_uppers = jp.array([1.5] * sys.nu)

        self._joint_range_lower = jp.full(sys.nu, -3.14159)
        self._joint_range_upper = jp.full(sys.nu, 3.14159)

        lower_leg_names = [
            "servos_rf_1",
            "servos_rr_1",
            "servos_lf_1",
            "servos_lr_1",
        ]

        lower_leg_body_ids = []
        for name in lower_leg_names:
            try:
                body_id = mujoco.mj_name2id(
                    sys.mj_model,
                    mujoco.mjtObj.mjOBJ_BODY.value,
                    name,
                )
            except ValueError:
                logger.warning("Lower leg body '%s' was not found in the MJCF model", name)
                continue

            lower_leg_body_ids.append(body_id)

        self._lower_leg_body_id = (
            np.asarray(lower_leg_body_ids, dtype=np.int32)
            if lower_leg_body_ids
            else np.array([], dtype=np.int32)
        )
        self._foot_radius = 0.015

        if log_init_summary:
            logger.info(
                "Initialized BittleEnv: actuators=%s nq=%s nv=%s base_body_id=%s action_scale=+/-%.3f",
                sys.nu,
                sys.nq,
                sys.nv,
                self._base_body_id,
                self._action_scale,
            )
            logger.info(
                "Joint slices: q=[%s:%s] qd=[%s:%s] lower_leg_bodies=%s",
                self._q_joint_start,
                self._q_joint_start + sys.nu,
                self._qd_joint_start,
                self._qd_joint_start + sys.nu,
                self._lower_leg_body_id.tolist(),
            )

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Sample a velocity command."""
        lin_vel_x = [-0.3, 0.6]
        lin_vel_y = [-0.3, 0.3]
        ang_vel_yaw = [-0.5, 0.5]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        return jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        qpos = jp.zeros(self.sys.nq)
        qpos = qpos.at[0:3].set(jp.array([0.0, 0.0, 0.075]))
        qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))
        qpos = qpos.at[self._q_joint_start :].set(self._default_pose)

        qvel = jp.zeros(self.sys.nv)
        pipeline_state = self.pipeline_init(qpos, qvel)

        state_info = {
            "rng": rng,
            "last_act": jp.zeros(self._nu),
            "last_joint_vel": jp.zeros(self._nu),
            "command": self.sample_command(key),
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4),
            "rewards": {key: 0.0 for key in self.reward_config.rewards.scales.keys()},
            "step": 0,
        }

        obs_size = 1 + 3 + 3 + self._nu + self._nu + self._nu
        obs_history = jp.zeros(15 * obs_size)
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward = jp.float32(0.0)
        done = jp.float32(0.0)
        metrics = {"total_dist": 0.0}
        for key in state_info["rewards"]:
            metrics[key] = state_info["rewards"][key]

        return State(pipeline_state, obs, reward, done, metrics, state_info)

    def step(self, state: State, action: jax.Array) -> State:
        """
        Step the environment.

        Args:
            action: Joint position offsets normalized to [-1, 1]. The values
                are scaled to [-action_scale, action_scale] radians and added
                to the default pose to form target joint positions.
        """
        rng, cmd_rng, kick_rng = jax.random.split(state.info["rng"], 3)

        kick_vel = jp.where(
            self._enable_kicks & (jax.random.uniform(kick_rng) < 0.001),
            jax.random.uniform(
                kick_rng, (3,), minval=-self._kick_vel, maxval=self._kick_vel
            ),
            jp.zeros(3),
        )

        position_offsets = action * self._action_scale
        target_positions = self._default_pose + position_offsets
        target_positions = jp.clip(
            target_positions, self._joint_range_lower, self._joint_range_upper
        )

        pipeline_state = self.pipeline_step(state.pipeline_state, target_positions)
        pipeline_state = pipeline_state.replace(
            qd=pipeline_state.qd.at[:3].set(pipeline_state.qd[:3] + kick_vel)
        )

        x, xd = pipeline_state.x, pipeline_state.xd
        joint_angles = pipeline_state.q[self._q_joint_start :]
        joint_vel = pipeline_state.qd[self._qd_joint_start :]

        if len(self._lower_leg_body_id) > 0:
            lower_leg_pos = pipeline_state.xpos[self._lower_leg_body_id]
            foot_z = lower_leg_pos[:, 2] - 0.06
            contact = foot_z < self._foot_radius
        else:
            contact = jp.ones(4, dtype=bool)

        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt
        state.info["feet_air_time"] += self.dt

        up_vec = math.rotate(jp.array([0, 0, 1]), x.rot[self._base_body_id])
        done = up_vec[2] < 0.5
        done |= pipeline_state.x.pos[self._base_body_id, 2] < 0.02
        done |= jp.any(joint_angles < self.pos_lowers - 0.3)
        done |= jp.any(joint_angles > self.pos_uppers + 0.3)

        rewards = {
            "tracking_lin_vel": self._reward_tracking_lin_vel(state.info["command"], x, xd),
            "tracking_ang_vel": self._reward_tracking_ang_vel(state.info["command"], x, xd),
            "lin_vel_z": self._reward_lin_vel_z(xd),
            "ang_vel_xy": self._reward_ang_vel_xy(xd),
            "orientation": self._reward_orientation(x),
            "torques": self._reward_torques(pipeline_state.qfrc_actuator),
            "action_rate": self._reward_action_rate(action, state.info["last_act"]),
            "joint_acc": self._reward_joint_acc(joint_vel, state.info["last_joint_vel"]),
            "stand_still": self._reward_stand_still(state.info["command"], joint_vel),
            "feet_air_time": self._reward_feet_air_time(
                state.info["feet_air_time"],
                first_contact,
                state.info["command"],
            ),
            "foot_slip": self._reward_foot_slip(pipeline_state, contact_filt),
            "termination": self._reward_termination(done, state.info["step"]),
            "energy": self._reward_energy(joint_vel, pipeline_state.qfrc_actuator),
        }

        rewards = {
            key: value * self.reward_config.rewards.scales[key]
            for key, value in rewards.items()
        }

        reward = jp.clip(sum(rewards.values()) * self.dt, -10.0, 10.0)
        reward = jp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        state.info["last_act"] = action
        state.info["last_joint_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt
        state.info["last_contact"] = contact
        state.info["rewards"] = rewards
        state.info["step"] += 1
        state.info["rng"] = rng

        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500), 0, state.info["step"]
        )

        state.metrics["total_dist"] = (
            jp.linalg.norm(xd.vel[self._base_body_id, :2]) * self.dt
        )
        state.metrics.update(state.info["rewards"])

        done = jp.float32(done)
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
        )

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        obs_history: jax.Array,
    ) -> jax.Array:
        """Get observation from the current state."""
        inv_base_rot = math.quat_inv(pipeline_state.x.rot[self._base_body_id])
        local_rpyrate = math.rotate(
            pipeline_state.xd.ang[self._base_body_id], inv_base_rot
        )

        joint_angles = pipeline_state.q[self._q_joint_start :]
        joint_vels = pipeline_state.qd[self._qd_joint_start :]

        obs = jp.concatenate(
            [
                jp.array([local_rpyrate[2]]) * 0.25,
                math.rotate(jp.array([0, 0, -1]), inv_base_rot),
                state_info["command"] * jp.array([2.0, 2.0, 0.25]),
                joint_angles - self._default_pose,
                joint_vels * 0.05,
                state_info["last_act"],
            ]
        )

        obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
            state_info["rng"], obs.shape, minval=-1, maxval=1
        )

        return jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        """Penalize vertical velocity of the base."""
        return jp.square(xd.vel[self._base_body_id, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        """Penalize roll and pitch rates."""
        return jp.sum(jp.square(xd.ang[self._base_body_id, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        """Penalize non-upright orientation."""
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[self._base_body_id])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        """Penalize high torques."""
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        """Penalize rapid action changes."""
        return jp.sum(jp.square(act - last_act))

    def _reward_joint_acc(
        self, joint_vel: jax.Array, last_joint_vel: jax.Array
    ) -> jax.Array:
        """Penalize joint accelerations."""
        joint_acc = (joint_vel - last_joint_vel) / self.dt
        return jp.sum(jp.square(joint_acc))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        """Reward matching the linear velocity command."""
        local_vel = math.rotate(
            xd.vel[self._base_body_id],
            math.quat_inv(x.rot[self._base_body_id]),
        )
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(-lin_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        """Reward matching the angular velocity command."""
        base_ang_vel = math.rotate(
            xd.ang[self._base_body_id],
            math.quat_inv(x.rot[self._base_body_id]),
        )
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        """Reward appropriate swing durations."""
        reward_air_time = jp.sum((air_time - 0.15) * first_contact)
        reward_air_time *= math.normalize(commands[:2])[1] > 0.05
        return reward_air_time

    def _reward_stand_still(
        self, commands: jax.Array, joint_vel: jax.Array
    ) -> jax.Array:
        """Penalize motion when the command is near zero."""
        return jp.sum(jp.abs(joint_vel)) * (math.normalize(commands[:2])[1] < 0.1)

    def _reward_foot_slip(
        self, pipeline_state: base.State, contact_filt: jax.Array
    ) -> jax.Array:
        """Penalize foot slipping while in contact."""
        if len(self._lower_leg_body_id) == 0:
            return 0.0

        lower_leg_vel = pipeline_state.xd.vel[self._lower_leg_body_id]
        vel_xy = lower_leg_vel[:, :2]
        vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
        return jp.sum(vel_xy_norm_sq * contact_filt)

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        """Penalize early termination."""
        return done & (step < 500)

    def _reward_energy(
        self, qvel: jax.Array, qfrc_actuator: jax.Array
    ) -> jax.Array:
        """Penalize energy consumption."""
        actuator_forces = qfrc_actuator[self._qd_joint_start :]
        return jp.sum(jp.abs(qvel) * jp.abs(actuator_forces))

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)
