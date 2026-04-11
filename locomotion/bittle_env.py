"""
Custom Brax environment for the Bittle quadruped.

High-level environment design
-----------------------------
This environment trains a policy to track commanded planar velocities while
remaining upright, efficient, and smooth.

Key design choices:

- action space:
  The policy outputs joint position offsets relative to a learned default pose.
- observation space:
  The observation includes base orientation cues, the commanded velocity, joint
  states, and a short history stack.
- reward:
  The reward mixes command tracking with penalties for instability, jerky
  motion, slipping feet, excess torque, and early termination.
- robustness:
  Optional random kicks make the policy recover from small disturbances.

The implementation below keeps the behavior the same as before, but organizes
the logic into smaller helpers so the flow of ``reset()`` and ``step()`` is
easier to understand.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

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


CONTROL_DT = 0.02
SIMULATION_TIMESTEP = 0.004
JOINT_DAMPING = 5

COMMAND_RESAMPLE_STEPS = 500
KICK_PROBABILITY = 0.001
OBSERVATION_HISTORY_LENGTH = 15

BASE_POSITION_SLICE = slice(0, 3)
BASE_ROTATION_SLICE = slice(3, 7)
BASE_LINEAR_VELOCITY_SLICE = slice(0, 3)
FREEJOINT_Q_START = 7
FREEJOINT_QD_START = 6

BASE_BODY_NAME = "base"
LOWER_LEG_BODY_NAMES = (
    "servos_rf_1",
    "servos_rr_1",
    "servos_lf_1",
    "servos_lr_1",
)

DEFAULT_BASE_POSITION = jp.array([0.0, 0.0, 0.075])
DEFAULT_BASE_ROTATION = jp.array([1.0, 0.0, 0.0, 0.0])
DEFAULT_POSE = jp.array(
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

TERMINATION_POSITION_LIMIT = 1.5
TERMINATION_MARGIN = 0.3
CONTROL_RANGE = 3.14159
MIN_BASE_HEIGHT = 0.02
UPRIGHT_THRESHOLD = 0.5

FOOT_HEIGHT_OFFSET = 0.06
FOOT_CONTACT_RADIUS = 0.015

COMMAND_RANGE_X = (-0.3, 0.6)
COMMAND_RANGE_Y = (-0.3, 0.3)
COMMAND_RANGE_YAW = (-0.5, 0.5)


def build_reward_config() -> config_dict.ConfigDict:
    """Build the reward-scale configuration used by the environment."""
    return config_dict.ConfigDict(
        dict(
            rewards=config_dict.ConfigDict(
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
        )
    )


def get_config() -> config_dict.ConfigDict:
    """Backward-compatible wrapper for older callers that import ``get_config``."""
    return build_reward_config()


def _find_body_ids(mj_model: mujoco.MjModel, body_names: Sequence[str]) -> np.ndarray:
    """Resolve a list of MuJoCo body names into numeric body ids."""
    body_ids: list[int] = []

    for body_name in body_names:
        try:
            body_id = mujoco.mj_name2id(
                mj_model,
                mujoco.mjtObj.mjOBJ_BODY.value,
                body_name,
            )
        except ValueError:
            logger.warning("Lower leg body '%s' was not found in the MJCF model", body_name)
            continue

        body_ids.append(body_id)

    return np.asarray(body_ids, dtype=np.int32)


class BittleEnv(PipelineEnv):
    """
    Brax environment for Bittle locomotion with relative position control.

    The policy outputs one value per actuator in ``[-1, 1]``. Those values are
    scaled and added to ``DEFAULT_POSE`` to produce the target joint positions
    used by the MuJoCo position actuators.
    """

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
        sys = sys.tree_replace({"opt.timestep": SIMULATION_TIMESTEP})
        sys = sys.replace(dof_damping=sys.dof_damping.at[FREEJOINT_QD_START:].set(JOINT_DAMPING))

        self._dt = CONTROL_DT
        n_frames = kwargs.pop("n_frames", int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self.reward_config = build_reward_config()
        for key, value in kwargs.items():
            if key.endswith("_scale"):
                self.reward_config.rewards.scales[key[:-6]] = value

        self._base_body_id = mujoco.mj_name2id(
            sys.mj_model,
            mujoco.mjtObj.mjOBJ_BODY.value,
            BASE_BODY_NAME,
        )

        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._enable_kicks = enable_kicks
        self._nu = sys.nu

        self._q_joint_start = FREEJOINT_Q_START
        self._qd_joint_start = FREEJOINT_QD_START
        self._default_pose = DEFAULT_POSE
        self._observation_size = 1 + 3 + 3 + self._nu + self._nu + self._nu

        self.pos_lowers = jp.full(sys.nu, -TERMINATION_POSITION_LIMIT)
        self.pos_uppers = jp.full(sys.nu, TERMINATION_POSITION_LIMIT)
        self._joint_range_lower = jp.full(sys.nu, -CONTROL_RANGE)
        self._joint_range_upper = jp.full(sys.nu, CONTROL_RANGE)

        self._lower_leg_body_id = _find_body_ids(sys.mj_model, LOWER_LEG_BODY_NAMES)
        self._foot_radius = FOOT_CONTACT_RADIUS

        if log_init_summary:
            self._log_initialization_summary()

    def _log_initialization_summary(self) -> None:
        """Log one concise environment summary for debugging."""
        logger.info(
            "Initialized BittleEnv: actuators=%s nq=%s nv=%s base_body_id=%s action_scale=+/-%.3f",
            self.sys.nu,
            self.sys.nq,
            self.sys.nv,
            self._base_body_id,
            self._action_scale,
        )
        logger.info(
            "Joint slices: q=[%s:%s] qd=[%s:%s] lower_leg_bodies=%s",
            self._q_joint_start,
            self._q_joint_start + self.sys.nu,
            self._qd_joint_start,
            self._qd_joint_start + self.sys.nu,
            self._lower_leg_body_id.tolist(),
        )

    def sample_command(self, rng: jax.Array) -> jax.Array:
        """Sample the target planar velocity command for the next episode chunk."""
        key_x, key_y, key_yaw = jax.random.split(rng, 3)
        command_x = jax.random.uniform(
            key_x,
            (),
            minval=COMMAND_RANGE_X[0],
            maxval=COMMAND_RANGE_X[1],
        )
        command_y = jax.random.uniform(
            key_y,
            (),
            minval=COMMAND_RANGE_Y[0],
            maxval=COMMAND_RANGE_Y[1],
        )
        command_yaw = jax.random.uniform(
            key_yaw,
            (),
            minval=COMMAND_RANGE_YAW[0],
            maxval=COMMAND_RANGE_YAW[1],
        )
        return jp.array([command_x, command_y, command_yaw])

    def _make_initial_state_info(self, rng: jax.Array, command: jax.Array) -> dict[str, Any]:
        """Build the mutable info dictionary carried by Brax ``State``."""
        return {
            "rng": rng,
            "last_act": jp.zeros(self._nu),
            "last_joint_vel": jp.zeros(self._nu),
            "command": command,
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4),
            "rewards": {
                reward_name: 0.0
                for reward_name in self.reward_config.rewards.scales.keys()
            },
            "step": 0,
        }

    def _build_initial_pipeline_state(self) -> base.State:
        """Construct the initial MuJoCo state for a new episode."""
        qpos = jp.zeros(self.sys.nq)
        qpos = qpos.at[BASE_POSITION_SLICE].set(DEFAULT_BASE_POSITION)
        qpos = qpos.at[BASE_ROTATION_SLICE].set(DEFAULT_BASE_ROTATION)
        qpos = qpos.at[self._q_joint_start :].set(self._default_pose)

        qvel = jp.zeros(self.sys.nv)
        return self.pipeline_init(qpos, qvel)

    def reset(self, rng: jax.Array) -> State:
        """Reset the environment and return the initial Brax ``State``."""
        rng, command_rng = jax.random.split(rng)
        pipeline_state = self._build_initial_pipeline_state()
        state_info = self._make_initial_state_info(rng, self.sample_command(command_rng))

        obs_history = jp.zeros(OBSERVATION_HISTORY_LENGTH * self._observation_size)
        obs = self._get_obs(pipeline_state, state_info, obs_history)

        metrics = {"total_dist": 0.0}
        metrics.update(state_info["rewards"])

        return State(
            pipeline_state,
            obs,
            jp.float32(0.0),
            jp.float32(0.0),
            metrics,
            state_info,
        )

    def _sample_kick_velocity(self, rng: jax.Array) -> jax.Array:
        """Sample a small random kick to the base velocity."""
        return jp.where(
            self._enable_kicks & (jax.random.uniform(rng) < KICK_PROBABILITY),
            jax.random.uniform(
                rng,
                (3,),
                minval=-self._kick_vel,
                maxval=self._kick_vel,
            ),
            jp.zeros(3),
        )

    def _estimate_foot_contact(self, pipeline_state: base.State) -> jax.Array:
        """Approximate which feet are in contact using lower-leg body heights."""
        if len(self._lower_leg_body_id) == 0:
            return jp.ones(4, dtype=bool)

        lower_leg_pos = pipeline_state.xpos[self._lower_leg_body_id]
        foot_height = lower_leg_pos[:, 2] - FOOT_HEIGHT_OFFSET
        return foot_height < self._foot_radius

    def _compute_termination(
        self,
        x: Transform,
        pipeline_state: base.State,
        joint_angles: jax.Array,
    ) -> jax.Array:
        """Apply early termination checks for falls and extreme joint angles."""
        up_vec = math.rotate(jp.array([0, 0, 1]), x.rot[self._base_body_id])

        done = up_vec[2] < UPRIGHT_THRESHOLD
        done |= pipeline_state.x.pos[self._base_body_id, 2] < MIN_BASE_HEIGHT
        done |= jp.any(joint_angles < self.pos_lowers - TERMINATION_MARGIN)
        done |= jp.any(joint_angles > self.pos_uppers + TERMINATION_MARGIN)
        return done

    def _compute_reward_terms(
        self,
        *,
        command: jax.Array,
        x: Transform,
        xd: Motion,
        action: jax.Array,
        last_action: jax.Array,
        joint_vel: jax.Array,
        last_joint_vel: jax.Array,
        feet_air_time: jax.Array,
        first_contact: jax.Array,
        contact_filt: jax.Array,
        pipeline_state: base.State,
        done: jax.Array,
        step: jax.Array,
    ) -> dict[str, jax.Array]:
        """Compute the unscaled reward components for the current step."""
        return {
            "tracking_lin_vel": self._reward_tracking_lin_vel(command, x, xd),
            "tracking_ang_vel": self._reward_tracking_ang_vel(command, x, xd),
            "lin_vel_z": self._reward_lin_vel_z(xd),
            "ang_vel_xy": self._reward_ang_vel_xy(xd),
            "orientation": self._reward_orientation(x),
            "torques": self._reward_torques(pipeline_state.qfrc_actuator),
            "action_rate": self._reward_action_rate(action, last_action),
            "joint_acc": self._reward_joint_acc(joint_vel, last_joint_vel),
            "stand_still": self._reward_stand_still(command, joint_vel),
            "feet_air_time": self._reward_feet_air_time(
                feet_air_time,
                first_contact,
                command,
            ),
            "foot_slip": self._reward_foot_slip(pipeline_state, contact_filt),
            "termination": self._reward_termination(done, step),
            "energy": self._reward_energy(joint_vel, pipeline_state.qfrc_actuator),
        }

    def step(self, state: State, action: jax.Array) -> State:
        """
        Advance the environment by one control step.

        The action is interpreted as a joint offset in ``[-1, 1]``. That offset
        is scaled, added to the default pose, and sent directly to the MuJoCo
        position actuators.
        """
        rng, command_rng, kick_rng = jax.random.split(state.info["rng"], 3)

        position_offsets = action * self._action_scale
        target_positions = jp.clip(
            self._default_pose + position_offsets,
            self._joint_range_lower,
            self._joint_range_upper,
        )

        pipeline_state = self.pipeline_step(state.pipeline_state, target_positions)
        pipeline_state = pipeline_state.replace(
            qd=pipeline_state.qd.at[BASE_LINEAR_VELOCITY_SLICE].set(
                pipeline_state.qd[BASE_LINEAR_VELOCITY_SLICE] + self._sample_kick_velocity(kick_rng)
            )
        )

        x, xd = pipeline_state.x, pipeline_state.xd
        joint_angles = pipeline_state.q[self._q_joint_start :]
        joint_vel = pipeline_state.qd[self._qd_joint_start :]

        contact = self._estimate_foot_contact(pipeline_state)
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt
        state.info["feet_air_time"] += self.dt

        done = self._compute_termination(x, pipeline_state, joint_angles)

        reward_terms = self._compute_reward_terms(
            command=state.info["command"],
            x=x,
            xd=xd,
            action=action,
            last_action=state.info["last_act"],
            joint_vel=joint_vel,
            last_joint_vel=state.info["last_joint_vel"],
            feet_air_time=state.info["feet_air_time"],
            first_contact=first_contact,
            contact_filt=contact_filt,
            pipeline_state=pipeline_state,
            done=done,
            step=state.info["step"],
        )
        scaled_rewards = {
            key: value * self.reward_config.rewards.scales[key]
            for key, value in reward_terms.items()
        }

        reward = jp.clip(sum(scaled_rewards.values()) * self.dt, -10.0, 10.0)
        reward = jp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        state.info["last_act"] = action
        state.info["last_joint_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt
        state.info["last_contact"] = contact
        state.info["rewards"] = scaled_rewards
        state.info["step"] += 1
        state.info["rng"] = rng

        state.info["command"] = jp.where(
            state.info["step"] > COMMAND_RESAMPLE_STEPS,
            self.sample_command(command_rng),
            state.info["command"],
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > COMMAND_RESAMPLE_STEPS),
            0,
            state.info["step"],
        )

        state.metrics["total_dist"] = (
            jp.linalg.norm(xd.vel[self._base_body_id, :2]) * self.dt
        )
        state.metrics.update(state.info["rewards"])

        obs = self._get_obs(pipeline_state, state.info, state.obs)
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=jp.float32(done),
        )

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        obs_history: jax.Array,
    ) -> jax.Array:
        """Assemble the stacked observation vector used by the policy."""
        inv_base_rot = math.quat_inv(pipeline_state.x.rot[self._base_body_id])
        local_rpyrate = math.rotate(
            pipeline_state.xd.ang[self._base_body_id],
            inv_base_rot,
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
            state_info["rng"],
            obs.shape,
            minval=-1,
            maxval=1,
        )

        return jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        """Penalize vertical velocity of the base."""
        return jp.square(xd.vel[self._base_body_id, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        """Penalize roll and pitch rates."""
        return jp.sum(jp.square(xd.ang[self._base_body_id, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        """Penalize tilting away from the world up direction."""
        up = jp.array([0.0, 0.0, 1.0])
        rotated_up = math.rotate(up, x.rot[self._base_body_id])
        return jp.sum(jp.square(rotated_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        """Penalize large actuator efforts."""
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        """Penalize large changes between consecutive actions."""
        return jp.sum(jp.square(act - last_act))

    def _reward_joint_acc(
        self,
        joint_vel: jax.Array,
        last_joint_vel: jax.Array,
    ) -> jax.Array:
        """Penalize abrupt joint accelerations."""
        joint_acc = (joint_vel - last_joint_vel) / self.dt
        return jp.sum(jp.square(joint_acc))

    def _reward_tracking_lin_vel(
        self,
        commands: jax.Array,
        x: Transform,
        xd: Motion,
    ) -> jax.Array:
        """Reward matching the commanded linear velocity."""
        local_vel = math.rotate(
            xd.vel[self._base_body_id],
            math.quat_inv(x.rot[self._base_body_id]),
        )
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(-lin_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(
        self,
        commands: jax.Array,
        x: Transform,
        xd: Motion,
    ) -> jax.Array:
        """Reward matching the commanded yaw rate."""
        base_ang_vel = math.rotate(
            xd.ang[self._base_body_id],
            math.quat_inv(x.rot[self._base_body_id]),
        )
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_feet_air_time(
        self,
        air_time: jax.Array,
        first_contact: jax.Array,
        commands: jax.Array,
    ) -> jax.Array:
        """Reward swing durations when the robot is commanded to move."""
        reward_air_time = jp.sum((air_time - 0.15) * first_contact)
        reward_air_time *= math.normalize(commands[:2])[1] > 0.05
        return reward_air_time

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_vel: jax.Array,
    ) -> jax.Array:
        """Penalize joint motion when the commanded speed is near zero."""
        return jp.sum(jp.abs(joint_vel)) * (math.normalize(commands[:2])[1] < 0.1)

    def _reward_foot_slip(
        self,
        pipeline_state: base.State,
        contact_filt: jax.Array,
    ) -> jax.Array:
        """Penalize lateral foot motion while a foot is in contact."""
        if len(self._lower_leg_body_id) == 0:
            return 0.0

        lower_leg_vel = pipeline_state.xd.vel[self._lower_leg_body_id]
        vel_xy = lower_leg_vel[:, :2]
        vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
        return jp.sum(vel_xy_norm_sq * contact_filt)

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        """Penalize terminating before the command-resample horizon."""
        return done & (step < COMMAND_RESAMPLE_STEPS)

    def _reward_energy(
        self,
        qvel: jax.Array,
        qfrc_actuator: jax.Array,
    ) -> jax.Array:
        """Penalize energy use via joint speed times actuator force."""
        actuator_forces = qfrc_actuator[self._qd_joint_start :]
        return jp.sum(jp.abs(qvel) * jp.abs(actuator_forces))

    def render(
        self,
        trajectory: list[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        """Render a rollout using Brax's built-in MuJoCo renderer."""
        return super().render(
            trajectory,
            camera=camera or "track",
            width=width,
            height=height,
        )
