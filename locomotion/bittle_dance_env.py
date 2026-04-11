"""
Dance-focused Brax environment for the Bittle quadruped.

High-level idea
---------------
This environment keeps the same action interface as ``BittleEnv``:

- the policy outputs 9 normalized joint offsets in ``[-1, 1]``
- those offsets are scaled and added to a default pose
- the MuJoCo position actuators track the resulting joint targets

What changes is the task. Instead of tracking a walking velocity command, the
policy is rewarded for following a procedurally generated dance cycle. The
cycle is encoded by a phase variable, and the observation exposes that phase
using sine/cosine features so the policy can synchronize its motion to the
beat.

The reward mixes:

- phase-conditioned pose tracking
- synchronized base bounce
- upright posture
- staying roughly in place
- smooth, energy-efficient control

This is intentionally a separate file so the original locomotion environment
remains intact.
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

FREEJOINT_Q_START = 7
FREEJOINT_QD_START = 6
BASE_BODY_NAME = "base"

OBSERVATION_HISTORY_LENGTH = 15
MIN_BASE_HEIGHT = 0.02
UPRIGHT_THRESHOLD = 0.5
KICK_PROBABILITY = 0.0005

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
TERMINATION_MARGIN = 0.35
CONTROL_RANGE = 3.14159

LOWER_LEG_BODY_NAMES = (
    "servos_rf_1",
    "servos_rr_1",
    "servos_lf_1",
    "servos_lr_1",
)
FOOT_HEIGHT_OFFSET = 0.06
FOOT_CONTACT_RADIUS = 0.015


def build_reward_config() -> config_dict.ConfigDict:
    """Return the reward weights and shape parameters for the dance task."""
    return config_dict.ConfigDict(
        dict(
            rewards=config_dict.ConfigDict(
                dict(
                    scales=config_dict.ConfigDict(
                        dict(
                            pose_tracking=4.0,
                            bounce_tracking=1.5,
                            upright=1.25,
                            stay_centered=1.0,
                            foot_stability=0.6,
                            action_rate=-0.002,
                            joint_acc=-0.003,
                            torques=-0.0002,
                            energy=-0.002,
                            termination=-1.0,
                        )
                    ),
                    pose_sigma=0.20,
                    height_sigma=0.0015,
                    center_sigma=0.20,
                    upright_sigma=0.15,
                )
            )
        )
    )


def _find_body_ids(mj_model: mujoco.MjModel, body_names: Sequence[str]) -> np.ndarray:
    """Resolve body names into MuJoCo body ids, warning on missing bodies."""
    body_ids: list[int] = []

    for body_name in body_names:
        try:
            body_id = mujoco.mj_name2id(
                mj_model,
                mujoco.mjtObj.mjOBJ_BODY.value,
                body_name,
            )
        except ValueError:
            logger.warning("Body '%s' was not found in the MJCF model", body_name)
            continue

        body_ids.append(body_id)

    return np.asarray(body_ids, dtype=np.int32)


class BittleDanceEnv(PipelineEnv):
    """
    Bittle environment that rewards a rhythmic in-place dance cycle.

    The policy does not receive a velocity command. Instead, it receives a
    phase encoding and learns to track a scripted periodic choreography.
    """

    def __init__(
        self,
        xml_path: str,
        obs_noise: float = 0.05,
        action_scale: float = 0.5,
        kick_vel: float = 0.03,
        enable_kicks: bool = True,
        cycle_steps: int = 100,
        dance_amplitude: float = 1.0,
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
        self._lower_leg_body_id = _find_body_ids(sys.mj_model, LOWER_LEG_BODY_NAMES)

        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._enable_kicks = enable_kicks
        self._cycle_steps = cycle_steps
        self._phase_increment = 2.0 * jp.pi / cycle_steps
        self._dance_amplitude = dance_amplitude

        self._nu = sys.nu
        self._q_joint_start = FREEJOINT_Q_START
        self._qd_joint_start = FREEJOINT_QD_START
        self._default_pose = DEFAULT_POSE
        self._observation_size = 1 + 3 + 3 + self._nu + self._nu + self._nu

        self.pos_lowers = jp.full(sys.nu, -TERMINATION_POSITION_LIMIT)
        self.pos_uppers = jp.full(sys.nu, TERMINATION_POSITION_LIMIT)
        self._joint_range_lower = jp.full(sys.nu, -CONTROL_RANGE)
        self._joint_range_upper = jp.full(sys.nu, CONTROL_RANGE)
        self._foot_radius = FOOT_CONTACT_RADIUS

        if log_init_summary:
            logger.info(
                "Initialized BittleDanceEnv: cycle_steps=%s amplitude=%.3f obs=%s act=%s",
                self._cycle_steps,
                self._dance_amplitude,
                self.observation_size,
                self.action_size,
            )

    def _sample_initial_phase(self, rng: jax.Array) -> jax.Array:
        """Sample a random starting phase so the dance is not tied to one reset."""
        return jax.random.uniform(rng, (), minval=0.0, maxval=2.0 * jp.pi)

    def _phase_features(self, phase: jax.Array) -> jax.Array:
        """
        Encode the dance beat in three values.

        The shape matches the old locomotion command slot, which keeps the
        observation length unchanged relative to the walking environment.
        """
        return jp.array(
            [
                jp.sin(phase),
                jp.cos(phase),
                jp.sin(2.0 * phase),
            ]
        )

    def _target_pose(self, phase: jax.Array) -> jax.Array:
        """
        Produce a procedurally scripted dance pose for the given phase.

        The pattern alternates diagonal leg pairs, adds a small foreleg/hindleg
        counter-motion, and swings the neck side to side to make the motion
        visibly stylized rather than just a stationary gait.
        """
        sway = jp.sin(phase)
        counter = jp.cos(phase)
        accent = jp.sin(2.0 * phase)

        offsets = self._dance_amplitude * jp.array(
            [
                0.28 * sway,
                0.16 * counter,
                -0.28 * sway,
                -0.12 * counter,
                0.22 * accent,
                -0.28 * sway,
                -0.16 * counter,
                0.28 * sway,
                0.12 * counter,
            ]
        )
        return self._default_pose + offsets

    def _target_base_height(self, phase: jax.Array) -> jax.Array:
        """Target a gentle vertical bounce that follows the beat."""
        return DEFAULT_BASE_POSITION[2] + 0.008 * self._dance_amplitude * jp.sin(2.0 * phase)

    def _make_initial_state_info(
        self,
        *,
        rng: jax.Array,
        phase: jax.Array,
    ) -> dict[str, Any]:
        """Create the mutable info dictionary stored inside Brax state."""
        return {
            "rng": rng,
            "phase": phase,
            "last_act": jp.zeros(self._nu),
            "last_joint_vel": jp.zeros(self._nu),
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4),
            "rewards": {
                reward_name: 0.0
                for reward_name in self.reward_config.rewards.scales.keys()
            },
            "step": 0,
        }

    def _build_initial_pipeline_state(self) -> base.State:
        """Construct the initial MuJoCo state for a fresh episode."""
        qpos = jp.zeros(self.sys.nq)
        qpos = qpos.at[0:3].set(DEFAULT_BASE_POSITION)
        qpos = qpos.at[3:7].set(DEFAULT_BASE_ROTATION)
        qpos = qpos.at[self._q_joint_start :].set(self._default_pose)

        qvel = jp.zeros(self.sys.nv)
        return self.pipeline_init(qpos, qvel)

    def reset(self, rng: jax.Array) -> State:
        """Reset the dance environment and return the initial state."""
        rng, phase_rng = jax.random.split(rng)
        phase = self._sample_initial_phase(phase_rng)
        pipeline_state = self._build_initial_pipeline_state()
        state_info = self._make_initial_state_info(rng=rng, phase=phase)

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
        """Sample a small random disturbance to keep the dance robust."""
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
        """Approximate foot contact from the lower leg body heights."""
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
        """Terminate if the robot falls or joints exceed loose safety bounds."""
        up_vec = math.rotate(jp.array([0, 0, 1]), x.rot[self._base_body_id])

        done = up_vec[2] < UPRIGHT_THRESHOLD
        done |= pipeline_state.x.pos[self._base_body_id, 2] < MIN_BASE_HEIGHT
        done |= jp.any(joint_angles < self.pos_lowers - TERMINATION_MARGIN)
        done |= jp.any(joint_angles > self.pos_uppers + TERMINATION_MARGIN)
        return done

    def _reward_pose_tracking(self, joint_angles: jax.Array, target_pose: jax.Array) -> jax.Array:
        """Reward matching the scripted dance pose."""
        pose_error = jp.sum(jp.square(joint_angles - target_pose))
        sigma = self.reward_config.rewards.pose_sigma
        return jp.exp(-pose_error / sigma)

    def _reward_bounce_tracking(self, x: Transform, phase: jax.Array) -> jax.Array:
        """Reward matching a small rhythmic base-height bounce."""
        target_height = self._target_base_height(phase)
        height_error = jp.square(x.pos[self._base_body_id, 2] - target_height)
        sigma = self.reward_config.rewards.height_sigma
        return jp.exp(-height_error / sigma)

    def _reward_upright(self, x: Transform) -> jax.Array:
        """Reward keeping the torso aligned with world up."""
        up = jp.array([0.0, 0.0, 1.0])
        rotated_up = math.rotate(up, x.rot[self._base_body_id])
        tilt_error = jp.sum(jp.square(rotated_up[:2]))
        sigma = self.reward_config.rewards.upright_sigma
        return jp.exp(-tilt_error / sigma)

    def _reward_stay_centered(self, xd: Motion) -> jax.Array:
        """Reward dancing mostly in place rather than drifting across the floor."""
        planar_speed_sq = jp.sum(jp.square(xd.vel[self._base_body_id, :2]))
        sigma = self.reward_config.rewards.center_sigma
        return jp.exp(-planar_speed_sq / sigma)

    def _reward_foot_stability(self, contact_filt: jax.Array) -> jax.Array:
        """Encourage keeping some feet grounded for a stable dance posture."""
        num_contacts = jp.sum(contact_filt.astype(jp.float32))
        return jp.exp(-jp.square(num_contacts - 2.5))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        """Penalize abrupt action changes."""
        return jp.sum(jp.square(act - last_act))

    def _reward_joint_acc(self, joint_vel: jax.Array, last_joint_vel: jax.Array) -> jax.Array:
        """Penalize large joint accelerations."""
        joint_acc = (joint_vel - last_joint_vel) / self.dt
        return jp.sum(jp.square(joint_acc))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        """Penalize large actuator effort."""
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
        """Penalize energy consumption."""
        actuator_forces = qfrc_actuator[self._qd_joint_start :]
        return jp.sum(jp.abs(qvel) * jp.abs(actuator_forces))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        """Penalize early termination within the dance cycle."""
        return done & (step < self._cycle_steps)

    def step(self, state: State, action: jax.Array) -> State:
        """
        Advance the environment by one control step.

        The action is interpreted the same way as the locomotion environment:
        as an offset from ``DEFAULT_POSE`` that becomes the target for the
        position actuators.
        """
        rng, kick_rng = jax.random.split(state.info["rng"])
        phase = state.info["phase"]
        target_pose = self._target_pose(phase)

        position_offsets = action * self._action_scale
        target_positions = jp.clip(
            self._default_pose + position_offsets,
            self._joint_range_lower,
            self._joint_range_upper,
        )

        pipeline_state = self.pipeline_step(state.pipeline_state, target_positions)
        pipeline_state = pipeline_state.replace(
            qd=pipeline_state.qd.at[:3].set(pipeline_state.qd[:3] + self._sample_kick_velocity(kick_rng))
        )

        x, xd = pipeline_state.x, pipeline_state.xd
        joint_angles = pipeline_state.q[self._q_joint_start :]
        joint_vel = pipeline_state.qd[self._qd_joint_start :]

        contact = self._estimate_foot_contact(pipeline_state)
        contact_filt = contact | state.info["last_contact"]

        done = self._compute_termination(x, pipeline_state, joint_angles)

        reward_terms = {
            "pose_tracking": self._reward_pose_tracking(joint_angles, target_pose),
            "bounce_tracking": self._reward_bounce_tracking(x, phase),
            "upright": self._reward_upright(x),
            "stay_centered": self._reward_stay_centered(xd),
            "foot_stability": self._reward_foot_stability(contact_filt),
            "action_rate": self._reward_action_rate(action, state.info["last_act"]),
            "joint_acc": self._reward_joint_acc(joint_vel, state.info["last_joint_vel"]),
            "torques": self._reward_torques(pipeline_state.qfrc_actuator),
            "energy": self._reward_energy(joint_vel, pipeline_state.qfrc_actuator),
            "termination": self._reward_termination(done, state.info["step"]),
        }
        scaled_rewards = {
            key: value * self.reward_config.rewards.scales[key]
            for key, value in reward_terms.items()
        }

        reward = jp.clip(sum(scaled_rewards.values()) * self.dt, -10.0, 10.0)
        reward = jp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        next_phase = jp.mod(phase + self._phase_increment, 2.0 * jp.pi)
        state.info["phase"] = next_phase
        state.info["last_act"] = action
        state.info["last_joint_vel"] = joint_vel
        state.info["last_contact"] = contact
        state.info["rewards"] = scaled_rewards
        state.info["step"] = jp.where(done, 0, state.info["step"] + 1)
        state.info["rng"] = rng

        state.metrics["total_dist"] = jp.linalg.norm(xd.vel[self._base_body_id, :2]) * self.dt
        state.metrics.update(scaled_rewards)

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
        """Assemble the observation vector and append it to the history stack."""
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
                self._phase_features(state_info["phase"]),
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
