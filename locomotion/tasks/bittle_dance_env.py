"""
Dance-specific simulator rules for the Bittle robot.

In everyday terms, this file tells the simulator:

- what the robot is allowed to control
- what a "good dance move" looks like
- when the robot should be considered to have failed
- what information the learning system gets to see each step

The dance task keeps the same 9 joint controls as the walking task, but changes the
goal from "walk where commanded" to "follow an in-place rhythm."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
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

TERMINATION_MARGIN = 0.35

LOWER_LEG_BODY_NAMES = (
    "servos_rf_1",
    "servos_rr_1",
    "servos_lf_1",
    "servos_lr_1",
)
FOOT_HEIGHT_OFFSET = 0.06
FOOT_CONTACT_RADIUS = 0.015

TASK_HPARAMS_FILENAME = "bittle_dance_hparams.json"
TASK_HPARAMETER_KEYS = frozenset(
    {
        "obs_noise",
        "action_scale",
        "kick_vel",
        "enable_kicks",
        "cycle_steps",
        "dance_amplitude",
        "n_frames",
        "pose_tracking_scale",
        "bounce_tracking_scale",
        "upright_scale",
        "stay_centered_scale",
        "foot_stability_scale",
        "action_rate_scale",
        "joint_acc_scale",
        "torques_scale",
        "energy_scale",
        "termination_scale",
    }
)


def get_task_hparams_path() -> Path:
    """
    Return the JSON file that stores sweepable dance-task settings.

    Keeping the file beside this environment makes the relationship obvious to
    someone browsing the repo.
    """
    return Path(__file__).with_name(TASK_HPARAMS_FILENAME)


def _validate_task_hparam_entry(overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Check one dance-task override bundle for unsupported keys.

    This catches typos up front so a bad JSON entry does not waste a full trial.
    """
    cleaned: dict[str, Any] = {}

    for key, value in overrides.items():
        if key not in TASK_HPARAMETER_KEYS:
            valid = ", ".join(sorted(TASK_HPARAMETER_KEYS))
            raise ValueError(
                f"Unsupported dance task hyperparameter '{key}'. "
                f"Expected one of: {valid}"
            )
        cleaned[key] = value

    return cleaned


def load_task_hparam_sweep(path: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Load the dance task sweep JSON as a list of override dictionaries.

    Each list item is one "how should this dance world behave?" bundle the
    sweep runner can combine with the trainer-side PPO settings.
    """
    resolved_path = Path(path) if path is not None else get_task_hparams_path()
    if not resolved_path.is_absolute():
        resolved_path = (Path(__file__).parent / resolved_path).resolve()
    data = json.loads(resolved_path.read_text(encoding="utf-8"))

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(
            "Task hyperparameter JSON must be a list of objects, for example "
            '[{"action_scale": 0.4}, {"dance_amplitude": 0.9}]'
        )

    return [_validate_task_hparam_entry(entry) for entry in data]


def build_reward_config() -> config_dict.ConfigDict:
    """
    Return the scoring recipe for the dance task.

    These numbers decide how much the robot should care about things like:

    - matching the target pose
    - bobbing up and down with the beat
    - staying upright
    - staying near the starting point
    - avoiding jerky or wasteful motion
    """
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
                            # This is intentionally large because the reward is
                            # later scaled by dt, which would otherwise make an
                            # instant fall look almost harmless.
                            termination=-20.0,
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
    """
    Convert friendly body names into MuJoCo's numeric ids.

    MuJoCo uses integer ids internally, but the code is easier for humans to
    read when we start from names like ``servos_rf_1``.
    """
    body_ids: list[int] = []

    # Look up each requested body one by one and keep only the ones that exist.
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


def _actuator_position_ranges(sys: base.System) -> tuple[jax.Array, jax.Array]:
    """
    Read the real actuator position limits from the MuJoCo model.

    This avoids hardcoded safety bounds that can accidentally reject the
    robot's normal standing pose.
    """
    ctrl_range = jp.asarray(sys.mj_model.actuator_ctrlrange[: sys.nu], dtype=jp.float32)
    return ctrl_range[:, 0], ctrl_range[:, 1]


class BittleDanceEnv(PipelineEnv):
    """
    Bittle environment that rewards a rhythmic in-place dance.

    Instead of being told "walk forward" or "turn left," the policy is told
    where it is within a repeating dance beat and is rewarded for matching the
    scripted pose for that moment.
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
        # Load the robot model and make sure the simulator runs with the timing
        # this task expects.
        sys = mjcf.load(xml_path)
        sys = sys.tree_replace({"opt.timestep": SIMULATION_TIMESTEP})
        sys = sys.replace(dof_damping=sys.dof_damping.at[FREEJOINT_QD_START:].set(JOINT_DAMPING))

        # One control step in this environment covers several smaller MuJoCo
        # physics steps under the hood.
        self._dt = CONTROL_DT
        n_frames = kwargs.pop("n_frames", int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        # Start from the default reward recipe, then allow optional scale
        # overrides from the caller.
        self.reward_config = build_reward_config()
        for key, value in kwargs.items():
            if key.endswith("_scale"):
                self.reward_config.rewards.scales[key[:-6]] = value

        # Cache important model ids that we will read on almost every step.
        self._base_body_id = mujoco.mj_name2id(
            sys.mj_model,
            mujoco.mjtObj.mjOBJ_BODY.value,
            BASE_BODY_NAME,
        )
        self._lower_leg_body_id = _find_body_ids(sys.mj_model, LOWER_LEG_BODY_NAMES)

        # Save the main task knobs the environment will use throughout training.
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

        self.pos_lowers, self.pos_uppers = _actuator_position_ranges(sys)
        self._joint_range_lower = self.pos_lowers
        self._joint_range_upper = self.pos_uppers
        self._foot_radius = FOOT_CONTACT_RADIUS

        if log_init_summary:
            # Optional one-line summary for debugging environment setup.
            logger.info(
                "Initialized BittleDanceEnv: cycle_steps=%s amplitude=%.3f obs=%s act=%s",
                self._cycle_steps,
                self._dance_amplitude,
                self.observation_size,
                self.action_size,
            )

    def _sample_initial_phase(self, rng: jax.Array) -> jax.Array:
        """Pick a random starting point within the dance beat."""
        return jax.random.uniform(rng, (), minval=0.0, maxval=2.0 * jp.pi)

    def _phase_features(self, phase: jax.Array) -> jax.Array:
        """
        Turn the beat position into three smooth numbers.

        This gives the policy an easy way to know whether it is at the start,
        middle, or end of the repeating dance cycle.
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
        Build the target joint pose for the current moment in the dance.

        In plain language, this routine says:

        - swing one diagonal pair one way
        - swing the opposite diagonal pair the other way
        - add a smaller counter-motion on top
        - wag the neck side to side for extra style
        """
        sway = jp.sin(phase)
        counter = jp.cos(phase)
        accent = jp.sin(2.0 * phase)

        # One number per actuator. Positive and negative signs decide which
        # joints move together and which move opposite each other.
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
        """Return the torso height we want at this point in the beat."""
        return DEFAULT_BASE_POSITION[2] + 0.008 * self._dance_amplitude * jp.sin(2.0 * phase)

    def _make_initial_state_info(
        self,
        *,
        rng: jax.Array,
        phase: jax.Array,
    ) -> dict[str, Any]:
        """
        Build the extra bookkeeping carried inside the simulator state.

        This stores things the environment wants to remember from one step to
        the next, such as the current phase, last action, and last foot contact.
        """
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
        """Place the robot in its default standing pose at the start of an episode."""
        qpos = jp.zeros(self.sys.nq)

        # Fill in the base position/orientation first, then the controllable joints.
        qpos = qpos.at[0:3].set(DEFAULT_BASE_POSITION)
        qpos = qpos.at[3:7].set(DEFAULT_BASE_ROTATION)
        qpos = qpos.at[self._q_joint_start :].set(self._default_pose)

        qvel = jp.zeros(self.sys.nv)
        return self.pipeline_init(qpos, qvel)

    def reset(self, rng: jax.Array) -> State:
        """
        Start a fresh dance episode.

        This chooses a random starting beat, rebuilds the robot's initial pose,
        clears the history buffers, and returns the first observation.
        """
        rng, phase_rng = jax.random.split(rng)
        phase = self._sample_initial_phase(phase_rng)
        pipeline_state = self._build_initial_pipeline_state()
        state_info = self._make_initial_state_info(rng=rng, phase=phase)

        # The policy sees a history stack, so start with an all-zero history.
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
        """Occasionally add a tiny random shove so the policy learns some robustness."""
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
        """Guess which feet are touching the floor by checking how low the legs are."""
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
        """
        Decide whether the episode should end right now.

        The episode ends when the robot has clearly fallen over or bent its
        joints beyond the safety margin.
        """
        up_vec = math.rotate(jp.array([0, 0, 1]), x.rot[self._base_body_id])

        done = up_vec[2] < UPRIGHT_THRESHOLD
        done |= pipeline_state.x.pos[self._base_body_id, 2] < MIN_BASE_HEIGHT
        done |= jp.any(joint_angles < self.pos_lowers - TERMINATION_MARGIN)
        done |= jp.any(joint_angles > self.pos_uppers + TERMINATION_MARGIN)
        return done

    def _reward_pose_tracking(self, joint_angles: jax.Array, target_pose: jax.Array) -> jax.Array:
        """Score how closely the robot matches the pose we wanted for this beat."""
        pose_error = jp.sum(jp.square(joint_angles - target_pose))
        sigma = self.reward_config.rewards.pose_sigma
        return jp.exp(-pose_error / sigma)

    def _reward_bounce_tracking(self, x: Transform, phase: jax.Array) -> jax.Array:
        """Score how well the torso matches the target up-and-down bobbing motion."""
        target_height = self._target_base_height(phase)
        height_error = jp.square(x.pos[self._base_body_id, 2] - target_height)
        sigma = self.reward_config.rewards.height_sigma
        return jp.exp(-height_error / sigma)

    def _reward_upright(self, x: Transform) -> jax.Array:
        """Score how upright the robot is instead of leaning or tipping."""
        up = jp.array([0.0, 0.0, 1.0])
        rotated_up = math.rotate(up, x.rot[self._base_body_id])
        tilt_error = jp.sum(jp.square(rotated_up[:2]))
        sigma = self.reward_config.rewards.upright_sigma
        return jp.exp(-tilt_error / sigma)

    def _reward_stay_centered(self, xd: Motion) -> jax.Array:
        """Score how well the robot stays near its starting spot instead of drifting."""
        planar_speed_sq = jp.sum(jp.square(xd.vel[self._base_body_id, :2]))
        sigma = self.reward_config.rewards.center_sigma
        return jp.exp(-planar_speed_sq / sigma)

    def _reward_foot_stability(self, contact_filt: jax.Array) -> jax.Array:
        """Prefer having some feet on the ground so the dance stays stable."""
        num_contacts = jp.sum(contact_filt.astype(jp.float32))
        return jp.exp(-jp.square(num_contacts - 2.5))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        """Penalize sudden command jumps from one step to the next."""
        return jp.sum(jp.square(act - last_act))

    def _reward_joint_acc(self, joint_vel: jax.Array, last_joint_vel: jax.Array) -> jax.Array:
        """Penalize very abrupt changes in joint speed."""
        joint_acc = (joint_vel - last_joint_vel) / self.dt
        return jp.sum(jp.square(joint_acc))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        """Penalize pushing the motors too hard."""
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
        """Penalize wasting energy while moving."""
        actuator_forces = qfrc_actuator[self._qd_joint_start :]
        return jp.sum(jp.abs(qvel) * jp.abs(actuator_forces))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        """
        Penalize early falls more than late ones.

        In plain language: a face-plant on the first beat should hurt the score
        much more than losing balance near the end of the dance phrase.
        """
        done_float = done.astype(jp.float32)
        step_float = jp.asarray(step, dtype=jp.float32)
        cycle_steps = jp.asarray(self._cycle_steps, dtype=jp.float32)
        remaining_fraction = jp.clip((cycle_steps - step_float) / cycle_steps, 0.0, 1.0)
        return done_float * (1.0 + remaining_fraction)

    def step(self, state: State, action: jax.Array) -> State:
        """
        Advance the simulator by one control step.

        The policy does not command raw motor forces directly. Instead, it says
        "move each joint a little away from the default pose," and the simulator
        turns that into target joint positions.
        """
        # Split the random key so we can use one part now and keep the rest for later.
        rng, kick_rng = jax.random.split(state.info["rng"])
        phase = state.info["phase"]
        target_pose = self._target_pose(phase)

        # Convert the normalized action into real joint targets and clamp them to
        # safe ranges.
        position_offsets = action * self._action_scale
        target_positions = jp.clip(
            self._default_pose + position_offsets,
            self._joint_range_lower,
            self._joint_range_upper,
        )

        # Step the simulator once with those targets, then optionally add a tiny
        # random push to make the policy less fragile.
        pipeline_state = self.pipeline_step(state.pipeline_state, target_positions)
        pipeline_state = pipeline_state.replace(
            qd=pipeline_state.qd.at[:3].set(pipeline_state.qd[:3] + self._sample_kick_velocity(kick_rng))
        )

        # Read back the new physical state that resulted from the action.
        x, xd = pipeline_state.x, pipeline_state.xd
        joint_angles = pipeline_state.q[self._q_joint_start :]
        joint_vel = pipeline_state.qd[self._qd_joint_start :]

        # Estimate which feet are on the floor and whether the robot has failed.
        contact = self._estimate_foot_contact(pipeline_state)
        contact_filt = contact | state.info["last_contact"]

        done = self._compute_termination(x, pipeline_state, joint_angles)

        # Compute each reward piece separately so they can be logged and tuned.
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

        # Combine the reward pieces into one number, keeping it finite and bounded.
        reward = jp.clip(sum(scaled_rewards.values()) * self.dt, -10.0, 10.0)
        reward = jp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        # Advance the beat and update the "memory" fields the next step will use.
        next_phase = jp.mod(phase + self._phase_increment, 2.0 * jp.pi)
        state.info["phase"] = next_phase
        state.info["last_act"] = action
        state.info["last_joint_vel"] = joint_vel
        state.info["last_contact"] = contact
        state.info["rewards"] = scaled_rewards
        state.info["step"] = jp.where(done, 0, state.info["step"] + 1)
        state.info["rng"] = rng

        # Save a few user-facing metrics for logs and plots.
        state.metrics["total_dist"] = jp.linalg.norm(xd.vel[self._base_body_id, :2]) * self.dt
        state.metrics.update(scaled_rewards)

        # Build the next observation and package everything into the next state.
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
        """
        Build the observation the policy will see next.

        The observation answers basic questions like:

        - how is the body tilted?
        - where are we in the dance beat?
        - where are the joints now?
        - what did we command last step?
        """
        inv_base_rot = math.quat_inv(pipeline_state.x.rot[self._base_body_id])
        local_rpyrate = math.rotate(
            pipeline_state.xd.ang[self._base_body_id],
            inv_base_rot,
        )

        # Gather the current pose and joint-speed information.
        joint_angles = pipeline_state.q[self._q_joint_start :]
        joint_vels = pipeline_state.qd[self._qd_joint_start :]

        # Concatenate all observation pieces into one long vector.
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

        # Clamp extreme values and add a little noise so the policy does not
        # overfit to a perfectly clean simulator.
        obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
            state_info["rng"],
            obs.shape,
            minval=-1,
            maxval=1,
        )

        # Keep a short rolling history so the policy can infer motion, not just
        # the current snapshot.
        return jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

    def render(
        self,
        trajectory: list[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        """Render a list of saved simulator states into images."""
        return super().render(
            trajectory,
            camera=camera or "track",
            width=width,
            height=height,
        )
