"""
Improved Bittle Quadruped Environment
Fixed version with proper reward calculation and termination conditions.
"""

from typing import Any, List, Sequence

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from brax import base
from brax import math
from brax.base import Motion, Transform
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


def get_config():
  """Returns reward config for bittle quadruped environment."""
  
  def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            scales=config_dict.ConfigDict(
                dict(
                    # Tracking rewards
                    tracking_lin_vel=10.0,
                    tracking_ang_vel=0.05,
                    # Base state regularizations
                    #lin_vel_z=-2.0,
                    #ang_vel_xy=-0.05,
                    #orientation=-5.0,
                    # Joint regularizations
                    #torques=-0.0002,
                    #action_rate=-0.01,
                    # Behavior regularizations
                    #stand_still=-0.5,
                    #termination=-1.0,
                    # Feet rewards
                    #feet_air_time=0.1,
                    #foot_slip=-0.04,
                    # Energy efficiency
                    #energy=-0.002,
                )
            ),
            tracking_sigma=0.25,
        )
    )
    return default_config

  default_config = config_dict.ConfigDict(
      dict(
          rewards=get_default_rewards_config(),
      )
  )
  
  return default_config

def print_milestone_info(step, reward, done, milestone=100):
    # Ensure inputs are at least 1D arrays
    step = jp.atleast_1d(step)
    reward = jp.atleast_1d(reward)
    done = jp.atleast_1d(done)
    
    def print_single_env(env_idx, s, r, d):
        jax.lax.cond(
            s % milestone == 0,
            lambda: jax.debug.print(
                "Env {env}: Step {step}, Reward {rew:.3f}, Done {dn}",
                env=env_idx, step=s, rew=r, dn=d
            ),
            lambda: None
        )
    
    # Vmap over all environments
    jax.vmap(print_single_env)(jp.arange(len(step)), step, reward, done)

class BittleEnv(PipelineEnv):
  """Environment for Bittle quadruped using its actual structure."""

  def __init__(
      self,
      xml_path: str,
      obs_noise: float = 0.05,
      action_scale: float = 0.3,
      kick_vel: float = 0.05,
      **kwargs,
  ):
    sys = mjcf.load(xml_path)
    self._dt = 0.02  # 50 fps
    sys = sys.tree_replace({'opt.timestep': 0.004})

    # Adjust gains for Bittle's servos
    sys = sys.replace(
        dof_damping=sys.dof_damping.at[7:].set(0.5),  # Apply damping to actuated joints only
        actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(15.0),
        actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-15.0),
    )

    n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
    super().__init__(sys, backend='mjx', n_frames=n_frames)

    self.reward_config = get_config()
    for k, v in kwargs.items():
      if k.endswith('_scale'):
        self.reward_config.rewards.scales[k[:-6]] = v

    # Find the base body (the one with freejoint)
    self._base_body_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base')
    
    self._action_scale = action_scale
    self._obs_noise = obs_noise
    self._kick_vel = kick_vel
    
    self._nv = sys.nv
    self._nu = sys.nu
    
    print(f"Bittle has {sys.nu} actuators")
    print(f"Bittle has {sys.nq} position DOFs")
    print(f"Bittle has {sys.nv} velocity DOFs")
    
    # The actuated joint positions start after the freejoint (which takes 7 in q)
    self._q_joint_start = 7  # Skip freejoint quaternion (7 DOFs in q)
    self._qd_joint_start = 6  # Skip freejoint velocities (6 DOFs in qd)
    
    print(f"Joint positions in q: indices [{self._q_joint_start}:{self._q_joint_start + sys.nu}]")
    print(f"Joint velocities in qd: indices [{self._qd_joint_start}:{self._qd_joint_start + sys.nu}]")
    
    # Default pose for the 9 actuated joints (from home keyframe)
    # Order matches the joint order in the MJCF model
    # Keyframe qpos: 0 0 0.05 1 0 0 0 -0.31416 1.19381 0.188496 1.13098 -0.723 -0.47124 1.28806 0.345576 1.28806
    # After freejoint (7 DOFs): -0.31416 1.19381 0.188496 1.13098 -0.723 -0.47124 1.28806 0.345576 1.28806
    self._default_pose = jp.array([
        -0.31416,   # shrfs (right front shoulder)
        1.19381,    # shrft (right front thigh/knee)
        0.188496,   # shrrs (right rear shoulder)
        1.13098,    # shrrt (right rear thigh/knee)
        -0.723,     # neck
        -0.47124,   # shlfs (left front shoulder)
        1.28806,    # shlft (left front thigh/knee)
        0.345576,   # shlrs (left rear shoulder)
        1.28806,    # shlrt (left rear thigh/knee)
    ])
    
    assert len(self._default_pose) == sys.nu, f"Default pose length {len(self._default_pose)} != nu {sys.nu}"
    
    # Joint limits
    self.lowers = jp.array([-1.5] * sys.nu)
    self.uppers = jp.array([1.5] * sys.nu)
    
    # Find the lower leg bodies (shanks) for foot contact detection
    lower_leg_names = [
        'servos_rf_1',  # Right front lower leg
        'servos_rr_1',  # Right rear lower leg
        'servos_lf_1',  # Left front lower leg
        'servos_lr_1',  # Left rear lower leg
    ]
    
    self._lower_leg_body_id = []
    for name in lower_leg_names:
      try:
        body_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, name)
        self._lower_leg_body_id.append(body_id)
        print(f"Found lower leg body: {name} (id={body_id})")
      except:
        print(f"Warning: Body '{name}' not found")
    
    self._lower_leg_body_id = np.array(self._lower_leg_body_id) if self._lower_leg_body_id else np.array([])
    
    # Foot radius for contact detection
    self._foot_radius = 0.015  # 15mm for Bittle

  def sample_command(self, rng: jax.Array) -> jax.Array:
    """Sample a velocity command."""
    lin_vel_x = [-0.3, 0.6]   # m/s
    lin_vel_y = [-0.3, 0.3]   # m/s
    ang_vel_yaw = [-0.5, 0.5] # rad/s

    _, key1, key2, key3 = jax.random.split(rng, 4)
    lin_vel_x = jax.random.uniform(key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1])
    lin_vel_y = jax.random.uniform(key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1])
    ang_vel_yaw = jax.random.uniform(key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1])
    new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
    return new_cmd

  def reset(self, rng: jax.Array) -> State:
    rng, key = jax.random.split(rng)
    
    # Initialize with default pose from home keyframe
    # Keyframe: 0 0 0.05 1 0 0 0 -0.31416 1.19381 0.188496 1.13098 -0.723 -0.47124 1.28806 0.345576 1.28806
    qpos = jp.zeros(self.sys.nq)
    qpos = qpos.at[0:3].set(jp.array([0.0, 0.0, 0.05]))  # Initial position: 5cm above ground (from keyframe)
    qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))  # Identity quaternion (w, x, y, z)
    qpos = qpos.at[self._q_joint_start:].set(self._default_pose)  # Joint angles from keyframe
    
    qvel = jp.zeros(self.sys.nv)
    
    pipeline_state = self.pipeline_init(qpos, qvel)

    state_info = {
        'rng': rng,
        'last_act': jp.zeros(self._nu),
        'last_vel': jp.zeros(self._nu),
        'command': self.sample_command(key),
        'last_contact': jp.zeros(4, dtype=bool),
        'feet_air_time': jp.zeros(4),
        'rewards': {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
        'step': 0,
    }

    obs_size = 1 + 3 + 3 + self._nu + self._nu + self._nu
    obs_history = jp.zeros(15 * obs_size)
    obs = self._get_obs(pipeline_state, state_info, obs_history)
    reward, done = jp.zeros(2)
    metrics = {'total_dist': 0.0}
    for k in state_info['rewards']:
      metrics[k] = state_info['rewards'][k]
    state = State(pipeline_state, obs, reward, done, metrics, state_info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    rng, cmd_rng, kick_rng = jax.random.split(state.info['rng'], 3)

    # Random kick
    kick_vel = jp.where(
        jax.random.uniform(kick_rng) < 0.001,  # 0.1% chance per step
        jax.random.uniform(kick_rng, (3,), minval=-self._kick_vel, maxval=self._kick_vel),
        jp.zeros(3)
    )
    
    # Physics step
    motor_targets = self._default_pose + action * self._action_scale
    motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
    pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
    
    # Apply kick to base
    pipeline_state = pipeline_state.replace(
        qd=pipeline_state.qd.at[:3].set(pipeline_state.qd[:3] + kick_vel)
    )
    
    x, xd = pipeline_state.x, pipeline_state.xd

    # Extract joint states
    joint_angles = pipeline_state.q[self._q_joint_start:]
    joint_vel = pipeline_state.qd[self._qd_joint_start:]

    # Foot contact estimation
    if len(self._lower_leg_body_id) > 0:
      lower_leg_pos = pipeline_state.xpos[self._lower_leg_body_id]
      foot_z = lower_leg_pos[:, 2] - 0.06
      contact = foot_z < self._foot_radius
    else:
      contact = jp.ones(4, dtype=bool)
    
    contact_filt = contact | state.info['last_contact']
    first_contact = (state.info['feet_air_time'] > 0) * contact_filt
    state.info['feet_air_time'] += self.dt

    # Termination conditions
    # Robot starts at 5cm height, so we need appropriate thresholds
    up_vec = math.rotate(jp.array([0, 0, 1]), x.rot[self._base_body_id])
    done = up_vec[2] < 0.3  # Allow tilt up to ~70 degrees from vertical
    done |= pipeline_state.x.pos[self._base_body_id, 2] < 0.02  # Terminate if base drops below 2cm
    done |= jp.any(joint_angles < self.lowers - 0.1)  # Add some tolerance to joint limits
    done |= jp.any(joint_angles > self.uppers + 0.1)

    # Rewards
    # Overpenalizing leads to policy collapse; slowly reintroduce rewards
    rewards = {
        'tracking_lin_vel': self._reward_tracking_lin_vel(state.info['command'], x, xd),
        'tracking_ang_vel': self._reward_tracking_ang_vel(state.info['command'], x, xd),
        #'lin_vel_z': self._reward_lin_vel_z(xd),
        #'ang_vel_xy': self._reward_ang_vel_xy(xd),
        #'orientation': self._reward_orientation(x),
        #'torques': self._reward_torques(pipeline_state.qfrc_actuator),
        #'action_rate': self._reward_action_rate(action, state.info['last_act']),
        #'stand_still': self._reward_stand_still(state.info['command'], joint_angles),
        #'feet_air_time': self._reward_feet_air_time(
        #    state.info['feet_air_time'], first_contact, state.info['command']
        #),
        #'foot_slip': self._reward_foot_slip(pipeline_state, contact_filt),
        #'termination': self._reward_termination(done, state.info['step']),
        #'energy': self._reward_energy(joint_vel, pipeline_state.qfrc_actuator),
    }
    
    # Scale rewards
    rewards = {k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()}
    
    # Sum and clip reward - ALLOW NEGATIVE
    reward = jp.clip(sum(rewards.values()) * self.dt, -10.0, 10.0)
    
    # Replace NaN with 0 to prevent training crashes
    reward = jp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    #print_milestone_info(state.info['step'], reward, done, milestone=100)

    state.info['last_act'] = action
    state.info['last_vel'] = joint_vel
    state.info['feet_air_time'] *= ~contact_filt
    state.info['last_contact'] = contact
    state.info['rewards'] = rewards
    state.info['step'] += 1
    state.info['rng'] = rng

    # Sample new command periodically
    state.info['command'] = jp.where(
        state.info['step'] > 500,
        self.sample_command(cmd_rng),
        state.info['command'],
    )
    state.info['step'] = jp.where(
        done | (state.info['step'] > 500), 0, state.info['step']
    )

    # Metrics
    state.metrics['total_dist'] = jp.linalg.norm(xd.vel[self._base_body_id, :2]) * self.dt
    state.metrics.update(state.info['rewards'])

    done = jp.float32(done)
    obs = self._get_obs(pipeline_state, state.info, state.obs)
    state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
    return state

  def _get_obs(
      self,
      pipeline_state: base.State,
      state_info: dict[str, Any],
      obs_history: jax.Array,
  ) -> jax.Array:
    """Get observation from current state."""
    inv_base_rot = math.quat_inv(pipeline_state.x.rot[self._base_body_id])
    local_rpyrate = math.rotate(pipeline_state.xd.ang[self._base_body_id], inv_base_rot)

    # Extract only the actuated joint positions and velocities
    joint_angles = pipeline_state.q[self._q_joint_start:]
    joint_vels = pipeline_state.qd[self._qd_joint_start:]

    obs = jp.concatenate([
        jp.array([local_rpyrate[2]]) * 0.25,                    # yaw rate
        math.rotate(jp.array([0, 0, -1]), inv_base_rot),        # projected gravity (3)
        state_info['command'] * jp.array([2.0, 2.0, 0.25]),     # command (3)
        joint_angles - self._default_pose,                      # joint angles relative to default (9)
        joint_vels * 0.05,                                      # joint velocities (9)
        state_info['last_act'],                                 # last action (9)
    ])

    # Add noise
    obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
        state_info['rng'], obs.shape, minval=-1, maxval=1
    )
    
    # Stack history
    obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)
    return obs

  # ------------ Reward functions ----------------
  
  def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
    """Penalize vertical velocity of base."""
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

  def _reward_tracking_lin_vel(
      self, commands: jax.Array, x: Transform, xd: Motion
  ) -> jax.Array:
    """Reward for matching linear velocity command."""
    local_vel = math.rotate(xd.vel[self._base_body_id], math.quat_inv(x.rot[self._base_body_id]))
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self.reward_config.rewards.tracking_sigma)

  def _reward_tracking_ang_vel(
      self, commands: jax.Array, x: Transform, xd: Motion
  ) -> jax.Array:
    """Reward for matching angular velocity command."""
    base_ang_vel = math.rotate(xd.ang[self._base_body_id], math.quat_inv(x.rot[self._base_body_id]))
    ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
    return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    """Reward appropriate swing durations."""
    rew_air_time = jp.sum((air_time - 0.05) * first_contact)
    rew_air_time *= math.normalize(commands[:2])[1] > 0.05
    return rew_air_time

  def _reward_stand_still(
      self, commands: jax.Array, joint_angles: jax.Array
  ) -> jax.Array:
    """Penalize motion when command is zero."""
    return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
        math.normalize(commands[:2])[1] < 0.1
    )

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

  def _reward_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
    """Penalize energy consumption."""
    # qvel is 9 elements (joint velocities)
    # qfrc_actuator is 15 elements (6 for freejoint + 9 for actuated joints)
    # Extract only the actuated joint forces (skip the first 6 freejoint forces)
    actuator_forces = qfrc_actuator[self._qd_joint_start:]
    return jp.sum(jp.abs(qvel) * jp.abs(actuator_forces))

  def render(
      self, trajectory: List[base.State], camera: str | None = None,
      width: int = 240, height: int = 320,
  ) -> Sequence[np.ndarray]:
    camera = camera or 'track'
    return super().render(trajectory, camera=camera, width=width, height=height)