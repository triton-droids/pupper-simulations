"""
Adapted from Mujoco's Quadruped Barkour Environment
Bittle quadruped environment adapted to work with Bittle's actual structure.

This environment modifies the BarkourEnv logic to work with:
- Fixed base (no free joint) - Bittle is tethered/supported during training
- 2 DOF per leg (8 leg actuators) + 1 neck actuator
- Simplified contact detection based on body positions instead of sites
- No additional bodies or joints added

The robot structure remains EXACTLY as defined in your original XML.
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
                    # Tracking rewards (reduced importance since base is fixed)
                    tracking_lin_vel=1.0,
                    tracking_ang_vel=0.5,
                    # Base state regularizations (less relevant for fixed base)
                    lin_vel_z=-1.0,
                    ang_vel_xy=-0.05,
                    orientation=-3.0,
                    # Joint regularizations
                    torques=-0.0002,
                    action_rate=-0.01,
                    # Behavior regularizations
                    stand_still=-0.5,
                    termination=-1.0,
                    # Feet rewards (simplified)
                    feet_air_time=0.05,
                    foot_slip=-0.02,
                    # Energy efficiency
                    energy=-0.001,
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


class BittleEnv(PipelineEnv):
  """Environment for Bittle quadruped using its actual structure."""

  def __init__(
      self,
      xml_path: str,
      obs_noise: float = 0.05,
      action_scale: float = 0.3,
      kick_vel: float = 0.0,  # Disabled for fixed base
      **kwargs,
  ):
    sys = mjcf.load(xml_path)
    self._dt = 0.02  # 50 fps
    sys = sys.tree_replace({'opt.timestep': 0.004})

    # Adjust gains for Bittle's servos
    sys = sys.replace(
        dof_damping=sys.dof_damping.at[:].set(0.5),  # Apply to all joints
        actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(15.0),
        actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-15.0),
    )

    n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
    super().__init__(sys, backend='mjx', n_frames=n_frames)

    self.reward_config = get_config()
    for k, v in kwargs.items():
      if k.endswith('_scale'):
        self.reward_config.rewards.scales[k[:-6]] = v

    # Bittle doesn't have a free-floating base, so we use worldbody (index 0)
    # The base is fixed to the ground in your original XML
    self._base_body_id = 0  # Worldbody is the base
    
    self._action_scale = action_scale
    self._obs_noise = obs_noise
    self._kick_vel = kick_vel
    
    # Since there's no keyframe, create a default pose from current state
    # This is the standing position defined in your XML structure
    self._init_q = jp.array(sys.init_q) if hasattr(sys, 'init_q') else jp.zeros(sys.nq)
    
    # Default pose for joints - adjust these based on your Bittle's neutral standing position
    # You have 9 actuators: 8 leg joints + 1 neck joint
    # These should be the angles when Bittle is standing naturally
    self._default_pose = jp.array([
        0.0, 0.5,    # Right front: shoulder=0, knee=0.5 rad (~30 degrees)
        0.0, 0.5,    # Right rear
        0.0,         # Neck (straight)
        0.0, 0.5,    # Left front
        0.0, 0.5,    # Left rear
    ])
    
    print(f"Bittle has {sys.nu} actuators")
    print(f"Bittle has {sys.nq} position DOFs")
    print(f"Bittle has {sys.nv} velocity DOFs")
    
    # Adjust default pose to match actual number of actuators
    if sys.nu != len(self._default_pose):
      print(f"Warning: Expected 9 actuators, found {sys.nu}. Adjusting default pose.")
      if sys.nu < len(self._default_pose):
        self._default_pose = self._default_pose[:sys.nu]
      else:
        extra = sys.nu - len(self._default_pose)
        self._default_pose = jp.concatenate([self._default_pose, jp.zeros(extra)])

    # Joint limits based on your XML ctrlrange (0 to 6.28)
    # Convert to more reasonable limits around neutral pose
    self.lowers = jp.array([-1.5] * sys.nu)
    self.uppers = jp.array([1.5] * sys.nu)
    
    # Find the lower leg bodies (shanks) - these will be used for foot position estimation
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
    
    # Estimate foot radius based on shank geometry (rough approximation)
    self._foot_radius = 0.005  # 5mm - very small for Bittle
    
    self._nv = sys.nv
    self._nu = sys.nu

  def sample_command(self, rng: jax.Array) -> jax.Array:
    """Sample a velocity command.
    
    Note: With a fixed base, these are more like "desired joint behavior" 
    rather than actual base velocities. The robot will try to produce motions
    that would result in these velocities if it could move.
    """
    # Very conservative ranges for tethered training
    lin_vel_x = [-0.2, 0.5]   # Forward/backward motion intent
    lin_vel_y = [-0.2, 0.2]   # Left/right motion intent
    ang_vel_yaw = [-0.3, 0.3] # Turning motion intent

    _, key1, key2, key3 = jax.random.split(rng, 4)
    lin_vel_x = jax.random.uniform(key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1])
    lin_vel_y = jax.random.uniform(key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1])
    ang_vel_yaw = jax.random.uniform(key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1])
    new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
    return new_cmd

  def reset(self, rng: jax.Array) -> State:
    rng, key = jax.random.split(rng)
    
    # Initialize with default pose
    qpos = self._init_q
    qvel = jp.zeros(self._nv)
    
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

    # Observation size: 1 (yaw rate) + 3 (gravity) + 3 (command) + nu (joints) + nu (last_action)
    obs_size = 1 + 3 + 3 + self._nu + self._nu
    obs_history = jp.zeros(15 * obs_size)
    obs = self._get_obs(pipeline_state, state_info, obs_history)
    reward, done = jp.zeros(2)
    metrics = {'total_dist': 0.0}
    for k in state_info['rewards']:
      metrics[k] = state_info['rewards'][k]
    state = State(pipeline_state, obs, reward, done, metrics, state_info)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    rng, cmd_rng = jax.random.split(state.info['rng'], 2)

    # No kick for fixed-base robot
    
    # Physics step
    motor_targets = self._default_pose + action * self._action_scale
    motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
    pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
    x, xd = pipeline_state.x, pipeline_state.xd

    obs = self._get_obs(pipeline_state, state.info, state.obs)
    joint_angles = pipeline_state.q
    joint_vel = pipeline_state.qd

    # Foot contact estimation based on lower leg body positions
    if len(self._lower_leg_body_id) > 0:
      # Get positions of lower leg bodies
      lower_leg_pos = pipeline_state.xpos[self._lower_leg_body_id]
      # Estimate foot positions (lower legs point downward, feet are ~6cm below)
      foot_z = lower_leg_pos[:, 2] - 0.06  # 6cm below shank body
      # Check if feet are near ground
      contact = foot_z < 0.01  # Within 1cm of ground
    else:
      # Fallback: assume always in contact
      contact = jp.ones(4, dtype=bool)
    
    contact_filt = contact | state.info['last_contact']
    first_contact = (state.info['feet_air_time'] > 0) * contact_filt
    state.info['feet_air_time'] += self.dt

    # Termination conditions
    # For fixed-base robot: terminate if joints go out of bounds or excessive forces
    done = jp.any(joint_angles < self.lowers)
    done |= jp.any(joint_angles > self.uppers)
    
    # Also terminate if actuator forces are excessive (robot is struggling/stuck)
    max_torque = jp.max(jp.abs(pipeline_state.qfrc_actuator))
    done |= max_torque > 10.0  # Adjust threshold as needed

    # Rewards - adapted for fixed-base
    rewards = {
        'tracking_lin_vel': self._reward_tracking_lin_vel(state.info['command'], x, xd),
        'tracking_ang_vel': self._reward_tracking_ang_vel(state.info['command'], x, xd),
        'lin_vel_z': self._reward_lin_vel_z(xd),
        'ang_vel_xy': self._reward_ang_vel_xy(xd),
        'orientation': self._reward_orientation(x),
        'torques': self._reward_torques(pipeline_state.qfrc_actuator),
        'action_rate': self._reward_action_rate(action, state.info['last_act']),
        'stand_still': self._reward_stand_still(state.info['command'], joint_angles),
        'feet_air_time': self._reward_feet_air_time(
            state.info['feet_air_time'], first_contact, state.info['command']
        ),
        'foot_slip': self._reward_foot_slip(pipeline_state, contact_filt),
        'termination': self._reward_termination(done, state.info['step']),
        'energy': self._reward_energy(joint_vel, pipeline_state.qfrc_actuator),
    }
    rewards = {k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()}
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

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

    # Metrics - for fixed base, track joint space motion instead
    state.metrics['total_dist'] = jp.sum(jp.abs(joint_vel)) * self.dt  # Proxy for motion
    state.metrics.update(state.info['rewards'])

    done = jp.float32(done)
    state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
    return state

  def _get_obs(
      self,
      pipeline_state: base.State,
      state_info: dict[str, Any],
      obs_history: jax.Array,
  ) -> jax.Array:
    """Get observation from current state.
    
    For fixed-base robot, we focus on joint states and IMU-like measurements.
    """
    # Since base is fixed, use worldbody (index 0) for orientation reference
    inv_base_rot = math.quat_inv(pipeline_state.x.rot[self._base_body_id])
    local_rpyrate = math.rotate(pipeline_state.xd.ang[self._base_body_id], inv_base_rot)

    obs = jp.concatenate([
        jp.array([local_rpyrate[2]]) * 0.25,                    # yaw rate
        math.rotate(jp.array([0, 0, -1]), inv_base_rot),        # projected gravity
        state_info['command'] * jp.array([2.0, 2.0, 0.25]),     # command
        pipeline_state.q - self._default_pose,                  # joint angles relative to default
        state_info['last_act'],                                 # last action
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
    """Reward for matching linear velocity command.
    
    For fixed-base, this encourages leg motions that would produce the commanded velocity.
    """
    # Use the velocity of the base (even though it's fixed, the "attempted" motion matters)
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
    rew_air_time = jp.sum((air_time - 0.05) * first_contact)  # Lower threshold for Bittle
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
    
    # Get velocities of lower leg bodies (proxy for foot velocity)
    lower_leg_vel = pipeline_state.xd.vel[self._lower_leg_body_id]
    vel_xy = lower_leg_vel[:, :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    
    # Penalize horizontal velocity while in contact
    return jp.sum(vel_xy_norm_sq * contact_filt)

  def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
    """Penalize early termination."""
    return done & (step < 500)

  def _reward_energy(self, qvel: jax.Array, qfrc_actuator: jax.Array) -> jax.Array:
    """Penalize energy consumption."""
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def render(
      self, trajectory: List[base.State], camera: str | None = None,
      width: int = 240, height: int = 320,
  ) -> Sequence[np.ndarray]:
    camera = camera or 'track'
    return super().render(trajectory, camera=camera, width=width, height=height)
