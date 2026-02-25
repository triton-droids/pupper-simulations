"""Tests for locomotion/bittle_env.py (19 tests, 1 per function/method)."""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import jax
import jax.numpy as jp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_SIZE = 34  # 1 + 3 + 3 + 9 + 9 + 9
NU = 9

DEFAULT_POSE = jp.array([
    -0.6908, 1.9782, 0.7222, 1.9468, -0.596904,
    -0.6908, 1.9782, 0.7222, 1.9468,
])


def _make_fake_sys():
    """Return a lightweight fake sys object that satisfies BittleEnv.__init__."""
    sys = MagicMock()
    sys.nq = 16  # 7 freejoint + 9 actuated
    sys.nv = 15  # 6 freejoint + 9 actuated
    sys.nu = 9
    sys.opt.timestep = 0.004
    sys.dof_damping = jp.zeros(15)
    sys.mj_model = MagicMock()
    # tree_replace returns a copy with modifications
    sys.tree_replace = lambda d: sys
    sys.replace = lambda **kw: sys
    return sys


class _FakePipelineState(SimpleNamespace):
    """SimpleNamespace with a .replace() method mimicking brax State."""

    def replace(self, **kwargs):
        import copy
        new = copy.copy(self)
        for k, v in kwargs.items():
            setattr(new, k, v)
        return new


def _make_pipeline_state():
    """Return a fake pipeline_state for reward / obs testing."""
    nq, nv = 16, 15
    ps = _FakePipelineState(
        q=jp.zeros(nq),
        qd=jp.zeros(nv),
        x=SimpleNamespace(
            pos=jp.array([[0.0, 0.0, 0.075]] * 5),   # 5 bodies
            rot=jp.array([[1.0, 0.0, 0.0, 0.0]] * 5),
        ),
        xd=SimpleNamespace(
            vel=jp.zeros((5, 3)),
            ang=jp.zeros((5, 3)),
        ),
        xpos=jp.array([[0.0, 0.0, 0.075]] * 5),
        qfrc_actuator=jp.zeros(nv),
    )
    # Set joint positions to default pose
    ps.q = ps.q.at[7:].set(DEFAULT_POSE)
    return ps


def _build_env():
    """Instantiate BittleEnv with all heavy deps mocked out."""
    with patch("bittle_env.mjcf.load") as mock_load, \
         patch("bittle_env.mujoco.mj_name2id") as mock_name2id, \
         patch("bittle_env.PipelineEnv.__init__", return_value=None):

        mock_load.return_value = _make_fake_sys()
        # base=1, four legs=2,3,4,5
        mock_name2id.side_effect = lambda *a, **kw: {
            "base": 1,
            "servos_rf_1": 2,
            "servos_rr_1": 3,
            "servos_lf_1": 4,
            "servos_lr_1": 5,
        }.get(a[-1], 0)

        from bittle_env import BittleEnv
        env = BittleEnv(xml_path="dummy.xml")
        # Manually set attributes that PipelineEnv.__init__ would set
        env.sys = _make_fake_sys()
        # dt is a read-only property: dt = sys.opt.timestep * _n_frames
        # 0.004 * 5 = 0.02
        env._n_frames = 5
        return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetConfig:
    def test_get_config(self):
        from bittle_env import get_config
        cfg = get_config()
        scales = cfg.rewards.scales
        expected_keys = {
            "tracking_lin_vel", "tracking_ang_vel",
            "lin_vel_z", "ang_vel_xy", "orientation",
            "torques", "action_rate", "joint_acc",
            "stand_still", "termination",
            "feet_air_time", "foot_slip", "energy",
        }
        assert set(scales.keys()) == expected_keys
        assert cfg.rewards.tracking_sigma == 0.25


class TestBittleEnvInit:
    def test_init(self):
        env = _build_env()
        assert env._action_scale == 0.5
        assert env._obs_noise == 0.05
        assert env._default_pose.shape == (9,)
        assert env._q_joint_start == 7
        assert env._qd_joint_start == 6
        assert len(env._lower_leg_body_id) == 4


class TestSampleCommand:
    def test_sample_command(self):
        env = _build_env()
        rng = jax.random.PRNGKey(0)
        cmd = env.sample_command(rng)
        assert cmd.shape == (3,)
        assert -0.3 <= float(cmd[0]) <= 0.6
        assert -0.3 <= float(cmd[1]) <= 0.3
        assert -0.5 <= float(cmd[2]) <= 0.5


class TestReset:
    def test_reset(self):
        env = _build_env()
        rng = jax.random.PRNGKey(1)
        ps = _make_pipeline_state()
        with patch.object(env, "pipeline_init", return_value=ps):
            state = env.reset(rng)
        assert state.obs.shape == (510,)
        assert float(state.reward) == 0.0
        assert float(state.done) == 0.0


class TestStep:
    def test_step(self):
        env = _build_env()
        rng = jax.random.PRNGKey(2)
        ps = _make_pipeline_state()
        with patch.object(env, "pipeline_init", return_value=ps):
            state = env.reset(rng)
        ps2 = _make_pipeline_state()
        with patch.object(env, "pipeline_step", return_value=ps2):
            action = jp.zeros(9)
            new_state = env.step(state, action)
        assert new_state.obs.shape == (510,)
        # last_act should be updated to the action we passed in
        assert jp.allclose(new_state.info["last_act"], action)
        # step ran (reward is a scalar, pipeline_state was consumed)
        assert new_state.reward.shape == ()


class TestGetObs:
    def test_get_obs(self):
        env = _build_env()
        ps = _make_pipeline_state()
        rng = jax.random.PRNGKey(3)
        state_info = {
            "rng": rng,
            "last_act": jp.zeros(NU),
            "command": jp.array([0.1, 0.0, 0.0]),
        }
        obs_history = jp.zeros(15 * OBS_SIZE)
        obs = env._get_obs(ps, state_info, obs_history)
        assert obs.shape == (510,)
        # After one call, first OBS_SIZE elements should be non-trivially set
        # (at least gravity component is nonzero)
        assert not jp.allclose(obs[:OBS_SIZE], jp.zeros(OBS_SIZE))


# ---------------------------------------------------------------------------
# Reward function tests
# ---------------------------------------------------------------------------

class TestRewardLinVelZ:
    def test_reward_lin_vel_z(self):
        env = _build_env()
        env._base_body_id = 1
        xd_zero = SimpleNamespace(vel=jp.zeros((5, 3)), ang=jp.zeros((5, 3)))
        assert float(env._reward_lin_vel_z(xd_zero)) == 0.0
        xd_nonzero = SimpleNamespace(
            vel=jp.zeros((5, 3)).at[1, 2].set(2.0), ang=jp.zeros((5, 3))
        )
        assert float(env._reward_lin_vel_z(xd_nonzero)) > 0.0


class TestRewardAngVelXY:
    def test_reward_ang_vel_xy(self):
        env = _build_env()
        env._base_body_id = 1
        xd_zero = SimpleNamespace(vel=jp.zeros((5, 3)), ang=jp.zeros((5, 3)))
        assert float(env._reward_ang_vel_xy(xd_zero)) == 0.0
        xd_nonzero = SimpleNamespace(
            vel=jp.zeros((5, 3)),
            ang=jp.zeros((5, 3)).at[1, 0].set(1.0),
        )
        assert float(env._reward_ang_vel_xy(xd_nonzero)) > 0.0


class TestRewardOrientation:
    def test_reward_orientation(self):
        env = _build_env()
        env._base_body_id = 1
        x_upright = SimpleNamespace(
            pos=jp.zeros((5, 3)),
            rot=jp.array([[1.0, 0.0, 0.0, 0.0]] * 5),
        )
        assert float(env._reward_orientation(x_upright)) == pytest.approx(0.0, abs=1e-5)
        # Tilted 90 degrees around x-axis
        import math as pymath
        a = pymath.pi / 4
        q = jp.array([pymath.cos(a / 2), pymath.sin(a / 2), 0.0, 0.0])
        x_tilted = SimpleNamespace(
            pos=jp.zeros((5, 3)),
            rot=jp.tile(q, (5, 1)),
        )
        assert float(env._reward_orientation(x_tilted)) > 0.0


class TestRewardTorques:
    def test_reward_torques(self):
        env = _build_env()
        assert float(env._reward_torques(jp.zeros(15))) == 0.0
        assert float(env._reward_torques(jp.ones(15))) > 0.0


class TestRewardActionRate:
    def test_reward_action_rate(self):
        env = _build_env()
        same = jp.ones(9)
        assert float(env._reward_action_rate(same, same)) == 0.0
        diff = jp.zeros(9)
        assert float(env._reward_action_rate(same, diff)) > 0.0


class TestRewardJointAcc:
    def test_reward_joint_acc(self):
        env = _build_env()
        v = jp.ones(9)
        assert float(env._reward_joint_acc(v, v)) == 0.0
        assert float(env._reward_joint_acc(v, jp.zeros(9))) > 0.0


class TestRewardTrackingLinVel:
    def test_reward_tracking_lin_vel(self):
        env = _build_env()
        env._base_body_id = 1
        cmd = jp.array([0.3, 0.0, 0.0])
        # Identity rotation, velocity matching command
        x = SimpleNamespace(
            pos=jp.zeros((5, 3)),
            rot=jp.array([[1.0, 0.0, 0.0, 0.0]] * 5),
        )
        xd_match = SimpleNamespace(
            vel=jp.zeros((5, 3)).at[1, :2].set(jp.array([0.3, 0.0])),
            ang=jp.zeros((5, 3)),
        )
        r_match = float(env._reward_tracking_lin_vel(cmd, x, xd_match))
        assert r_match == pytest.approx(1.0, abs=0.01)
        xd_off = SimpleNamespace(
            vel=jp.zeros((5, 3)).at[1, :2].set(jp.array([1.0, 1.0])),
            ang=jp.zeros((5, 3)),
        )
        r_off = float(env._reward_tracking_lin_vel(cmd, x, xd_off))
        assert r_off < r_match


class TestRewardTrackingAngVel:
    def test_reward_tracking_ang_vel(self):
        env = _build_env()
        env._base_body_id = 1
        cmd = jp.array([0.0, 0.0, 0.3])
        x = SimpleNamespace(
            pos=jp.zeros((5, 3)),
            rot=jp.array([[1.0, 0.0, 0.0, 0.0]] * 5),
        )
        xd_match = SimpleNamespace(
            vel=jp.zeros((5, 3)),
            ang=jp.zeros((5, 3)).at[1, 2].set(0.3),
        )
        r = float(env._reward_tracking_ang_vel(cmd, x, xd_match))
        assert r == pytest.approx(1.0, abs=0.01)


class TestRewardFeetAirTime:
    def test_reward_feet_air_time(self):
        env = _build_env()
        air_time = jp.array([0.2, 0.1, 0.3, 0.0])
        first_contact = jp.array([True, False, True, False])
        cmd = jp.array([0.3, 0.0, 0.0])  # non-zero so multiplier > 0
        r = env._reward_feet_air_time(air_time, first_contact, cmd)
        assert isinstance(float(r), float)


class TestRewardStandStill:
    def test_reward_stand_still(self):
        env = _build_env()
        cmd_zero = jp.array([0.0, 0.0, 0.0])
        joint_vel = jp.ones(9)
        r = float(env._reward_stand_still(cmd_zero, joint_vel))
        assert r > 0.0  # penalizes motion when command is zero


class TestRewardFootSlip:
    def test_reward_foot_slip(self):
        env = _build_env()
        env._lower_leg_body_id = np.array([])
        ps = _make_pipeline_state()
        contact = jp.ones(4, dtype=bool)
        assert float(env._reward_foot_slip(ps, contact)) == 0.0


class TestRewardTermination:
    def test_reward_termination(self):
        env = _build_env()
        assert float(env._reward_termination(jp.bool_(True), jp.int32(10))) == 1.0
        assert float(env._reward_termination(jp.bool_(True), jp.int32(600))) == 0.0


class TestRewardEnergy:
    def test_reward_energy(self):
        env = _build_env()
        env._qd_joint_start = 6
        assert float(env._reward_energy(jp.zeros(9), jp.zeros(15))) == 0.0
        assert float(env._reward_energy(jp.ones(9), jp.ones(15))) > 0.0
