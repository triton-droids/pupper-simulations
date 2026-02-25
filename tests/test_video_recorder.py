"""Tests for locomotion/video_recorder.py."""

import os
import platform
import sys
import numpy as np
import pytest

# Add locomotion to path so imports work the same as when running from locomotion/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "locomotion"))

# Use EGL on Linux (remote training node), GLFW on macOS
if platform.system() == "Darwin":
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
from brax import envs
from bittle_env import BittleEnv
from video_recorder import generate_rollout, render_frames, save_video_mp4, record_video


XML_PATH = os.path.join(os.path.dirname(__file__), "..", "locomotion", "bittle_adapted_scene.xml")


@pytest.fixture(scope="module")
def env():
    """Create a Brax BittleEnv for tests (shared across module)."""
    envs.register_environment("bittle", BittleEnv)
    return envs.get_environment("bittle", xml_path=XML_PATH)


@pytest.fixture(scope="module")
def dummy_inference_fn():
    """Return a simple zero-action inference function."""
    def fn(obs, rng_key):
        return jp.zeros(9), {}
    return fn


# ── generate_rollout tests ─────────────────────────────────────────────

class TestGenerateRollout:
    def test_correct_state_count(self, env, dummy_inference_fn):
        num_steps = 5
        rollout = generate_rollout(env, dummy_inference_fn, num_steps=num_steps, seed=0)
        # num_steps + 1 because we include the reset state
        assert len(rollout) == num_steps + 1

    def test_states_have_q_and_qd(self, env, dummy_inference_fn):
        rollout = generate_rollout(env, dummy_inference_fn, num_steps=3, seed=0)
        for state in rollout:
            q = np.array(state.q)
            qd = np.array(state.qd)
            assert q.ndim == 1 and q.shape[0] > 0
            assert qd.ndim == 1 and qd.shape[0] > 0

    def test_deterministic_with_same_seed(self, env, dummy_inference_fn):
        rollout_a = generate_rollout(env, dummy_inference_fn, num_steps=5, seed=42)
        rollout_b = generate_rollout(env, dummy_inference_fn, num_steps=5, seed=42)
        for sa, sb in zip(rollout_a, rollout_b):
            np.testing.assert_array_equal(np.array(sa.q), np.array(sb.q))


# ── save_video_mp4 tests ──────────────────────────────────────────────

class TestSaveVideoMp4:
    def test_creates_file(self, tmp_path):
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(10)]
        out = tmp_path / "test.mp4"
        save_video_mp4(frames, str(out), fps=10)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "a" / "b" / "test.mp4"
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(5)]
        save_video_mp4(frames, str(out), fps=10)
        assert out.exists()

    def test_raises_on_empty_frames(self, tmp_path):
        with pytest.raises(ValueError, match="No frames"):
            save_video_mp4([], str(tmp_path / "empty.mp4"))


# ── render_frames tests ───────────────────────────────────────────────

class TestRenderFrames:
    def test_correct_shape_and_dtype(self, env, dummy_inference_fn):
        rollout = generate_rollout(env, dummy_inference_fn, num_steps=3, seed=0)
        width, height = 64, 48
        frames = render_frames(env, rollout, width=width, height=height)
        assert len(frames) == len(rollout)
        for f in frames:
            assert f.shape == (height, width, 3)
            assert f.dtype == np.uint8


# ── record_video end-to-end test ──────────────────────────────────────

class TestRecordVideo:
    def test_produces_valid_mp4(self, env, tmp_path):
        def make_policy(params, deterministic=False):
            def policy(obs, rng_key):
                return jp.zeros(9), {}
            return policy

        out = tmp_path / "video.mp4"
        record_video(
            env,
            make_policy,
            params=None,
            output_path=str(out),
            num_steps=5,
            width=64,
            height=48,
            fps=10,
        )
        assert out.exists()
        assert out.stat().st_size > 0
