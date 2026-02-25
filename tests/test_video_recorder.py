"""Tests for locomotion/video_recorder.py.

All physics, rendering, and JAX dependencies are mocked so tests run fast.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# Add locomotion to path so imports work the same as when running from locomotion/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "locomotion"))

from video_recorder import generate_rollout, render_frames, save_video_mp4, record_video


# ── generate_rollout ─────────────────────────────────────────────────

class TestGenerateRollout:
    def test_returns_correct_states(self):
        """Mock env.reset/step and jax.jit to verify rollout length and state contents."""
        num_steps = 5
        nq, nqd = 7, 6

        def make_pipeline_state(i):
            return SimpleNamespace(q=np.ones(nq) * i, qd=np.ones(nqd) * i)

        # Build fake brax states: reset returns state 0, steps return 1..num_steps
        reset_state = SimpleNamespace(
            obs=np.zeros(10),
            pipeline_state=make_pipeline_state(0),
        )
        step_states = []
        for i in range(1, num_steps + 1):
            step_states.append(SimpleNamespace(
                obs=np.zeros(10),
                pipeline_state=make_pipeline_state(i),
            ))

        env = MagicMock()
        env.reset.return_value = reset_state
        env.step.side_effect = step_states

        inference_fn = MagicMock(return_value=(np.zeros(9), {}))

        with patch("video_recorder.jax") as mock_jax:
            # jax.jit returns the function unchanged
            mock_jax.jit.side_effect = lambda fn: fn
            mock_jax.random.PRNGKey.return_value = np.array([0, 0])
            mock_jax.random.split.return_value = (np.array([0, 0]), np.array([1, 1]))

            rollout = generate_rollout(env, inference_fn, num_steps=num_steps, seed=0)

        assert len(rollout) == num_steps + 1
        for i, state in enumerate(rollout):
            assert hasattr(state, "q") and hasattr(state, "qd")
            assert state.q.shape[0] == nq
            assert state.qd.shape[0] == nqd


# ── render_frames ────────────────────────────────────────────────────

class TestRenderFrames:
    def test_correct_shape_and_dtype(self):
        """Mock MuJoCo renderer to verify frame count, shape, and dtype."""
        width, height = 64, 48
        num_states = 4

        rollout = [
            SimpleNamespace(q=np.zeros(7), qd=np.zeros(6))
            for _ in range(num_states)
        ]

        fake_pixel = np.zeros((height, width, 3), dtype=np.uint8)
        mock_renderer = MagicMock()
        mock_renderer.render.return_value = fake_pixel

        mock_mj_model = MagicMock()
        env = MagicMock()
        env.sys.mj_model = mock_mj_model

        with patch("video_recorder.mujoco") as mock_mujoco:
            mock_mujoco.Renderer.return_value = mock_renderer
            mock_mujoco.MjData.return_value = MagicMock()

            frames = render_frames(env, rollout, width=width, height=height)

        assert len(frames) == num_states
        for f in frames:
            assert f.shape == (height, width, 3)
            assert f.dtype == np.uint8


# ── save_video_mp4 ──────────────────────────────────────────────────

class TestSaveVideoMp4:
    def test_creates_file_and_handles_errors(self, tmp_path):
        """Verify file creation, parent dir creation, and empty-frame error."""
        # Creates file
        frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(10)]
        out = tmp_path / "test.mp4"
        save_video_mp4(frames, str(out), fps=10)
        assert out.exists()
        assert out.stat().st_size > 0

        # Creates parent dirs
        out2 = tmp_path / "a" / "b" / "test.mp4"
        save_video_mp4(frames[:5], str(out2), fps=10)
        assert out2.exists()

        # Raises on empty frames
        with pytest.raises(ValueError, match="No frames"):
            save_video_mp4([], str(tmp_path / "empty.mp4"))


# ── record_video ─────────────────────────────────────────────────────

class TestRecordVideo:
    def test_calls_pipeline_correctly(self):
        """Patch sub-functions and verify record_video orchestrates them."""
        fake_rollout = [SimpleNamespace(q=np.zeros(7), qd=np.zeros(6))]
        fake_frames = [np.zeros((48, 64, 3), dtype=np.uint8)]

        with patch("video_recorder.generate_rollout", return_value=fake_rollout) as mock_gen, \
             patch("video_recorder.render_frames", return_value=fake_frames) as mock_render, \
             patch("video_recorder.save_video_mp4") as mock_save:

            mock_env = MagicMock()
            mock_make_policy = MagicMock()
            mock_inference_fn = MagicMock()
            mock_make_policy.return_value = mock_inference_fn
            mock_params = MagicMock()

            record_video(
                mock_env,
                mock_make_policy,
                params=mock_params,
                output_path="/tmp/out.mp4",
                num_steps=10,
                seed=42,
                width=64,
                height=48,
                fps=10,
            )

            mock_make_policy.assert_called_once_with(mock_params, deterministic=True)
            mock_gen.assert_called_once_with(
                mock_env, mock_inference_fn, num_steps=10, seed=42
            )
            mock_render.assert_called_once_with(
                mock_env, fake_rollout, width=64, height=48
            )
            mock_save.assert_called_once_with(fake_frames, "/tmp/out.mp4", fps=10)
