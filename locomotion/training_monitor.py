"""
Progress monitoring and artifact generation for training runs.

The monitor acts as the callback passed into Brax PPO. Each time Brax reports
evaluation metrics, the monitor:

1. records scalar metrics in memory
2. writes the metrics to JSON
3. saves a progress plot
4. optionally renders a short rollout video using the latest policy snapshot

Keeping that logic in one callback object makes ``train.py`` much easier to
read and keeps visualization concerns out of the training loop itself.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import jax
import mujoco
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


ROLLOUT_VIDEO_STEPS = 250
VIDEO_FPS = 50
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
PLOT_FIGSIZE = (14, 5)


def _coerce_scalar_metric(value: Any) -> float | None:
    """Convert a metric value to a float if it represents a scalar."""
    if isinstance(value, (int, float, np.generic)):
        return float(value)

    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(()))

    return None


def _extract_reward_components(metrics: dict[str, Any]) -> dict[str, float]:
    """Extract only the reward component metrics used in the bar chart."""
    reward_components: dict[str, float] = {}

    for key, value in metrics.items():
        is_reward_component = key.startswith("eval/episode_reward/") or (
            key.startswith("eval/")
            and "reward" in key.lower()
            and key != "eval/episode_reward"
        )
        if not is_reward_component:
            continue

        scalar_value = _coerce_scalar_metric(value)
        if scalar_value is not None:
            reward_components[key] = scalar_value

    return reward_components


class TrainingMonitor:
    """Collect training metrics and generate plots, JSON, and videos."""

    def __init__(
        self,
        output_dir: Path,
        num_timesteps: int,
        logger: logging.Logger,
        env: Any | None = None,
        make_inference_fn: Any | None = None,
    ):
        self.output_dir = output_dir
        self.num_timesteps = num_timesteps
        self.logger = logger
        self.env = env
        self.make_inference_fn_cached = make_inference_fn

        self.x_data: list[int] = []
        self.y_data: list[float] = []
        self.y_std_data: list[float] = []
        self.times: list[datetime] = [datetime.now()]
        self.all_metrics: list[dict[str, float]] = []

        self.plots_dir = output_dir / "plots"
        self.metrics_dir = output_dir / "metrics"
        self.videos_dir = output_dir / "videos"

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, num_steps: int, metrics: dict[str, Any]) -> None:
        """Record one evaluation event and materialize the current artifacts."""
        self.times.append(datetime.now())
        time_delta = (self.times[-1] - self.times[-2]).total_seconds()

        episode_reward = float(metrics["eval/episode_reward"])
        episode_reward_std = float(metrics.get("eval/episode_reward_std", 0.0))

        self.x_data.append(num_steps)
        self.y_data.append(episode_reward)
        self.y_std_data.append(episode_reward_std)
        self.all_metrics.append(self._collect_numeric_metrics(metrics))

        self._log_evaluation_summary(num_steps, episode_reward, episode_reward_std, time_delta)
        self._plot_progress(num_steps, metrics)
        self._save_metrics(num_steps)

        if self.env is not None and self.make_inference_fn_cached is not None:
            self._generate_video(num_steps)

    def _collect_numeric_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Filter the Brax metrics dictionary down to scalar numeric values."""
        numeric_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            scalar_value = _coerce_scalar_metric(value)
            if scalar_value is not None:
                numeric_metrics[key] = scalar_value
        return numeric_metrics

    def _log_evaluation_summary(
        self,
        num_steps: int,
        episode_reward: float,
        episode_reward_std: float,
        time_delta: float,
    ) -> None:
        """Write the human-readable evaluation summary to the run log."""
        self.logger.info("=" * 80)
        self.logger.info("EVALUATION AT STEP %s", f"{num_steps:,}")
        self.logger.info("=" * 80)
        self.logger.info(
            "Episode Reward:     %.4f +/- %.4f",
            episode_reward,
            episode_reward_std,
        )
        self.logger.info("Time since last:    %.2fs", time_delta)

        self.logger.debug("All available metrics:")
        for key, value in sorted(self.all_metrics[-1].items()):
            self.logger.debug("  %-30s: %.6f", key, value)

        if len(self.y_data) > 1:
            improvement = self.y_data[-1] - self.y_data[-2]
            self.logger.info("Reward change:      %+0.4f", improvement)
            self.logger.info("Best so far:        %.4f", max(self.y_data))
            self.logger.info("Worst so far:       %.4f", min(self.y_data))

        self.logger.info("=" * 80)

    def _plot_progress(self, num_steps: int, metrics: dict[str, Any]) -> None:
        """Generate both the point-in-time plot and the rolling latest plot."""
        reward_components = _extract_reward_components(metrics)

        step_figure = self._build_progress_figure(
            num_steps=num_steps,
            reward_components=reward_components,
            title_prefix="Training Progress",
            component_title="Reward Components",
        )
        step_plot_path = self.plots_dir / f"progress_step_{num_steps:08d}.png"
        step_figure.savefig(step_plot_path, dpi=150, bbox_inches="tight")
        plt.close(step_figure)

        latest_figure = self._build_progress_figure(
            num_steps=num_steps,
            reward_components=reward_components,
            title_prefix="Final Progress",
            component_title="Final Reward Components",
        )
        latest_plot_path = self.plots_dir / "latest_progress.png"
        latest_figure.savefig(latest_plot_path, dpi=150, bbox_inches="tight")
        plt.close(latest_figure)

        self.logger.debug("Saved plot to %s", step_plot_path)

    def _build_progress_figure(
        self,
        *,
        num_steps: int,
        reward_components: dict[str, float],
        title_prefix: str,
        component_title: str,
    ) -> Figure:
        """Build the two-panel matplotlib figure used for progress snapshots."""
        figure, (reward_ax, component_ax) = plt.subplots(1, 2, figsize=PLOT_FIGSIZE)

        reward_ax.errorbar(
            self.x_data,
            self.y_data,
            yerr=self.y_std_data,
            marker="o",
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=8,
            label="Episode Reward",
        )
        reward_ax.axhline(y=0, color="r", linestyle="--", alpha=0.3, label="Zero reward")
        reward_ax.set_xlim([0, self.num_timesteps * 1.1])

        if self.y_data:
            y_min = min(self.y_data) - max(self.y_std_data) * 1.2
            y_max = max(self.y_data) + max(self.y_std_data) * 1.2
            reward_ax.set_ylim([y_min, y_max])

        reward_ax.set_xlabel("Environment Steps", fontsize=12)
        reward_ax.set_ylabel("Reward per Episode", fontsize=12)
        reward_ax.set_title(f"{title_prefix} (Step {num_steps:,})", fontsize=14)
        reward_ax.grid(True, alpha=0.3)
        reward_ax.legend()

        self._populate_reward_component_axis(component_ax, reward_components, component_title)
        figure.tight_layout()
        return figure

    def _populate_reward_component_axis(
        self,
        axis: Any,
        reward_components: dict[str, float],
        component_title: str,
    ) -> None:
        """Fill the right-hand plot with a reward breakdown or a placeholder."""
        if not reward_components:
            axis.text(
                0.5,
                0.5,
                "No component\nbreakdown available",
                ha="center",
                va="center",
                fontsize=12,
            )
            axis.set_xlim([0, 1])
            axis.set_ylim([0, 1])
            return

        names = [
            key.replace("eval/episode_reward/", "").replace("eval/", "")
            for key in reward_components.keys()
        ]
        values = list(reward_components.values())
        colors = ["green" if value > 0 else "red" for value in values]

        axis.barh(range(len(names)), values, color=colors, alpha=0.6)
        axis.set_yticks(range(len(names)))
        axis.set_yticklabels(names, fontsize=9)
        axis.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        axis.set_xlabel("Reward Contribution", fontsize=12)
        axis.set_title(component_title, fontsize=12)
        axis.grid(True, alpha=0.3, axis="x")

    def _save_metrics(self, num_steps: int) -> None:
        """Persist the full metrics history as JSON."""
        metrics_data = {
            "steps": self.x_data,
            "rewards": self.y_data,
            "reward_stds": self.y_std_data,
            "timestamps": [timestamp.isoformat() for timestamp in self.times[1:]],
            "all_metrics": self.all_metrics,
        }

        step_metrics_path = self.metrics_dir / f"metrics_step_{num_steps:08d}.json"
        latest_metrics_path = self.metrics_dir / "latest_metrics.json"

        with open(step_metrics_path, "w", encoding="utf-8") as file_handle:
            json.dump(metrics_data, file_handle, indent=2)

        with open(latest_metrics_path, "w", encoding="utf-8") as file_handle:
            json.dump(metrics_data, file_handle, indent=2)

        self.logger.debug("Saved metrics to %s", step_metrics_path)

    def _generate_video(self, num_steps: int) -> None:
        """Render a short rollout video using the latest cached policy."""
        try:
            self.logger.info("Generating video...")
            rollout = self._collect_rollout()
            frames = self._render_rollout_frames(rollout)

            video_path = self.videos_dir / f"video_step_{num_steps:08d}.mp4"
            self._write_video_opencv(frames, video_path, fps=VIDEO_FPS)
            self.logger.info("Saved video to %s", video_path)

            latest_path = self.videos_dir / "latest_video.mp4"
            self._write_video_opencv(frames, latest_path, fps=VIDEO_FPS)
        except Exception as exc:
            self.logger.warning("Failed to generate video: %s", exc)

    def _collect_rollout(self) -> list[Any]:
        """Roll out the latest policy in the environment and collect states."""
        inference_fn = self.make_inference_fn_cached
        jit_reset = jax.jit(self.env.reset)
        jit_step = jax.jit(self.env.step)
        jit_inference = jax.jit(inference_fn)

        rng = jax.random.PRNGKey(0)
        state = jit_reset(rng)
        rollout = [state.pipeline_state]

        for _ in range(ROLLOUT_VIDEO_STEPS):
            rng, key_sample = jax.random.split(rng)
            action, _ = jit_inference(state.obs, key_sample)
            state = jit_step(state, action)
            rollout.append(state.pipeline_state)

        return rollout

    def _render_rollout_frames(self, rollout: list[Any]) -> list[np.ndarray]:
        """Render MuJoCo frames from a collected rollout."""
        self.logger.info("Rendering frames...")

        mj_model = self.env.sys.mj_model
        renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)

        frames: list[np.ndarray] = []
        for pipeline_state in rollout:
            mj_data = mujoco.MjData(mj_model)
            mj_data.qpos[:] = np.array(pipeline_state.q)
            mj_data.qvel[:] = np.array(pipeline_state.qd)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data)
            frames.append(renderer.render())

        renderer.close()
        return frames

    def _write_video_opencv(
        self,
        frames: list[np.ndarray],
        output_path: Path,
        fps: int = VIDEO_FPS,
    ) -> None:
        """Write frames to a video file using OpenCV."""
        if not frames:
            raise ValueError("No frames to write")

        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()

    def get_summary(self) -> dict[str, Any]:
        """Return summary statistics computed from recorded evaluations."""
        if not self.y_data:
            return {}

        return {
            "final_reward": self.y_data[-1],
            "final_std": self.y_std_data[-1],
            "best_reward": max(self.y_data),
            "worst_reward": min(self.y_data),
            "mean_reward": np.mean(self.y_data),
            "total_time": (self.times[-1] - self.times[0]).total_seconds(),
            "num_evaluations": len(self.y_data),
        }
