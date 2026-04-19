"""
Track training progress and write the final human-friendly artifacts.

During training, Brax periodically says "here is the latest score." This class
collects those check-ins. At the end, it turns that history into the files a
person usually wants to inspect:

- one metrics JSON
- one final progress chart
- one final rollout video
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
FINAL_METRICS_FILENAME = "final_metrics.json"
FINAL_PLOT_FILENAME = "final_progress.png"
FINAL_VIDEO_FILENAME = "final_video.mp4"


def _coerce_scalar_metric(value: Any) -> float | None:
    """Return a plain number when a metric is a single value, otherwise skip it."""
    if isinstance(value, (int, float, np.generic)):
        return float(value)

    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(()))

    return None


def _extract_reward_components(metrics: dict[str, Any]) -> dict[str, float]:
    """
    Pull out the reward sub-pieces used in the bar chart.

    Training metrics contain many values. This helper keeps only the ones that
    explain why the total reward went up or down.
    """
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
    """
    Remember evaluation history and turn it into final files.

    Think of this class as the run's notebook: it watches the scores come in,
    then writes the summary materials once training is over.
    """

    def __init__(
        self,
        output_dir: Path,
        num_timesteps: int,
        logger: logging.Logger,
        env: Any | None = None,
        make_inference_fn: Any | None = None,
    ):
        # Save the run-wide context the monitor needs later when it writes files.
        self.output_dir = output_dir
        self.num_timesteps = num_timesteps
        self.logger = logger
        self.env = env
        self.make_inference_fn_cached = make_inference_fn

        # These lists grow one evaluation at a time and become the final plots
        # and metrics JSON.
        self.x_data: list[int] = []
        self.y_data: list[float] = []
        self.y_std_data: list[float] = []
        self.times: list[datetime] = [datetime.now()]
        self.all_metrics: list[dict[str, float]] = []
        self.latest_reward_components: dict[str, float] = {}
        self.latest_num_steps = 0

        # Keep outputs separated by type so a trial directory is easy to browse.
        self.plots_dir = output_dir / "plots"
        self.metrics_dir = output_dir / "metrics"
        self.videos_dir = output_dir / "videos"

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, num_steps: int, metrics: dict[str, Any]) -> None:
        """
        Record one progress update from PPO.

        Brax calls this during training every time it finishes an evaluation
        pass and has fresh numbers to report.
        """
        self.times.append(datetime.now())
        time_delta = (self.times[-1] - self.times[-2]).total_seconds()

        # Pull out the main score and its spread across evaluation episodes.
        episode_reward = float(metrics["eval/episode_reward"])
        episode_reward_std = float(metrics.get("eval/episode_reward_std", 0.0))

        # Append the new point to the in-memory history used by the final files.
        self.x_data.append(num_steps)
        self.y_data.append(episode_reward)
        self.y_std_data.append(episode_reward_std)
        self.all_metrics.append(self._collect_numeric_metrics(metrics))
        self.latest_reward_components = _extract_reward_components(metrics)
        self.latest_num_steps = num_steps

        self._log_evaluation_summary(num_steps, episode_reward, episode_reward_std, time_delta)

    def _collect_numeric_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Keep only the simple number-valued metrics that are easy to save."""
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
        """
        Print one compact progress update to the log.

        This is the part you watch in real time while training is running.
        """
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

    def _save_final_plot(self) -> None:
        """Draw and save the single final chart for the run."""
        figure = self._build_progress_figure(
            num_steps=self.latest_num_steps,
            reward_components=self.latest_reward_components,
            title_prefix="Training Progress",
            component_title="Final Reward Components",
        )
        plot_path = self.plots_dir / FINAL_PLOT_FILENAME
        figure.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        self.logger.info("Saved final plot to %s", plot_path)

    def _build_progress_figure(
        self,
        *,
        num_steps: int,
        reward_components: dict[str, float],
        title_prefix: str,
        component_title: str,
    ) -> Figure:
        """
        Build the chart image that summarizes the run.

        The left side shows reward over time. The right side shows which reward
        pieces helped or hurt most in the latest evaluation.
        """
        figure, (reward_ax, component_ax) = plt.subplots(1, 2, figsize=PLOT_FIGSIZE)

        # Plot the main reward curve and its uncertainty bars across evaluations.
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

        # Auto-size the vertical range so the chart hugs the data instead of
        # wasting space.
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
        """Fill the right-hand chart with reward parts, or a placeholder message."""
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

        # Convert internal metric keys into shorter labels for the chart.
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

    def _save_final_metrics(self) -> None:
        """Write the evaluation history to one JSON file at the end of the run."""
        metrics_data = {
            "steps": self.x_data,
            "rewards": self.y_data,
            "reward_stds": self.y_std_data,
            "timestamps": [timestamp.isoformat() for timestamp in self.times[1:]],
            "all_metrics": self.all_metrics,
        }

        metrics_path = self.metrics_dir / FINAL_METRICS_FILENAME
        with open(metrics_path, "w", encoding="utf-8") as file_handle:
            json.dump(metrics_data, file_handle, indent=2)

        self.logger.info("Saved final metrics to %s", metrics_path)

    def _generate_video(self) -> None:
        """Make the final rollout video using the latest saved policy."""
        try:
            self.logger.info("Generating final video...")
            rollout = self._collect_rollout()
            frames = self._render_rollout_frames(rollout)

            video_path = self.videos_dir / FINAL_VIDEO_FILENAME
            self._write_video_opencv(frames, video_path, fps=VIDEO_FPS)
            self.logger.info("Saved final video to %s", video_path)
        except Exception as exc:
            self.logger.warning("Failed to generate final video: %s", exc)

    def _collect_rollout(self) -> list[Any]:
        """
        Run the final policy inside the simulator and collect the poses.

        The returned list is the raw material used later to render the video
        frame by frame.
        """
        inference_fn = self.make_inference_fn_cached
        jit_reset = jax.jit(self.env.reset)
        jit_step = jax.jit(self.env.step)
        jit_inference = jax.jit(inference_fn)

        # Start from a fixed random seed so repeated videos are comparable.
        rng = jax.random.PRNGKey(0)
        state = jit_reset(rng)
        rollout = [state.pipeline_state]

        # Step the policy forward until we hit the step limit or the episode ends.
        for rollout_step in range(ROLLOUT_VIDEO_STEPS):
            rng, key_sample = jax.random.split(rng)
            action, _ = jit_inference(state.obs, key_sample)
            state = jit_step(state, action)
            rollout.append(state.pipeline_state)
            if float(np.asarray(state.done)) > 0.0:
                self.logger.info("Final rollout terminated at step %s", rollout_step + 1)
                break

        return rollout

    def _render_rollout_frames(self, rollout: list[Any]) -> list[np.ndarray]:
        """
        Turn the saved simulator states into video frames.

        Each simulator state becomes one rendered image.
        """
        self.logger.info("Rendering frames...")

        mj_model = self.env.sys.mj_model
        renderer = mujoco.Renderer(mj_model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)

        frames: list[np.ndarray] = []
        for pipeline_state in rollout:
            # Rebuild the MuJoCo data object for this moment in time and render it.
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
        """Write the rendered frames into a normal MP4 file."""
        if not frames:
            raise ValueError("No frames to write")

        # Open the output video and then append frames one by one.
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()

    def finalize(self) -> None:
        """
        Write the final user-facing outputs for the run.

        This is called once after training has fully finished.
        """
        if not self.y_data:
            self.logger.warning("Skipping final artifacts because no evaluation data was recorded")
            return

        # Save the plot and JSON even if the video later fails.
        try:
            self._save_final_plot()
        except Exception as exc:
            self.logger.warning("Failed to save final plot: %s", exc)

        try:
            self._save_final_metrics()
        except Exception as exc:
            self.logger.warning("Failed to save final metrics: %s", exc)

        if self.env is None:
            return

        if self.make_inference_fn_cached is None:
            self.logger.warning("Skipping final video because no inference function was cached")
            return

        # Only attempt the expensive rendering step when we have everything needed.
        self._generate_video()

    def get_summary(self) -> dict[str, Any]:
        """Return the small set of headline numbers used in summaries and sweeps."""
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
