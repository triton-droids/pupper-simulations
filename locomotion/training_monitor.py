import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import jax
import mujoco
import numpy as np
from matplotlib import pyplot as plt


class TrainingMonitor:
    """Monitor training progress with metrics tracking and visualization."""

    def __init__(
        self,
        output_dir: Path,
        num_timesteps: int,
        logger: logging.Logger,
        env=None,
        make_inference_fn=None,
    ):
        self.output_dir = output_dir
        self.num_timesteps = num_timesteps
        self.logger = logger
        self.env = env
        self.make_inference_fn_cached = make_inference_fn

        self.x_data: List[int] = []
        self.y_data: List[float] = []
        self.y_std_data: List[float] = []
        self.times: List[datetime] = [datetime.now()]
        self.all_metrics: List[Dict[str, float]] = []

        self.plots_dir = output_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_dir = output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.videos_dir = output_dir / "videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, num_steps: int, metrics: Dict[str, Any]) -> None:
        """Record evaluation metrics and generate artifacts."""
        self.times.append(datetime.now())
        time_delta = (self.times[-1] - self.times[-2]).total_seconds()

        episode_reward = float(metrics["eval/episode_reward"])
        episode_reward_std = float(metrics.get("eval/episode_reward_std", 0.0))

        self.x_data.append(num_steps)
        self.y_data.append(episode_reward)
        self.y_std_data.append(episode_reward_std)
        self.all_metrics.append(
            {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float, np.ndarray))
            }
        )

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
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float, np.ndarray)):
                if isinstance(value, np.ndarray):
                    value = float(value)
                self.logger.debug("  %-30s: %.6f", key, value)

        if len(self.y_data) > 1:
            improvement = self.y_data[-1] - self.y_data[-2]
            self.logger.info("Reward change:      %+0.4f", improvement)
            self.logger.info("Best so far:        %.4f", max(self.y_data))
            self.logger.info("Worst so far:       %.4f", min(self.y_data))

        self.logger.info("=" * 80)

        self._plot_progress(num_steps, metrics)
        self._save_metrics(num_steps)

        if self.env is not None and self.make_inference_fn_cached is not None:
            self._generate_video(num_steps)

    def _plot_progress(self, num_steps: int, metrics: Dict[str, Any]) -> None:
        """Generate and save training progress plots."""
        reward_components = {
            key: value
            for key, value in metrics.items()
            if key.startswith("eval/episode_reward/")
            or (key.startswith("eval/") and "reward" in key.lower() and key != "eval/episode_reward")
        }

        fig = self._build_progress_figure(
            num_steps=num_steps,
            reward_components=reward_components,
            title_prefix="Training Progress",
            component_title="Reward Components",
        )
        plot_path = self.plots_dir / f"progress_step_{num_steps:08d}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        latest_fig = self._build_progress_figure(
            num_steps=num_steps,
            reward_components=reward_components,
            title_prefix="Final Progress",
            component_title="Final Reward Components",
        )
        latest_path = self.plots_dir / "latest_progress.png"
        latest_fig.savefig(latest_path, dpi=150, bbox_inches="tight")
        plt.close(latest_fig)

        self.logger.debug("Saved plot to %s", plot_path)

    def _build_progress_figure(
        self,
        num_steps: int,
        reward_components: Dict[str, Any],
        title_prefix: str,
        component_title: str,
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.errorbar(
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
        ax1.axhline(y=0, color="r", linestyle="--", alpha=0.3, label="Zero reward")
        ax1.set_xlim([0, self.num_timesteps * 1.1])

        if self.y_data:
            y_min = min(self.y_data) - max(self.y_std_data) * 1.2
            y_max = max(self.y_data) + max(self.y_std_data) * 1.2
            ax1.set_ylim([y_min, y_max])

        ax1.set_xlabel("Environment Steps", fontsize=12)
        ax1.set_ylabel("Reward per Episode", fontsize=12)
        ax1.set_title(f"{title_prefix} (Step {num_steps:,})", fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        if reward_components:
            names = [
                key.replace("eval/episode_reward/", "").replace("eval/", "")
                for key in reward_components.keys()
            ]
            values = [float(value) for value in reward_components.values()]
            colors = ["green" if value > 0 else "red" for value in values]

            ax2.barh(range(len(names)), values, color=colors, alpha=0.6)
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names, fontsize=9)
            ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
            ax2.set_xlabel("Reward Contribution", fontsize=12)
            ax2.set_title(component_title, fontsize=12)
            ax2.grid(True, alpha=0.3, axis="x")
        else:
            ax2.text(
                0.5,
                0.5,
                "No component\nbreakdown available",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])

        fig.tight_layout()
        return fig

    def _save_metrics(self, num_steps: int) -> None:
        """Save metrics to JSON files."""
        metrics_data = {
            "steps": self.x_data,
            "rewards": self.y_data,
            "reward_stds": self.y_std_data,
            "timestamps": [timestamp.isoformat() for timestamp in self.times[1:]],
            "all_metrics": self.all_metrics,
        }

        metrics_path = self.metrics_dir / f"metrics_step_{num_steps:08d}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

        latest_path = self.metrics_dir / "latest_metrics.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)

        self.logger.debug("Saved metrics to %s", metrics_path)

    def _generate_video(self, num_steps: int) -> None:
        """Generate and save a video of the current policy."""
        try:
            self.logger.info("Generating video...")

            inference_fn = self.make_inference_fn_cached
            jit_reset = jax.jit(self.env.reset)
            jit_step = jax.jit(self.env.step)
            jit_inference = jax.jit(inference_fn)

            rng = jax.random.PRNGKey(0)
            state = jit_reset(rng)
            rollout = [state.pipeline_state]

            for _ in range(250):
                rng, key_sample = jax.random.split(rng)
                action, _ = jit_inference(state.obs, key_sample)
                state = jit_step(state, action)
                rollout.append(state.pipeline_state)

            self.logger.info("Rendering frames...")

            mj_model = self.env.sys.mj_model
            renderer = mujoco.Renderer(mj_model, height=480, width=640)

            frames = []
            for pipeline_state in rollout:
                mj_data = mujoco.MjData(mj_model)
                mj_data.qpos[:] = np.array(pipeline_state.q)
                mj_data.qvel[:] = np.array(pipeline_state.qd)
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data)
                frames.append(renderer.render())

            renderer.close()

            video_path = self.videos_dir / f"video_step_{num_steps:08d}.mp4"
            self._write_video_opencv(frames, video_path, fps=50)
            self.logger.info("Saved video to %s", video_path)

            latest_path = self.videos_dir / "latest_video.mp4"
            self._write_video_opencv(frames, latest_path, fps=50)

        except Exception as exc:
            self.logger.warning("Failed to generate video: %s", exc)

    def _write_video_opencv(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        fps: int = 50,
    ) -> None:
        """Write frames to a video file using OpenCV."""
        if not frames:
            raise ValueError("No frames to write")

        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not out.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")

        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()

    def get_summary(self) -> Dict[str, Any]:
        """Return summary statistics from recorded evaluations."""
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
