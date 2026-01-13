import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import jax
import numpy as np
from matplotlib import pyplot as plt
import mediapy as media
import mujoco

# ============================================================================
# Training Callbacks and Visualization
# ============================================================================

class TrainingMonitor:
    """Monitor training progress with metrics tracking and visualization."""

    def __init__(self, output_dir: Path, num_timesteps: int, logger: logging.Logger, env=None, make_inference_fn=None):
        self.output_dir = output_dir
        self.num_timesteps = num_timesteps
        self.logger = logger
        self.env = env
        self.make_inference_fn_cached = None

        # Tracking data
        self.x_data: List[int] = []
        self.y_data: List[float] = []
        self.y_std_data: List[float] = []
        self.times: List[datetime] = [datetime.now()]
        self.all_metrics: List[Dict[str, float]] = []

        # Create plots directory
        self.plots_dir = output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Metrics directory
        self.metrics_dir = output_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Videos directory
        self.videos_dir = output_dir / 'videos'
        self.videos_dir.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, num_steps: int, metrics: Dict[str, Any]) -> None:
        """
        Callback function for training progress.
        
        Args:
            num_steps: Current training step
            metrics: Dictionary of evaluation metrics
        """
        self.times.append(datetime.now())
        time_delta = (self.times[-1] - self.times[-2]).total_seconds()
        
        # Extract key metrics
        episode_reward = float(metrics['eval/episode_reward'])
        episode_reward_std = float(metrics.get('eval/episode_reward_std', 0.0))
        
        # Store data
        self.x_data.append(num_steps)
        self.y_data.append(episode_reward)
        self.y_std_data.append(episode_reward_std)
        self.all_metrics.append({k: float(v) for k, v in metrics.items() 
                                 if isinstance(v, (int, float, np.ndarray))})
        
        # Log to console
        self.logger.info("=" * 80)
        self.logger.info(f"EVALUATION AT STEP {num_steps:,}")
        self.logger.info("=" * 80)
        self.logger.info(f"Episode Reward:     {episode_reward:.4f} ± {episode_reward_std:.4f}")
        self.logger.info(f"Time since last:    {time_delta:.2f}s")
        
        # Log all metrics (debug level)
        self.logger.debug("All available metrics:")
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float, np.ndarray)):
                if isinstance(value, np.ndarray):
                    value = float(value)
                self.logger.debug(f"  {key:30s}: {value:.6f}")
        
        # Calculate statistics
        if len(self.y_data) > 1:
            improvement = self.y_data[-1] - self.y_data[-2]
            self.logger.info(f"Reward change:      {improvement:+.4f}")
            self.logger.info(f"Best so far:        {max(self.y_data):.4f}")
            self.logger.info(f"Worst so far:       {min(self.y_data):.4f}")
        
        self.logger.info("=" * 80)
        
        # Generate plots
        self._plot_progress(num_steps, metrics)

        # Save metrics to JSON
        self._save_metrics(num_steps)

        # Generate video if we have the necessary components
        if self.env is not None and self.make_inference_fn_cached is not None:
            self._generate_video(num_steps, metrics)
    
    def _plot_progress(self, num_steps: int, metrics: Dict[str, Any]) -> None:
        """Generate and save training progress plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: Reward over time
        ax1.errorbar(
            self.x_data, self.y_data, yerr=self.y_std_data,
            marker='o', capsize=5, capthick=2,
            linewidth=2, markersize=8, label='Episode Reward'
        )
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Zero reward')
        ax1.set_xlim([0, self.num_timesteps * 1.1])
        
        # Auto-adjust y-limits based on data
        if self.y_data:
            y_min = min(self.y_data) - max(self.y_std_data) * 1.2
            y_max = max(self.y_data) + max(self.y_std_data) * 1.2
            ax1.set_ylim([y_min, y_max])
        
        ax1.set_xlabel('Environment Steps', fontsize=12)
        ax1.set_ylabel('Reward per Episode', fontsize=12)
        ax1.set_title(f'Training Progress (Step {num_steps:,})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Right plot: Reward components
        reward_components = {
            k: v for k, v in metrics.items()
            if k.startswith('eval/episode_reward/') or
               (k.startswith('eval/') and 'reward' in k.lower() and k != 'eval/episode_reward')
        }
        
        if reward_components:
            names = [
                k.replace('eval/episode_reward/', '').replace('eval/', '')
                for k in reward_components.keys()
            ]
            values = [float(v) for v in reward_components.values()]
            
            colors = ['green' if v > 0 else 'red' for v in values]
            ax2.barh(range(len(names)), values, color=colors, alpha=0.6)
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names, fontsize=9)
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Reward Contribution', fontsize=12)
            ax2.set_title('Reward Components', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='x')
        else:
            ax2.text(
                0.5, 0.5, 'No component\nbreakdown available',
                ha='center', va='center', fontsize=12
            )
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / f'progress_step_{num_steps:08d}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Also save as latest
        latest_path = self.plots_dir / 'latest_progress.png'
        plt.figure(figsize=(14, 5))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.errorbar(
            self.x_data, self.y_data, yerr=self.y_std_data,
            marker='o', capsize=5, capthick=2,
            linewidth=2, markersize=8, label='Episode Reward'
        )
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Zero reward')
        ax1.set_xlim([0, self.num_timesteps * 1.1])
        if self.y_data:
            y_min = min(self.y_data) - max(self.y_std_data) * 1.2
            y_max = max(self.y_data) + max(self.y_std_data) * 1.2
            ax1.set_ylim([y_min, y_max])
        ax1.set_xlabel('Environment Steps', fontsize=12)
        ax1.set_ylabel('Reward per Episode', fontsize=12)
        ax1.set_title(f'Final Progress (Step {num_steps:,})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        if reward_components:
            names = [
                k.replace('eval/episode_reward/', '').replace('eval/', '')
                for k in reward_components.keys()
            ]
            values = [float(v) for v in reward_components.values()]
            colors = ['green' if v > 0 else 'red' for v in values]
            ax2.barh(range(len(names)), values, color=colors, alpha=0.6)
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names, fontsize=9)
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Reward Contribution', fontsize=12)
            ax2.set_title('Final Reward Components', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(latest_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.debug(f"Saved plot to {plot_path}")
    
    def _save_metrics(self, num_steps: int) -> None:
        """Save metrics to JSON file."""
        metrics_data = {
            'steps': self.x_data,
            'rewards': self.y_data,
            'reward_stds': self.y_std_data,
            'timestamps': [t.isoformat() for t in self.times[1:]],  # Skip initial time
            'all_metrics': self.all_metrics,
        }
        
        metrics_path = self.metrics_dir / f'metrics_step_{num_steps:08d}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Also save as latest
        latest_path = self.metrics_dir / 'latest_metrics.json'
        with open(latest_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.logger.debug(f"Saved metrics to {metrics_path}")

    def _generate_video(self, num_steps: int, metrics: Dict[str, Any]) -> None:
        """Generate and save a video of the current policy."""
        try:
            self.logger.info("Generating video...")

            # Create inference function if not already created
            inference_fn = self.make_inference_fn_cached

            # Run a single episode to collect states
            jit_reset = jax.jit(self.env.reset)
            jit_step = jax.jit(self.env.step)
            jit_inference = jax.jit(inference_fn)

            rng = jax.random.PRNGKey(0)
            state = jit_reset(rng)
            rollout = [state.pipeline_state]

            # Run for ~5 seconds at 50Hz = 250 steps
            max_steps = 250
            for _ in range(max_steps):
                rng, key_sample = jax.random.split(rng)
                obs = state.obs
                action, _ = jit_inference(obs, key_sample)
                state = jit_step(state, action)
                rollout.append(state.pipeline_state)

            # Render frames from states
            self.logger.info("Rendering frames...")

            # Create MuJoCo model from Brax system
            mj_model = self.env.sys.mj_model

            # Create renderer with EGL backend for headless rendering
            renderer = mujoco.Renderer(mj_model, height=480, width=640)

            frames = []
            for pipeline_state in rollout:
                # Create MuJoCo data from pipeline state
                mj_data = mujoco.MjData(mj_model)
                mj_data.qpos[:] = np.array(pipeline_state.q)
                mj_data.qvel[:] = np.array(pipeline_state.qd)

                # Forward kinematics
                mujoco.mj_forward(mj_model, mj_data)

                # Update renderer and capture frame
                renderer.update_scene(mj_data)
                pixels = renderer.render()
                frames.append(pixels)

            renderer.close()

            # Convert to numpy array
            frames = np.array(frames)

            # Save as MP4
            video_path = self.videos_dir / f'video_step_{num_steps:08d}.mp4'
            media.write_video(video_path, frames, fps=50)
            self.logger.info(f"Saved video to {video_path}")

            # Also save as latest
            latest_path = self.videos_dir / 'latest_video.mp4'
            media.write_video(latest_path, frames, fps=50)

        except Exception as e:
            self.logger.warning(f"Failed to generate video: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from training."""
        if not self.y_data:
            return {}
        
        return {
            'final_reward': self.y_data[-1],
            'final_std': self.y_std_data[-1],
            'best_reward': max(self.y_data),
            'worst_reward': min(self.y_data),
            'mean_reward': np.mean(self.y_data),
            'total_time': (self.times[-1] - self.times[0]).total_seconds(),
            'num_evaluations': len(self.y_data),
        }