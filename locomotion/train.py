#!/usr/bin/env python3
"""
Bittle Quadruped Locomotion Training Script
============================================

Trains a PPO policy for the Bittle quadruped robot using Brax/MuJoCo.

Usage:
    python train_bittle.py [--config CONFIG] [--test] [--output_dir DIR]

Examples:
    # Quick test run (minimal training)
    python train_bittle.py --test
    
    # Full training run
    python train_bittle.py --config configs/bittle_default.py
    
    # Custom output directory
    python train_bittle.py --output_dir ./experiments/run_001
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

import jax
import jax.numpy as jp
import numpy as np
from matplotlib import pyplot as plt

from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.io import model, html
from flax.training import orbax_utils
from orbax import checkpoint as ocp

# Import custom environment
from bittle_env import BittleEnv


# ============================================================================
# Configuration
# ============================================================================

class TrainingConfig:
    """Training configuration with sensible defaults."""
    
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        
        if test_mode:
            # Minimal config for fast iteration (~6 min on A100)
            self.num_timesteps = 10_000
            self.num_evals = 2
            self.episode_length = 100
            self.num_envs = 128
            self.batch_size = 128
            self.unroll_length = 5
            self.num_minibatches = 2
            self.num_updates_per_batch = 1
        else:
            # Full training config (~30 min on A100)
            self.num_timesteps = 10_000_000
            self.num_evals = 10
            self.episode_length = 1000
            self.num_envs = 4096
            self.batch_size = 512
            self.unroll_length = 20
            self.num_minibatches = 8
            self.num_updates_per_batch = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'test_mode': self.test_mode,
            'num_timesteps': self.num_timesteps,
            'num_evals': self.num_evals,
            'episode_length': self.episode_length,
            'num_envs': self.num_envs,
            'batch_size': self.batch_size,
            'unroll_length': self.unroll_length,
            'num_minibatches': self.num_minibatches,
            'num_updates_per_batch': self.num_updates_per_batch,
        }


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Set up comprehensive logging to both console and file.
    
    Args:
        output_dir: Directory for log files
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger
    """
    # Create logs directory
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('bittle_training')
    logger.setLevel(level)
    
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler with detailed formatting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(log_dir / f'training_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


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
            states = [state.pipeline_state]

            # Run for ~5 seconds at 50Hz = 250 steps
            max_steps = 250
            for _ in range(max_steps):
                obs = state.obs
                action, _ = jit_inference(obs)
                state = jit_step(state, action)
                states.append(state.pipeline_state)

            # Generate HTML with video
            html_path = self.videos_dir / f'video_step_{num_steps:08d}.html'
            html_content = html.render(self.env.sys.tree_replace({'opt.timestep': self.env.sys.opt.timestep}), states)

            with open(html_path, 'w') as f:
                f.write(html_content)

            self.logger.info(f"Saved video to {html_path}")

            # Also save as latest
            latest_path = self.videos_dir / 'latest_video.html'
            with open(latest_path, 'w') as f:
                f.write(html_content)

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


def policy_params_callback(output_dir: Path, logger: logging.Logger, monitor=None):
    """
    Create a callback for saving policy checkpoints.

    Args:
        output_dir: Base output directory
        logger: Logger instance
        monitor: TrainingMonitor instance to update with inference function

    Returns:
        Callback function
    """
    ckpt_path = (output_dir / 'checkpoints').resolve()
    ckpt_path.mkdir(parents=True, exist_ok=True)

    def callback(current_step: int, make_policy: Any, params: Any) -> None:
        """Save checkpoint at current step."""
        # Update monitor with inference function for video generation
        if monitor is not None and monitor.make_inference_fn_cached is None:
            monitor.make_inference_fn_cached = make_policy(params)

        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f'step_{current_step:08d}'
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        logger.info(f"Saved checkpoint to {path}")

    return callback


# ============================================================================
# Domain Randomization
# ============================================================================

def domain_randomize(sys, rng):
    """
    Apply domain randomization for sim-to-real transfer.
    
    Randomizes:
    - Friction coefficients
    - Actuator gains and biases
    
    Args:
        sys: MuJoCo system
        rng: JAX random key
    
    Returns:
        Tuple of (randomized_sys, in_axes)
    """
    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        
        # Friction randomization
        friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)
        
        # Actuator gain randomization
        _, key = jax.random.split(key, 2)
        gain_range = (-5, 5)
        param = jax.random.uniform(
            key, (1,), minval=gain_range[0], maxval=gain_range[1]
        ) + sys.actuator_gainprm[:, 0]
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)
        
        return friction, gain, bias
    
    friction, gain, bias = rand(rng)
    
    # Set up in_axes for vmap
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_friction': 0,
        'actuator_gainprm': 0,
        'actuator_biasprm': 0,
    })
    
    sys = sys.tree_replace({
        'geom_friction': friction,
        'actuator_gainprm': gain,
        'actuator_biasprm': bias,
    })
    
    return sys, in_axes


# ============================================================================
# Main Training Function
# ============================================================================

def train_bittle(
    config: TrainingConfig,
    xml_path: str,
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Train Bittle locomotion policy.
    
    Args:
        config: Training configuration
        xml_path: Path to MuJoCo XML file
        output_dir: Output directory for checkpoints and logs
        logger: Logger instance
    
    Returns:
        Dictionary with training results
    """
    logger.info("=" * 80)
    logger.info("BITTLE LOCOMOTION TRAINING")
    logger.info("=" * 80)
    logger.info(f"Mode: {'TEST' if config.test_mode else 'FULL TRAINING'}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"XML path: {xml_path}")
    logger.info("")
    
    # Log configuration
    logger.info("Training Configuration:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key:30s}: {value}")
    logger.info("")
    
    # Check JAX devices
    devices = jax.devices()
    logger.info(f"JAX devices available: {len(devices)}")
    for i, device in enumerate(devices):
        logger.info(f"  Device {i}: {device}")
    logger.info("")
    
    # Set MuJoCo rendering backend
    os.environ['MUJOCO_GL'] = 'egl'
    logger.info("Set MUJOCO_GL=egl for rendering")
    logger.info("")
    
    # Register environment
    envs.register_environment('bittle', BittleEnv)
    logger.info("Registered Bittle environment")
    
    # Create environment
    logger.info("Creating environment...")
    env = envs.get_environment('bittle', xml_path=xml_path)
    logger.info("Environment created successfully")
    logger.info("")
    
    # Set up monitoring
    monitor = TrainingMonitor(output_dir, config.num_timesteps, logger, env=env)

    # Set up checkpoint callback
    checkpoint_callback = policy_params_callback(output_dir, logger, monitor=monitor)
    
    # Start training
    logger.info("Starting training...")
    start_time = datetime.now()
    
    try:
        make_inference_fn, params, _ = ppo.train(
            environment=env,
            num_timesteps=config.num_timesteps,
            num_evals=config.num_evals,
            episode_length=config.episode_length,
            num_envs=config.num_envs,
            batch_size=config.batch_size,
            unroll_length=config.unroll_length,
            num_minibatches=config.num_minibatches,
            num_updates_per_batch=config.num_updates_per_batch,
            progress_fn=monitor,
            policy_params_fn=checkpoint_callback,
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total time: {training_time:.2f}s ({training_time/60:.2f} min)")
        logger.info(f"Time to JIT: {(monitor.times[1] - monitor.times[0]).total_seconds():.2f}s")
        logger.info(f"Time to train: {(monitor.times[-1] - monitor.times[1]).total_seconds():.2f}s")
        
        # Log summary statistics
        summary = monitor.get_summary()
        logger.info("")
        logger.info("Training Summary:")
        logger.info(f"  Final reward:      {summary['final_reward']:.4f} ± {summary['final_std']:.4f}")
        logger.info(f"  Best reward:       {summary['best_reward']:.4f}")
        logger.info(f"  Mean reward:       {summary['mean_reward']:.4f}")
        logger.info(f"  Worst reward:      {summary['worst_reward']:.4f}")
        logger.info(f"  Num evaluations:   {summary['num_evaluations']}")
        logger.info("")
        
        # Save final model
        model_dir = output_dir / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / 'final_policy'
        
        logger.info(f"Saving final model to {model_path}...")
        model.save_params(str(model_path), params)
        logger.info("Model saved successfully")
        
        # Save training summary
        summary_path = output_dir / 'training_summary.json'
        summary['config'] = config.to_dict()
        summary['training_time'] = training_time
        summary['start_time'] = start_time.isoformat()
        summary['end_time'] = end_time.isoformat()
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved training summary to {summary_path}")
        
        logger.info("=" * 80)
        
        return {
            'success': True,
            'params': params,
            'make_inference_fn': make_inference_fn,
            'summary': summary,
        }
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("TRAINING FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error("=" * 80)
        
        return {
            'success': False,
            'error': str(e),
        }


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Bittle quadruped locomotion policy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  python train_bittle.py --test
  
  # Full training run
  python train_bittle.py
  
  # Custom output directory
  python train_bittle.py --output_dir ./experiments/run_001
        """
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (minimal training for fast iteration)'
    )
    
    parser.add_argument(
        '--xml_path',
        type=str,
        default='bittle_adapted_scene.xml',
        help='Path to MuJoCo XML scene file (default: bittle_adapted_scene.xml)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for checkpoints and logs (default: auto-generated)'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode = 'test' if args.test else 'train'
        output_dir = Path(f'./outputs/bittle_{mode}_{timestamp}')
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(output_dir, level=log_level)
    
    # Create configuration
    config = TrainingConfig(test_mode=args.test)
    
    # Train
    results = train_bittle(
        config=config,
        xml_path=args.xml_path,
        output_dir=output_dir,
        logger=logger,
    )
    
    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == '__main__':
    main()