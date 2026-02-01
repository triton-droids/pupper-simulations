#!/usr/bin/env python3
"""
Bittle Quadruped Locomotion Training Script
============================================

Trains a PPO policy for the Bittle quadruped robot using Brax/MuJoCo.

Usage:
    python train_bittle.py [--config CONFIG] [--test] [--output_dir DIR]

Examples:
    # Quick test run (minimal training)
    python train.py --test

    # Full training run
    python train.py --config configs/bittle_default.py

    # Custom output directory
    python train.py --output_dir ./experiments/run_001
"""

import os

# Set MuJoCo rendering backend BEFORE importing any MuJoCo/Brax modules
os.environ["MUJOCO_GL"] = "egl"

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json
import warnings

# Suppress JAX overflow warning in type casting
warnings.filterwarnings("ignore", message="overflow encountered in cast")

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

# Import custom modules
from training_config import TrainingConfig
from training_helpers import setup_logging, policy_params_callback, parse_args
from training_monitor import TrainingMonitor
from domain_randomization import domain_randomize


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

    # Register environment
    envs.register_environment("bittle", BittleEnv)
    logger.info("Registered Bittle environment")

    # Create environment
    logger.info("Creating environment...")
    env = envs.get_environment("bittle", xml_path=xml_path)
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
        logger.info(f"Total time: {training_time:.2f}s ({training_time / 60:.2f} min)")
        logger.info(
            f"Time to JIT: {(monitor.times[1] - monitor.times[0]).total_seconds():.2f}s"
        )
        logger.info(
            f"Time to train: {(monitor.times[-1] - monitor.times[1]).total_seconds():.2f}s"
        )

        # Log summary statistics
        summary = monitor.get_summary()
        logger.info("")
        logger.info("Training Summary:")
        logger.info(
            f"  Final reward:      {summary['final_reward']:.4f} ± {summary['final_std']:.4f}"
        )
        logger.info(f"  Best reward:       {summary['best_reward']:.4f}")
        logger.info(f"  Mean reward:       {summary['mean_reward']:.4f}")
        logger.info(f"  Worst reward:      {summary['worst_reward']:.4f}")
        logger.info(f"  Num evaluations:   {summary['num_evaluations']}")
        logger.info("")

        # Save final model
        model_dir = output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "final_policy"

        logger.info(f"Saving final model to {model_path}...")
        model.save_params(str(model_path), params)
        logger.info("Model saved successfully")

        # Export policy to outputs/policy.pt for easy retrieval
        policy_export_path = output_dir / "policy.pt"
        logger.info(f"Exporting policy to {policy_export_path}...")
        model.save_params(str(policy_export_path), params)
        logger.info("Policy exported successfully")

        # Export policy to ONNX format
        from onnx_export import export_policy_to_onnx

        onnx_export_path = output_dir / "policy.onnx"
        logger.info(f"Exporting policy to ONNX format at {onnx_export_path}...")
        export_policy_to_onnx(params, str(onnx_export_path), deterministic=True)
        logger.info("ONNX export successful")

        # Save training summary
        summary_path = output_dir / "training_summary.json"
        summary["config"] = config.to_dict()
        summary["training_time"] = training_time
        summary["start_time"] = start_time.isoformat()
        summary["end_time"] = end_time.isoformat()

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved training summary to {summary_path}")

        logger.info("=" * 80)

        return {
            "success": True,
            "params": params,
            "make_inference_fn": make_inference_fn,
            "summary": summary,
        }

    except Exception as e:
        logger.error("=" * 80)
        logger.error("TRAINING FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error("=" * 80)

        return {
            "success": False,
            "error": str(e),
        }


def main():
    """Main entry point."""
    args = parse_args()

    # Create output directory
    if args.output_dir is None:
        mode = "test" if args.test else "train"
        output_dir = Path(f"./outputs/bittle_{mode}_latest")
    else:
        output_dir = Path(args.output_dir)

    # Remove old run if it exists and create fresh directory
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
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
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
