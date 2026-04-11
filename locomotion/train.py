#!/usr/bin/env python3
"""
Bittle quadruped locomotion training entry point.

Trains a PPO policy for the Bittle quadruped robot using Brax/MuJoCo.

Usage:
    python locomotion/train.py [--test] [--output_dir DIR]
    python -m locomotion.train [--test] [--output_dir DIR]
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any


def configure_mujoco_backend() -> None:
    """Choose a default MuJoCo GL backend that matches the current platform."""
    if "MUJOCO_GL" in os.environ:
        return

    system = platform.system()
    if system == "Linux":
        os.environ["MUJOCO_GL"] = "egl"
    elif system == "Darwin":
        os.environ["MUJOCO_GL"] = "glfw"


configure_mujoco_backend()

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from locomotion.bittle_env import BittleEnv
    from locomotion.paths import REPO_ROOT
    from locomotion.training_config import TrainingConfig
    from locomotion.training_helpers import (
        parse_args,
        policy_params_callback,
        setup_logging,
    )
    from locomotion.training_monitor import TrainingMonitor
else:
    from .bittle_env import BittleEnv
    from .paths import REPO_ROOT
    from .training_config import TrainingConfig
    from .training_helpers import parse_args, policy_params_callback, setup_logging
    from .training_monitor import TrainingMonitor

warnings.filterwarnings("ignore", message="overflow encountered in cast")

import jax

from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo


def train_bittle(
    config: TrainingConfig,
    xml_path: str,
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Train a Bittle locomotion policy."""
    logger.info("=" * 80)
    logger.info("BITTLE LOCOMOTION TRAINING")
    logger.info("=" * 80)
    logger.info("Mode: %s", "TEST" if config.test_mode else "FULL TRAINING")
    logger.info("Output directory: %s", output_dir)
    logger.info("XML path: %s", xml_path)
    logger.info("")

    logger.info("Training Configuration:")
    for key, value in config.to_dict().items():
        logger.info("  %-30s: %s", key, value)
    logger.info("")

    devices = jax.devices()
    logger.info("JAX devices available: %s", len(devices))
    for i, device in enumerate(devices):
        logger.info("  Device %s: %s", i, device)
    logger.info("")

    envs.register_environment("bittle", BittleEnv)
    logger.info("Registered Bittle environment")

    logger.info("Creating environment...")
    env = envs.get_environment("bittle", xml_path=xml_path)
    logger.info("Environment created successfully")
    logger.info("Environment sizes: obs=%s act=%s", env.observation_size, env.action_size)
    logger.info("")

    monitor = TrainingMonitor(output_dir, config.num_timesteps, logger, env=env)
    checkpoint_callback = policy_params_callback(output_dir, logger, monitor=monitor)

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
        logger.info("Total time: %.2fs (%.2f min)", training_time, training_time / 60)

        if len(monitor.times) >= 2:
            logger.info(
                "Time to JIT: %.2fs",
                (monitor.times[1] - monitor.times[0]).total_seconds(),
            )
            logger.info(
                "Time to train: %.2fs",
                (monitor.times[-1] - monitor.times[1]).total_seconds(),
            )

        summary = monitor.get_summary()
        logger.info("")
        if summary:
            logger.info("Training Summary:")
            logger.info(
                "  Final reward:      %.4f +/- %.4f",
                summary["final_reward"],
                summary["final_std"],
            )
            logger.info("  Best reward:       %.4f", summary["best_reward"])
            logger.info("  Mean reward:       %.4f", summary["mean_reward"])
            logger.info("  Worst reward:      %.4f", summary["worst_reward"])
            logger.info("  Num evaluations:   %s", summary["num_evaluations"])
        else:
            logger.warning("Training finished without evaluation metrics to summarize")
        logger.info("")

        model_dir = output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "final_policy"

        logger.info("Saving final model to %s...", model_path)
        model.save_params(str(model_path), params)
        logger.info("Model saved successfully")

        policy_export_path = output_dir / "policy.pt"
        logger.info("Exporting policy to %s...", policy_export_path)
        model.save_params(str(policy_export_path), params)
        logger.info("Policy exported successfully")

        if __package__ in (None, ""):
            from locomotion.onnx_export import export_policy_to_onnx
        else:
            from .onnx_export import export_policy_to_onnx

        onnx_export_path = output_dir / "policy.onnx"
        logger.info("Exporting policy to ONNX format at %s...", onnx_export_path)
        export_policy_to_onnx(params, str(onnx_export_path), deterministic=True)
        logger.info("ONNX export successful")

        summary_payload = dict(summary)
        summary_payload["config"] = config.to_dict()
        summary_payload["training_time"] = training_time
        summary_payload["start_time"] = start_time.isoformat()
        summary_payload["end_time"] = end_time.isoformat()

        summary_path = output_dir / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        logger.info("Saved training summary to %s", summary_path)
        logger.info("=" * 80)

        return {
            "success": True,
            "params": params,
            "make_inference_fn": make_inference_fn,
            "summary": summary_payload,
        }

    except Exception as exc:
        logger.error("=" * 80)
        logger.error("TRAINING FAILED")
        logger.error("=" * 80)
        logger.error("Error: %s", exc, exc_info=True)
        logger.error("=" * 80)

        return {
            "success": False,
            "error": str(exc),
        }


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.output_dir is None:
        mode = "test" if args.test else "train"
        output_dir = REPO_ROOT / "outputs" / f"bittle_{mode}_latest"
    else:
        output_dir = Path(args.output_dir).expanduser()

    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(output_dir, level=log_level)
    config = TrainingConfig(test_mode=args.test)

    results = train_bittle(
        config=config,
        xml_path=args.xml_path,
        output_dir=output_dir,
        logger=logger,
    )

    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
