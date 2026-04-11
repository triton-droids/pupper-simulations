#!/usr/bin/env python3
"""
Train a PPO locomotion policy for the Bittle quadruped.

High-level workflow
-------------------
``train.py`` is the orchestration layer for the locomotion package. It ties
together:

- the Brax environment from ``bittle_env.py``
- training presets from ``training_config.py``
- checkpointing and CLI helpers from ``training_helpers.py``
- plots and rollout videos from ``training_monitor.py``
- ONNX export from ``onnx_export.py``

That separation keeps the entry point focused on process flow rather than on
implementation details.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any


def configure_mujoco_backend() -> None:
    """
    Choose a default MuJoCo GL backend when the caller has not set one.

    Linux training nodes usually use EGL for headless rendering, while macOS
    generally needs GLFW. Windows is left unchanged because forcing the wrong
    backend there can break imports.
    """
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
    from locomotion.paths import OUTPUTS_ROOT
    from locomotion.training_config import TrainingConfig
    from locomotion.training_helpers import (
        parse_args,
        policy_params_callback,
        setup_logging,
    )
    from locomotion.training_monitor import TrainingMonitor
else:
    from .bittle_env import BittleEnv
    from .paths import OUTPUTS_ROOT
    from .training_config import TrainingConfig
    from .training_helpers import parse_args, policy_params_callback, setup_logging
    from .training_monitor import TrainingMonitor

warnings.filterwarnings("ignore", message="overflow encountered in cast")

import jax

from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo


def _import_onnx_exporter():
    """Import the ONNX exporter without assuming one package execution style."""
    if __package__ in (None, ""):
        from locomotion.onnx_export import export_policy_to_onnx
    else:
        from .onnx_export import export_policy_to_onnx
    return export_policy_to_onnx


def _build_summary_payload(
    summary: dict[str, Any],
    config: TrainingConfig,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, Any]:
    """Add metadata to the monitor summary before writing it to disk."""
    payload = dict(summary)
    payload["config"] = config.to_dict()
    payload["training_time"] = (end_time - start_time).total_seconds()
    payload["start_time"] = start_time.isoformat()
    payload["end_time"] = end_time.isoformat()
    return payload


def _log_training_config(config: TrainingConfig, logger: logging.Logger) -> None:
    """Log the training configuration in a compact aligned block."""
    logger.info("Training Configuration:")
    for key, value in config.to_dict().items():
        logger.info("  %-30s: %s", key, value)
    logger.info("")


def _log_jax_devices(logger: logging.Logger) -> None:
    """Log the JAX devices visible to the current runtime."""
    devices = jax.devices()
    logger.info("JAX devices available: %s", len(devices))
    for index, device in enumerate(devices):
        logger.info("  Device %s: %s", index, device)
    logger.info("")


def _log_training_summary(summary: dict[str, Any], logger: logging.Logger) -> None:
    """Log the post-training summary block if evaluations were recorded."""
    logger.info("")
    if not summary:
        logger.warning("Training finished without evaluation metrics to summarize")
        logger.info("")
        return

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
    logger.info("")


def _resolve_output_dir(output_dir_arg: str | None, *, test_mode: bool) -> Path:
    """Resolve the output directory from CLI input or package defaults."""
    if output_dir_arg is not None:
        return Path(output_dir_arg).expanduser()

    mode = "test" if test_mode else "train"
    return OUTPUTS_ROOT / f"bittle_{mode}_latest"


def _prepare_output_dir(output_dir: Path) -> None:
    """Start each run from a clean output directory."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def train_bittle(
    config: TrainingConfig,
    xml_path: str,
    output_dir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    Train a Bittle locomotion policy and write all run artifacts to ``output_dir``.

    Returns:
        A small result dictionary so sweep runners can inspect success status,
        summary metrics, and the final policy objects.
    """
    logger.info("=" * 80)
    logger.info("BITTLE LOCOMOTION TRAINING")
    logger.info("=" * 80)
    logger.info("Mode: %s", "TEST" if config.test_mode else "FULL TRAINING")
    logger.info("Output directory: %s", output_dir)
    logger.info("XML path: %s", xml_path)
    logger.info("")

    _log_training_config(config, logger)
    _log_jax_devices(logger)

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
    except Exception as exc:
        logger.error("=" * 80)
        logger.error("TRAINING FAILED")
        logger.error("=" * 80)
        logger.error("Error: %s", exc, exc_info=True)
        logger.error("=" * 80)
        return {"success": False, "error": str(exc)}

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
    _log_training_summary(summary, logger)

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = model_dir / "final_policy"

    logger.info("Saving final model to %s...", final_model_path)
    model.save_params(str(final_model_path), params)
    logger.info("Model saved successfully")

    policy_export_path = output_dir / "policy.pt"
    logger.info("Exporting policy to %s...", policy_export_path)
    model.save_params(str(policy_export_path), params)
    logger.info("Policy exported successfully")

    export_policy_to_onnx = _import_onnx_exporter()
    onnx_export_path = output_dir / "policy.onnx"
    logger.info("Exporting policy to ONNX format at %s...", onnx_export_path)
    export_policy_to_onnx(params, str(onnx_export_path), deterministic=True)
    logger.info("ONNX export successful")

    summary_payload = _build_summary_payload(summary, config, start_time, end_time)
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(summary_payload, file_handle, indent=2)
    logger.info("Saved training summary to %s", summary_path)
    logger.info("=" * 80)

    return {
        "success": True,
        "params": params,
        "make_inference_fn": make_inference_fn,
        "summary": summary_payload,
    }


def main() -> None:
    """Parse the CLI and launch one training run."""
    args = parse_args()
    output_dir = _resolve_output_dir(args.output_dir, test_mode=args.test)
    _prepare_output_dir(output_dir)

    logger = setup_logging(output_dir, level=getattr(logging, args.log_level.upper()))
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
