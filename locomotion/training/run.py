#!/usr/bin/env python3
"""
Main entrypoint for training the simulated robot.

This file is the conductor for the whole training job. It does not contain the
robot physics or the reward rules itself. Instead, it lines up the steps in a
human-friendly order:

1. pick the task and settings
2. create the simulator environment
3. hand that environment to the PPO trainer
4. save the finished policy and summary files
5. write the final plot, metrics, and video
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def configure_mujoco_backend() -> None:
    """
    Pick a safe default graphics backend for MuJoCo.

    MuJoCo needs a rendering backend even for headless jobs that only create
    videos at the end. Different operating systems prefer different backends,
    so this chooses a default only when the caller has not already chosen one.
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
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from locomotion.paths import OUTPUTS_ROOT, resolve_output_path
    from locomotion.tasks import normalize_task_name
    from locomotion.tasks.bittle_dance_env import BittleDanceEnv
    from locomotion.tasks.bittle_walk_env import BittleWalkingEnv
    from locomotion.training.config import TrainingConfig
    from locomotion.training.helpers import (
        parse_args,
        policy_params_callback,
        setup_logging,
    )
    from locomotion.training.monitor import TrainingMonitor
else:
    from ..paths import OUTPUTS_ROOT, resolve_output_path
    from ..tasks import normalize_task_name
    from ..tasks.bittle_dance_env import BittleDanceEnv
    from ..tasks.bittle_walk_env import BittleWalkingEnv
    from .config import TrainingConfig
    from .helpers import parse_args, policy_params_callback, setup_logging
    from .monitor import TrainingMonitor

warnings.filterwarnings("ignore", message="overflow encountered in cast")

import jax

from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Describe one trainable Bittle task exposed by the CLI."""

    env_name: str
    env_class: type[Any]
    banner_title: str
    output_prefix: str
    description: str


TASK_SPECS = {
    "walking": TaskSpec(
        env_name="bittle_walking",
        env_class=BittleWalkingEnv,
        banner_title="BITTLE WALKING TRAINING",
        output_prefix="bittle_walking",
        description="walk at the asked speed and turn rate",
    ),
    "dance": TaskSpec(
        env_name="bittle_dance",
        env_class=BittleDanceEnv,
        banner_title="BITTLE DANCE TRAINING",
        output_prefix="bittle_dance",
        description="phase-conditioned rhythmic dance",
    ),
}
GPU_PLATFORM_NAMES = frozenset({"gpu", "cuda", "rocm"})


def _get_task_spec(task_name: str) -> TaskSpec:
    """
    Fetch the small info bundle that describes one task.

    This keeps task-specific names and labels in one place so the rest of the
    code can say "give me the dance task" without hardcoding strings everywhere.
    """
    task_name = normalize_task_name(task_name)
    try:
        return TASK_SPECS[task_name]
    except KeyError as exc:
        valid = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"Unknown task '{task_name}'. Expected one of: {valid}") from exc


def _import_onnx_exporter():
    """Load the ONNX export helper in a way that works from either run style."""
    if __package__ in (None, ""):
        from locomotion.onnx_export import export_policy_to_onnx
    else:
        from ..onnx_export import export_policy_to_onnx
    return export_policy_to_onnx


def _build_summary_payload(
    summary: dict[str, Any],
    config: TrainingConfig,
    start_time: datetime,
    end_time: datetime,
    *,
    task_name: str,
    env_overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Add extra facts to the summary before saving it.

    The monitor knows the reward history, but this helper adds context like the
    task name, chosen settings, and start/end times.
    """
    payload = dict(summary)
    payload["task"] = normalize_task_name(task_name)
    payload["config"] = config.to_dict()
    payload["task_hyperparameters"] = env_overrides
    payload["training_time"] = (end_time - start_time).total_seconds()
    payload["start_time"] = start_time.isoformat()
    payload["end_time"] = end_time.isoformat()
    return payload


def _log_training_config(
    config: TrainingConfig,
    logger: logging.Logger,
    *,
    task_name: str,
) -> None:
    """Print the chosen task and settings in a readable block for the log."""
    logger.info("Task:")
    logger.info("  %-30s: %s", "name", task_name)
    logger.info("  %-30s: %s", "description", _get_task_spec(task_name).description)
    logger.info("")
    logger.info("Training Configuration:")
    for key, value in config.to_dict().items():
        logger.info("  %-30s: %s", key, value)
    logger.info("")


def _log_jax_devices(logger: logging.Logger) -> tuple[Any, ...]:
    """Print which CPU/GPU devices JAX can see before training starts."""
    devices = tuple(jax.devices())
    logger.info("JAX devices available: %s", len(devices))
    for index, device in enumerate(devices):
        logger.info("  Device %s: %s", index, device)
    logger.info("")
    return devices


def _parse_visible_devices(csv: str | None) -> list[str]:
    """Split a CUDA-visible-devices string into clean GPU ids."""
    if csv is None:
        return []
    return [gpu_id.strip() for gpu_id in csv.split(",") if gpu_id.strip()]


def _build_gpu_requirement_reason(*, require_gpu: bool) -> str | None:
    """Explain why this run should refuse a CPU-only JAX fallback."""
    reasons: list[str] = []
    if require_gpu:
        reasons.append("--require_gpu was set")

    visible_devices = _parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible_devices:
        reasons.append(f"CUDA_VISIBLE_DEVICES={','.join(visible_devices)}")

    if not reasons:
        return None
    return " and ".join(reasons)


def _has_gpu_backed_jax_device(devices: tuple[Any, ...]) -> bool:
    """Return whether JAX exposed at least one accelerator device."""
    return any(str(getattr(device, "platform", "")).lower() in GPU_PLATFORM_NAMES for device in devices)


def _fail_if_required_gpu_missing(
    logger: logging.Logger,
    devices: tuple[Any, ...],
    *,
    require_gpu: bool,
) -> str | None:
    """Return an error message when a GPU-backed run fell back to CPU-only JAX."""
    reason = _build_gpu_requirement_reason(require_gpu=require_gpu)
    if reason is None or _has_gpu_backed_jax_device(devices):
        return None

    platforms = sorted(
        {
            str(getattr(device, "platform", "")).lower()
            for device in devices
            if str(getattr(device, "platform", "")).strip()
        }
    )
    platform_summary = ", ".join(platforms) if platforms else "<none>"
    error = (
        "GPU was required because "
        f"{reason}, but JAX only exposed platform(s): {platform_summary}. "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}."
    )

    logger.error("=" * 80)
    logger.error("GPU PRECHECK FAILED")
    logger.error("=" * 80)
    logger.error("%s", error)
    logger.error("Refusing to continue with a CPU-only fallback.")
    logger.error("=" * 80)
    return error


def _log_training_summary(summary: dict[str, Any], logger: logging.Logger) -> None:
    """Print the final reward summary after training finishes."""
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


def _resolve_output_dir(
    output_dir_arg: str | None,
    *,
    test_mode: bool,
    task_name: str,
) -> Path:
    """
    Decide where this run should save its files.

    If the user passed an explicit output folder, use it. Otherwise, build the
    usual "latest train/test" folder for the selected task.
    """
    if output_dir_arg is not None:
        return resolve_output_path(output_dir_arg)

    mode = "test" if test_mode else "train"
    return OUTPUTS_ROOT / f"{_get_task_spec(task_name).output_prefix}_{mode}_latest"


def _prepare_output_dir(output_dir: Path) -> None:
    """
    Make sure the output folder starts empty for this run.

    That prevents old artifacts from a previous run from being mistaken for new
    results.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def train_bittle(
    config: TrainingConfig,
    xml_path: str,
    output_dir: Path,
    logger: logging.Logger,
    task_name: str = "walking",
    env_overrides: dict[str, Any] | None = None,
    require_gpu: bool = False,
) -> dict[str, Any]:
    """
    Run one full training job and save the outputs.

    In everyday terms, this function is:

    - create the environment
    - start PPO training
    - save the trained policy in a few formats
    - save the human-facing artifacts and summary

    It returns a small status dictionary so the sweep runner can tell whether
    the job finished cleanly.
    """
    output_dir = resolve_output_path(output_dir)
    task_name = normalize_task_name(task_name)
    task_spec = _get_task_spec(task_name)
    env_overrides = dict(env_overrides or {})

    # Log the "what are we about to do?" header first so the run log is readable.
    logger.info("=" * 80)
    logger.info(task_spec.banner_title)
    logger.info("=" * 80)
    logger.info("Mode: %s", "TEST" if config.test_mode else "FULL TRAINING")
    logger.info("Task: %s", task_name)
    logger.info("Output directory: %s", output_dir)
    logger.info("XML path: %s", xml_path)
    logger.info("Task hyperparameters: %s", env_overrides if env_overrides else "{}")
    logger.info("")

    _log_training_config(config, logger, task_name=task_name)
    devices = _log_jax_devices(logger)
    gpu_error = _fail_if_required_gpu_missing(
        logger,
        devices,
        require_gpu=require_gpu,
    )
    if gpu_error is not None:
        return {"success": False, "error": gpu_error}

    # Register the chosen environment class with Brax, then build one instance
    # so PPO knows what world it will be training in.
    envs.register_environment(task_spec.env_name, task_spec.env_class)
    logger.info("Registered %s environment", task_spec.env_name)

    logger.info("Creating environment...")
    env = envs.get_environment(task_spec.env_name, xml_path=xml_path, **env_overrides)
    logger.info("Environment created successfully")
    logger.info("Environment sizes: obs=%s act=%s", env.observation_size, env.action_size)
    logger.info("")

    monitor = TrainingMonitor(output_dir, config.num_timesteps, logger, env=env)
    checkpoint_callback = policy_params_callback(output_dir, logger, monitor=monitor)

    # Hand the whole job over to Brax PPO. This is the part that actually does
    # the repeated trial-and-error learning.
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

    # Training is done. From here on, we are packaging the results.
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

    # Save the final learned parameters in the native Brax/JAX format.
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = model_dir / "final_policy"

    logger.info("Saving final model to %s...", final_model_path)
    model.save_params(str(final_model_path), params)
    logger.info("Model saved successfully")

    # Save a second copy in the legacy project location used by other tools.
    policy_export_path = output_dir / "policy.pt"
    logger.info("Exporting policy to %s...", policy_export_path)
    model.save_params(str(policy_export_path), params)
    logger.info("Policy exported successfully")

    # Export ONNX so the policy can be loaded outside the training stack.
    export_policy_to_onnx = _import_onnx_exporter()
    onnx_export_path = output_dir / "policy.onnx"
    logger.info("Exporting policy to ONNX format at %s...", onnx_export_path)
    export_policy_to_onnx(params, str(onnx_export_path), deterministic=True)
    logger.info("ONNX export successful")

    summary_payload = _build_summary_payload(
        summary,
        config,
        start_time,
        end_time,
        task_name=task_name,
        env_overrides=env_overrides,
    )
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file_handle:
        json.dump(summary_payload, file_handle, indent=2)
    logger.info("Saved training summary to %s", summary_path)

    # Make sure the monitor uses the final policy, then ask it to write the
    # final plot, metrics JSON, and rollout video.
    monitor.make_inference_fn_cached = make_inference_fn(params, deterministic=True)
    monitor.finalize()
    logger.info("=" * 80)

    return {
        "success": True,
        "params": params,
        "make_inference_fn": make_inference_fn,
        "summary": summary_payload,
    }


def main() -> None:
    """
    Read terminal arguments, prepare the run folder, and launch training.

    This is the small wrapper that turns `python Scripts/train.py ...` into a
    real training job.
    """
    args = parse_args()
    output_dir = _resolve_output_dir(
        args.output_dir,
        test_mode=args.test,
        task_name=args.task,
    )

    # Start from a clean folder so each run tells a clear story.
    _prepare_output_dir(output_dir)

    logger = setup_logging(output_dir, level=getattr(logging, args.log_level.upper()))
    config = TrainingConfig.for_task(args.task, test_mode=args.test)

    # Run one training job, then return success/failure to the shell.
    results = train_bittle(
        config=config,
        xml_path=args.xml_path,
        output_dir=output_dir,
        logger=logger,
        task_name=args.task,
        require_gpu=args.require_gpu,
    )
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
