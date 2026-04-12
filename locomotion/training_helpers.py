"""
Helper utilities shared by training entry points and sweep scripts.

This module groups together three concerns that are reused from multiple
callers:

- logger setup
- checkpoint callback construction
- command-line argument parsing for ``train.py``

The parser intentionally stays lightweight: it only exposes the task selector
and runtime paths, while the actual environment class lookup happens inside
``train.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from flax.training import orbax_utils
from orbax import checkpoint as ocp

from locomotion.paths import DEFAULT_SCENE_PATH, resolve_output_path


LOGGER_NAME = "bittle_training"
TASK_CHOICES = ("locomotion", "dance")


def setup_logging(output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Configure the package logger for one training run.

    Existing handlers are removed before new ones are added. That matters for
    sweeps and repeated local experiments where ``setup_logging`` may be called
    more than once in the same Python process.
    """
    output_dir = resolve_output_path(output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"training_{timestamp}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def policy_params_callback(
    output_dir: Path,
    logger: logging.Logger,
    monitor: Any | None = None,
):
    """
    Create the callback that Brax calls whenever policy parameters are saved.

    The callback performs two jobs:

    1. cache an inference function in ``TrainingMonitor`` for video generation
    2. write the latest policy parameters to an Orbax checkpoint directory
    """
    output_dir = resolve_output_path(output_dir)
    checkpoint_dir = (output_dir / "checkpoints").resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()

    def callback(current_step: int, make_policy: Any, params: Any) -> None:
        """Persist one checkpoint snapshot for the current training step."""
        if monitor is not None and monitor.make_inference_fn_cached is None:
            monitor.make_inference_fn_cached = make_policy(params)

        save_args = orbax_utils.save_args_from_target(params)
        path = checkpoint_dir / f"step_{current_step:08d}"
        checkpointer.save(path, params, force=True, save_args=save_args)
        logger.info("Saved checkpoint to %s", path)

    return callback


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser shared by training entry points."""
    parser = argparse.ArgumentParser(
        description="Train a Bittle quadruped locomotion or dance policy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test
  python locomotion/train.py --test

  # Full training run
  python locomotion/train.py

  # Train the dance task instead of locomotion
  python locomotion/train.py --task dance

  # Custom output directory
  python locomotion/train.py --output_dir ./experiments/run_001
        """,
    )

    parser.add_argument(
        "--task",
        type=str,
        default="locomotion",
        choices=TASK_CHOICES,
        help="Training task/environment to run.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the smaller test-mode configuration.",
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default=str(DEFAULT_SCENE_PATH),
        help="Path to the MuJoCo scene XML file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory for checkpoints, plots, videos, and logs. Relative paths "
            "are placed under the repo outputs/ folder."
        ),
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging verbosity.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse training CLI arguments from ``argv`` or from ``sys.argv``."""
    return build_arg_parser().parse_args(argv)
