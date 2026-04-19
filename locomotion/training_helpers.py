"""
Shared support code for training runs and sweep runs.

This file handles the supporting chores that are easy to forget about but need
to work every time:

- building the log files
- saving checkpoints while training is running
- parsing command-line options from the terminal

The main trainer calls into this file so the "real work" file can stay focused
on the training flow instead of setup details.
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
    Build the logger for one run.

    The logger writes to both the terminal and a timestamped log file. Old
    handlers are removed first so repeated runs in the same Python process do
    not accidentally duplicate each message.
    """
    output_dir = resolve_output_path(output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Reuse one named logger for the project so all training messages go through
    # the same channel.
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # Clear any handlers left over from an earlier run in the same process.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    # One handler prints short, readable messages to the console.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    # The second handler keeps the full detailed record on disk.
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
    Build the save hook that training calls every time it has a new policy.

    In plain language, this hook does two chores whenever training reports a new
    "current best guess" for the robot's brain:

    1. keep the latest version ready for the final video
    2. save a checkpoint on disk in case the run is interrupted
    """
    output_dir = resolve_output_path(output_dir)
    checkpoint_dir = (output_dir / "checkpoints").resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()

    def callback(current_step: int, make_policy: Any, params: Any) -> None:
        """
        Save one snapshot of the current policy.

        ``current_step`` is just the progress counter so the saved folder names
        stay easy to sort later.
        """
        if monitor is not None:
            # Remember the newest policy so the final video really reflects the
            # final training state rather than an earlier checkpoint.
            monitor.make_inference_fn_cached = make_policy(params)

        save_args = orbax_utils.save_args_from_target(params)
        path = checkpoint_dir / f"step_{current_step:08d}"
        checkpointer.save(path, params, force=True, save_args=save_args)
        logger.info("Saved checkpoint to %s", path)

    return callback


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Define the command-line options for ``train.py``.

    This is the list of switches a person can type at the terminal to choose
    the task, test mode, output folder, and log level.
    """
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
    """Read command-line options from the terminal and turn them into fields."""
    return build_arg_parser().parse_args(argv)
