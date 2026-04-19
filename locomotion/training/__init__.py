"""
Training package for Bittle runs.

This package groups the pieces that describe how training works:

- ``config``: preset run sizes and durations
- ``helpers``: terminal arguments, logs, and checkpoints
- ``monitor``: final metrics, plots, and rollout videos
- ``run``: the top-level training workflow
"""

from __future__ import annotations

from .config import TrainingConfig
from .helpers import build_arg_parser, parse_args, policy_params_callback, setup_logging
from .monitor import (
    FINAL_METRICS_FILENAME,
    FINAL_PLOT_FILENAME,
    FINAL_VIDEO_FILENAME,
    TrainingMonitor,
)
from .run import main, train_bittle

__all__ = [
    "FINAL_METRICS_FILENAME",
    "FINAL_PLOT_FILENAME",
    "FINAL_VIDEO_FILENAME",
    "TrainingConfig",
    "TrainingMonitor",
    "build_arg_parser",
    "main",
    "parse_args",
    "policy_params_callback",
    "setup_logging",
    "train_bittle",
]
