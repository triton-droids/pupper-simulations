"""Launch the Bittle MuJoCo viewer with optional MJCF regeneration.

High-level responsibility
-------------------------
This script is the entry point for the ``asset_visualization`` subsystem.

It does three things:
1. Make sure the expected MJCF artifacts exist.
2. Choose which model file to visualize.
3. Load the model into MuJoCo and launch the interactive viewer.

Recommended usage
-----------------
From the repository root, prefer:

    python -m asset_visualization.main

That uses package imports cleanly and makes file resolution more predictable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

try:  # Support both module execution and direct script execution.
    from . import constants
    from .logging_utils import get_logger
    from .model_converter import convert_to_mjcf, model_stats
except ImportError:  # pragma: no cover - convenience fallback for script usage
    import constants  # type: ignore
    from logging_utils import get_logger  # type: ignore
    from model_converter import convert_to_mjcf, model_stats  # type: ignore


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize the Bittle robot in MuJoCo.",
    )
    parser.add_argument(
        "--regenerate-mjcf",
        action="store_true",
        help="Force regeneration of the base MJCF from the URDF before launching.",
    )
    parser.add_argument(
        "--robot-only",
        action="store_true",
        help="Load the raw robot MJCF instead of the adapted scene file.",
    )
    return parser.parse_args()


def ensure_mjcf_artifacts(*, regenerate: bool) -> None:
    """Ensure the extracted MJCF files needed by the workflow exist.

    The visualization workflow expects the main MJCF file plus extracted asset
    and body include files. If those do not exist yet, this function creates
    them from the URDF source.
    """
    asset_file_missing = not constants.BITTLE_MJCF_ASSETS_PATH.exists()
    body_file_missing = not constants.BITTLE_MJCF_BODY_PATH.exists()

    if asset_file_missing or body_file_missing or regenerate:
        logger.info("MJCF artifacts are missing or regeneration was requested.")
        convert_to_mjcf(
            urdf_path=constants.BITTLE_URDF_PATH,
            mjcf_path=constants.BITTLE_MJCF_PATH,
            asset_path=constants.BITTLE_MJCF_ASSETS_PATH,
            body_path=constants.BITTLE_MJCF_BODY_PATH,
            regenerate=regenerate,
        )
    else:
        logger.info("MJCF artifacts already exist. Reusing current files.")


def resolve_model_path(*, robot_only: bool) -> Path:
    """Choose the model file to load into the viewer."""
    return constants.BITTLE_MJCF_PATH if robot_only else constants.BITTLE_ADAPTED_SCENE_PATH


def load_model(model_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a MuJoCo model and initialize its runtime data."""
    logger.info("Loading MuJoCo model from %s", model_path)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Reset and forward once so the viewer opens from a clean, valid state.
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    logger.info("Model stats: %s", model_stats(model))
    return model, data


def main() -> int:
    """Run the visualization workflow."""
    args = parse_args()

    ensure_mjcf_artifacts(regenerate=args.regenerate_mjcf)

    model_path = resolve_model_path(robot_only=args.robot_only)
    model, data = load_model(model_path)

    logger.info("Launching interactive MuJoCo viewer")
    mujoco.viewer.launch(model, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
