"""Central configuration for the ``asset_visualization`` subsystem.

This module exists to answer one simple question for the rest of the package:
"Where do the Bittle robot files live, and which of those files should we use
for conversion or visualization?"

Why keep these values here?
- It gives the package one canonical place for path definitions.
- It reduces copy-pasted relative strings sprinkled across the codebase.
- It makes the code easier to read because paths are grouped by purpose.

The paths are built with ``pathlib.Path`` relative to this file, which makes
them more robust than bare string literals such as ``"../assets/..."``.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project roots
# ---------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
ASSETS_ROOT = REPO_ROOT / "assets"
DESCRIPTIONS_ROOT = ASSETS_ROOT / "descriptions"

# ---------------------------------------------------------------------------
# Bittle robot asset tree
# ---------------------------------------------------------------------------

BITTLE_ROOT = DESCRIPTIONS_ROOT / "bittle"
BITTLE_URDF_ROOT = BITTLE_ROOT / "urdf"
BITTLE_MJCF_ROOT = BITTLE_ROOT / "mjcf"
BITTLE_XACRO_ROOT = BITTLE_ROOT / "xacro"
BITTLE_MESH_ROOT = BITTLE_ROOT / "meshes" / "stl"

# ---------------------------------------------------------------------------
# Source files used during conversion
# ---------------------------------------------------------------------------

BITTLE_URDF_PATH = BITTLE_URDF_ROOT / "bittle.urdf"
BITTLE_XACRO_PATH = BITTLE_XACRO_ROOT / "bittle.xacro"

# ---------------------------------------------------------------------------
# Generated or visualization-ready MJCF files
# ---------------------------------------------------------------------------

BITTLE_MJCF_PATH = BITTLE_MJCF_ROOT / "bittle.xml"
BITTLE_MJCF_ASSETS_PATH = BITTLE_MJCF_ROOT / "bittle_assets.xml"
BITTLE_MJCF_BODY_PATH = BITTLE_MJCF_ROOT / "bittle_body.xml"

BITTLE_ADAPTED_PATH = BITTLE_MJCF_ROOT / "bittle_adapted.xml"
BITTLE_ADAPTED_SCENE_PATH = BITTLE_MJCF_ROOT / "bittle_adapted_scene.xml"

# Optional legacy environment file. It is kept here for completeness even
# though the current viewer defaults to the adapted scene.
BITTLE_ENVIRONMENT_PATH = DESCRIPTIONS_ROOT / "bittleEnvironment.xml"

# ---------------------------------------------------------------------------
# Runtime toggles
# ---------------------------------------------------------------------------

# When True, regenerate the base MJCF from the URDF before visualization.
REGENERATE_MJCF = False

# Default viewer mode:
# - False: load the adapted scene used by the current project.
# - True:  load the raw robot-only MJCF.
LOAD_ENVIRONMENT = False


def to_str(path: Path) -> str:
    """Return a platform-safe string representation for external libraries."""
    return str(path)


__all__ = [
    "PACKAGE_DIR",
    "REPO_ROOT",
    "ASSETS_ROOT",
    "DESCRIPTIONS_ROOT",
    "BITTLE_ROOT",
    "BITTLE_URDF_ROOT",
    "BITTLE_MJCF_ROOT",
    "BITTLE_XACRO_ROOT",
    "BITTLE_MESH_ROOT",
    "BITTLE_URDF_PATH",
    "BITTLE_XACRO_PATH",
    "BITTLE_MJCF_PATH",
    "BITTLE_MJCF_ASSETS_PATH",
    "BITTLE_MJCF_BODY_PATH",
    "BITTLE_ADAPTED_PATH",
    "BITTLE_ADAPTED_SCENE_PATH",
    "BITTLE_ENVIRONMENT_PATH",
    "REGENERATE_MJCF",
    "LOAD_ENVIRONMENT",
    "to_str",
]
