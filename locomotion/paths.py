"""
Repository-relative paths used across the locomotion package.

Keeping these values in one module avoids duplicated ``../assets/...`` strings
and makes entry points less sensitive to the current working directory.

The module also exposes a small helper for resolving user-supplied output
paths. Relative output paths are anchored under the repository-level
``outputs/`` directory so ad hoc run names like ``run2`` do not end up inside
temporary working directories.
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

ASSETS_ROOT = REPO_ROOT / "assets"
OUTPUTS_ROOT = REPO_ROOT / "outputs"

BITTLE_ROOT = ASSETS_ROOT / "descriptions" / "bittle"
BITTLE_MJCF_ROOT = BITTLE_ROOT / "mjcf"

DEFAULT_SCENE_PATH = BITTLE_MJCF_ROOT / "bittle_adapted_scene.xml"


def resolve_output_path(path_like: str | Path) -> Path:
    """
    Resolve a user-supplied output path into an absolute repository path.

    Rules:
    - absolute paths are kept as-is
    - relative paths beginning with ``outputs/`` are interpreted relative to
      the repository root
    - all other relative paths are placed under the repository ``outputs/``
      directory
    """
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    if path.parts and path.parts[0] == OUTPUTS_ROOT.name:
        return (REPO_ROOT / path).resolve()

    return (OUTPUTS_ROOT / path).resolve()

__all__ = [
    "PACKAGE_DIR",
    "REPO_ROOT",
    "ASSETS_ROOT",
    "OUTPUTS_ROOT",
    "BITTLE_ROOT",
    "BITTLE_MJCF_ROOT",
    "DEFAULT_SCENE_PATH",
    "resolve_output_path",
]
