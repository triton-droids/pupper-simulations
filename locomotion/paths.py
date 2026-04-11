"""
Repository-relative paths used across the locomotion package.

Keeping these values in one module avoids duplicated ``../assets/...`` strings
and makes entry points less sensitive to the current working directory.
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

__all__ = [
    "PACKAGE_DIR",
    "REPO_ROOT",
    "ASSETS_ROOT",
    "OUTPUTS_ROOT",
    "BITTLE_ROOT",
    "BITTLE_MJCF_ROOT",
    "DEFAULT_SCENE_PATH",
]
