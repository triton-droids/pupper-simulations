"""Shared repository-relative paths for locomotion entry points."""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_SCENE_PATH = (
    REPO_ROOT / "assets" / "descriptions" / "bittle" / "mjcf" / "bittle_adapted_scene.xml"
)

__all__ = ["PACKAGE_DIR", "REPO_ROOT", "DEFAULT_SCENE_PATH"]
