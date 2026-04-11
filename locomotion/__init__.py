"""
Shared entry points for the Bittle locomotion package.

This package intentionally keeps ``__init__`` lightweight. Importing the full
training stack pulls in heavier dependencies such as JAX, MuJoCo, and Brax, so
the package root only re-exports cheap path utilities that are safe to import
from tools, scripts, or tests.
"""

from __future__ import annotations

from .paths import DEFAULT_SCENE_PATH, PACKAGE_DIR, REPO_ROOT

__all__ = [
    "PACKAGE_DIR",
    "REPO_ROOT",
    "DEFAULT_SCENE_PATH",
]
