"""
Repository-relative paths used across the locomotion package.

Keeping these values in one module avoids duplicated ``../assets/...`` strings
and makes entry points less sensitive to the current working directory.

The module also exposes small helpers for resolving user-supplied output
paths and for allocating numbered sweep folders. Relative output paths are
anchored under ``Scripts/Outputs/`` so ad hoc run names like ``run2`` do not
end up inside temporary working directories.
"""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "Scripts"

ASSETS_ROOT = REPO_ROOT / "assets"
OUTPUTS_ROOT = SCRIPTS_DIR / "Outputs"
SWEEP_COUNTER_FILE = SCRIPTS_DIR / ".sweep_counter"

BITTLE_ROOT = ASSETS_ROOT / "descriptions" / "bittle"
BITTLE_MJCF_ROOT = BITTLE_ROOT / "mjcf"

DEFAULT_SCENE_PATH = BITTLE_MJCF_ROOT / "bittle_adapted_scene.xml"


def resolve_output_path(path_like: str | Path) -> Path:
    """
    Resolve a user-supplied output path into an absolute repository path.

    Rules:
    - absolute paths are kept as-is
    - relative paths beginning with ``Outputs/`` or legacy ``outputs/`` are
      interpreted under ``Scripts/``
    - relative paths beginning with ``Scripts/Outputs/`` are interpreted
      relative to the repository root
    - all other relative paths are placed under ``Scripts/Outputs/``
    """
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path

    if path.parts:
        first = path.parts[0].lower()
        if first == "scripts" and len(path.parts) > 1 and path.parts[1].lower() == "outputs":
            return (REPO_ROOT / path).resolve()
        if first == "outputs":
            return (SCRIPTS_DIR / path).resolve()

    return (OUTPUTS_ROOT / path).resolve()


def allocate_sweep_number(counter_path: Path = SWEEP_COUNTER_FILE) -> int:
    """
    Reserve the next sweep number from the shared counter file.

    The first call returns ``0``, the next returns ``1``, and so on. The file
    is updated immediately so separate sweep launches do not reuse the same
    label by accident.
    """
    counter_path.parent.mkdir(parents=True, exist_ok=True)

    if counter_path.exists():
        raw_value = counter_path.read_text(encoding="utf-8").strip()
        current_number = int(raw_value) if raw_value.isdigit() else 0
    else:
        current_number = 0

    counter_path.write_text(f"{current_number + 1}\n", encoding="utf-8")
    return current_number


def build_numbered_sweep_output_dir(
    base_dir: Path | None = None,
    *,
    counter_path: Path = SWEEP_COUNTER_FILE,
) -> Path:
    """
    Build the next default sweep folder using the project's ``Sweep #N`` naming.

    This keeps direct Python sweeps consistent with the shell script wrapper.
    """
    sweep_root = (base_dir or OUTPUTS_ROOT).resolve()
    sweep_number = allocate_sweep_number(counter_path)
    return sweep_root / f"Sweep #{sweep_number}"

__all__ = [
    "PACKAGE_DIR",
    "REPO_ROOT",
    "SCRIPTS_DIR",
    "ASSETS_ROOT",
    "OUTPUTS_ROOT",
    "SWEEP_COUNTER_FILE",
    "BITTLE_ROOT",
    "BITTLE_MJCF_ROOT",
    "DEFAULT_SCENE_PATH",
    "resolve_output_path",
    "allocate_sweep_number",
    "build_numbered_sweep_output_dir",
]
