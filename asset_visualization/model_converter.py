"""Utilities for converting and restructuring Bittle robot models.

High-level responsibility
-------------------------
This module sits between raw robot description files and the viewer / simulator.

It can:
1. Convert a XACRO file into MJCF.
2. Convert a URDF file into MJCF through MuJoCo.
3. Split a standalone MJCF file into smaller include-style XML fragments
   for assets and body geometry.

Why split MJCF into include files?
----------------------------------
Keeping assets and body structure in smaller files can make the model layout
easier to inspect and reuse in larger scene files.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from xacrodoc import XacroDoc

try:  # Support both ``python -m asset_visualization.main`` and direct execution.
    from . import constants
    from .logging_utils import get_logger
except ImportError:  # pragma: no cover - convenience fallback for script usage
    import constants  # type: ignore
    from logging_utils import get_logger  # type: ignore


logger = get_logger(__name__)


def model_stats(model: mujoco.MjModel) -> str:
    """Return a short human-readable summary of a MuJoCo model."""
    return (
        f"{model.nbody} bodies, {model.njnt} joints, "
        f"{model.nv} DOF, {model.nu} actuators, {model.nmesh} meshes"
    )


def _copy_element(element: ET.Element) -> ET.Element:
    """Deep-copy an XML element using round-trip serialization."""
    return ET.fromstring(ET.tostring(element, encoding="unicode"))


def _write_xml(tree: ET.ElementTree, path: Path) -> None:
    """Write an XML tree to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def xacro_to_mjcf(xacro_path: Path | str, mjcf_path: Path | str, mesh_dir: Path | str) -> Path:
    """Convert a XACRO model to MJCF.

    Args:
        xacro_path: Source XACRO file.
        mjcf_path: Output MJCF file to create.
        mesh_dir: Mesh directory written into the generated MJCF.

    Returns:
        The output MJCF path.
    """
    xacro_path = Path(xacro_path)
    mjcf_path = Path(mjcf_path)
    mesh_dir = Path(mesh_dir)

    logger.info("Converting XACRO to MJCF: %s -> %s", xacro_path, mjcf_path)
    mjcf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = XacroDoc.from_file(str(xacro_path))
    doc.to_mjcf_file(str(mjcf_path), strippath="true", meshdir=str(mesh_dir))
    return mjcf_path


def convert_urdf_to_mjcf(
    urdf_path: Path | str,
    mjcf_path: Path | str,
    *,
    regenerate: bool = False,
) -> Path:
    """Convert a URDF model into a standalone MJCF file using MuJoCo.

    Args:
        urdf_path: Source URDF file.
        mjcf_path: Output MJCF file to create.
        regenerate: When True, overwrite an existing MJCF file.

    Returns:
        The output MJCF path.
    """
    urdf_path = Path(urdf_path)
    mjcf_path = Path(mjcf_path)

    if mjcf_path.exists() and not regenerate:
        logger.info("Reusing existing MJCF at %s", mjcf_path)
        return mjcf_path

    logger.info("Converting URDF to MJCF: %s -> %s", urdf_path, mjcf_path)
    mjcf_path.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    mujoco.mj_saveLastXML(str(mjcf_path), model)

    logger.info("Saved MJCF to %s", mjcf_path)
    logger.info("Converted model stats: %s", model_stats(model))
    return mjcf_path


def extract_mjcf_includes(
    mjcf_path: Path | str,
    asset_path: Path | str,
    body_path: Path | str,
    *,
    body_name: str = "bittle",
    body_position: str = "0 0 0.5",
) -> tuple[Path, Path]:
    """Extract asset and worldbody sections into include-style XML files.

    The output files use a ``<mujocoinclude>`` wrapper so they can be included
    by other MJCF scene files.

    Args:
        mjcf_path: Source standalone MJCF file.
        asset_path: Output file for the extracted ``<asset>`` block.
        body_path: Output file for the extracted ``<worldbody>`` contents.
        body_name: Name of the wrapper body used in the extracted body file.
        body_position: Position assigned to the wrapper body.

    Returns:
        Tuple of ``(asset_path, body_path)``.
    """
    mjcf_path = Path(mjcf_path)
    asset_path = Path(asset_path)
    body_path = Path(body_path)

    logger.info("Extracting include files from %s", mjcf_path)

    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    asset_section = root.find("asset")
    if asset_section is None:
        raise ValueError(f"No <asset> section found in {mjcf_path}")

    worldbody_section = root.find("worldbody")
    if worldbody_section is None:
        raise ValueError(f"No <worldbody> section found in {mjcf_path}")

    asset_wrapper = ET.Element("mujocoinclude")
    asset_wrapper.append(_copy_element(asset_section))
    ET.indent(asset_wrapper, space="  ")
    _write_xml(ET.ElementTree(asset_wrapper), asset_path)
    logger.info("Wrote asset include file to %s", asset_path)

    body_wrapper = ET.Element("mujocoinclude")
    body_element = ET.SubElement(
        body_wrapper,
        "body",
        attrib={"name": body_name, "pos": body_position},
    )
    for child in worldbody_section:
        body_element.append(_copy_element(child))

    ET.indent(body_wrapper, space="  ")
    _write_xml(ET.ElementTree(body_wrapper), body_path)
    logger.info("Wrote body include file to %s", body_path)

    return asset_path, body_path


def convert_to_mjcf(
    urdf_path: Path | str,
    mjcf_path: Path | str,
    asset_path: Path | str,
    body_path: Path | str,
    *,
    regenerate: bool | None = None,
) -> tuple[Path, Path, Path]:
    """Run the full URDF -> MJCF -> include extraction pipeline.

    This is the main entry point used by the visualization script.

    Args:
        urdf_path: Source URDF file.
        mjcf_path: Output standalone MJCF path.
        asset_path: Output path for the extracted asset include file.
        body_path: Output path for the extracted body include file.
        regenerate: Overrides ``constants.REGENERATE_MJCF`` when provided.

    Returns:
        Tuple of ``(mjcf_path, asset_path, body_path)``.
    """
    should_regenerate = constants.REGENERATE_MJCF if regenerate is None else regenerate

    mjcf_path = convert_urdf_to_mjcf(
        urdf_path=urdf_path,
        mjcf_path=mjcf_path,
        regenerate=should_regenerate,
    )
    asset_path, body_path = extract_mjcf_includes(mjcf_path, asset_path, body_path)
    return mjcf_path, asset_path, body_path


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

XACRO_to_MJCF = xacro_to_mjcf
convert_to_MJCF = convert_to_mjcf
extractMJCFAssets = extract_mjcf_includes

__all__ = [
    "XACRO_to_MJCF",
    "convert_to_MJCF",
    "extractMJCFAssets",
    "xacro_to_mjcf",
    "convert_urdf_to_mjcf",
    "extract_mjcf_includes",
    "convert_to_mjcf",
    "model_stats",
]
