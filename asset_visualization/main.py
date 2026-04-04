import mujoco
import mujoco.viewer

import os

from model_converter import convert_to_MJCF
from init import logger
import constants


if __name__ == "__main__":
    # If MJCF file does not exist, convert from URDF
    if (
        not os.path.exists(constants.BITTLE_MJCF_ASSETS_PATH)
        or not os.path.exists(constants.BITTLE_MJCF_BODY_PATH)
        or constants.REGENERATE_MJCF
    ):
        logger.info("MJCF asset and body files not found, converting from URDF...")
        convert_to_MJCF(
            constants.BITTLE_URDF_PATH,
            constants.BITTLE_MJCF_PATH,
            constants.BITTLE_MJCF_ASSETS_PATH,
            constants.BITTLE_MJCF_BODY_PATH,
        )
    else:
        logger.info("MJCF file exists, proceeding to visualization")

    logger.info("Loading Mujoco model")

    if constants.LOAD_ENV:
        # Load environment model into memory
        model = mujoco.MjModel.from_xml_path(constants.BITTLE_MJCF_PATH)
    else:
        # Load the bittle model into memory
        # model = mujoco.MjModel.from_xml_path(constants.BITTLE_MJCF_PATH)
        model = mujoco.MjModel.from_xml_path(constants.BITTLE_ADAPTED_SCENE_PATH)

    data = mujoco.MjData(model)

    # Initialize with default XML values
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    logger.info(
        f"Model stats: {model.nbody} bodies, {model.njnt} joints, "
        f"{model.nv} DOF, {model.nu} actuators, {model.nmesh} meshes"
    )

    mujoco.viewer.launch(model, data)
