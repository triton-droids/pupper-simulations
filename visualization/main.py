import mujoco
import mujoco.viewer

import sys
import os

from model_converter import convert_to_MJCF
from init import logger
import constants


if __name__ == "__main__":
    # If MJCF file does not exist, convert from URDF
    if not os.path.exists(constants.BITTLE_MJCF_ASSETS_PATH) or not os.path.exists(constants.BITTLE_MJCF_BODY_PATH):
        logger.info("MJCF asset and body files not found, converting from URDF...")
        convert_to_MJCF(constants.BITTLE_URDF_PATH, constants.BITTLE_MJCF_PATH, 
        constants.BITTLE_MJCF_ASSETS_PATH, constants.BITTLE_MJCF_BODY_PATH)
    else:
        logger.info("MJCF file exists, proceeding to visualization")

    logger.info("Loading Mujoco model")

    # Load environment model into memory
    model = mujoco.MjModel.from_xml_path(constants.BITTLE_ENVIRONMENT_PATH)
    data = mujoco.MjData(model)
    
    # Initialize with default XML values
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    logger.info(f"Model stats: {model.nbody} bodies, {model.njnt} joints, "
                f"{model.nv} DOF, {model.nu} actuators, {model.nmesh} meshes")
    
    # Launch viewer
    logger.info("Launching Mujoco viewer")
    mujoco.viewer.launch(model, data)

    