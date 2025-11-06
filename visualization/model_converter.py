import mujoco
import xml.etree.ElementTree as ET

from init import logger
import os

'''
Converts URDF model to MJCF, which is Mujoco compatible
(Model is converted as a standalone file, not an asset file)
'''
def convert_to_MJCF(URDF_PATH, MJCF_PATH, ASSET_PATH, BODY_PATH):
    if not os.path.exists(MJCF_PATH):
        logger.info(f"MJCF file not found at {MJCF_PATH}, converting from URDF at {URDF_PATH}")

        # Loads URDF model into memory and saves as MJCF
        model = mujoco.MjModel.from_xml_path(URDF_PATH)
        mujoco.mj_saveLastXML(MJCF_PATH, model)

        logger.info(f"Converted URDF to MJCF and saved to {MJCF_PATH}")
        logger.info(f"Model stats: {model.nbody} bodies, {model.njnt} joints, "
                    f"{model.nv} DOF, {model.nu} actuators, {model.nmesh} meshes")

    logger.info(f"Extracting assets and body to separate files")
    extractMJCFAssets(MJCF_PATH, ASSET_PATH, BODY_PATH)

'''
Extracts the worldbody and asset sections from a standalone MJCF file
and creates separate asset-only files (without <mujoco> wrapper)
'''
def extractMJCFAssets(MJCF_PATH, ASSET_PATH, BODY_PATH):
    # Parse the MJCF XML
    tree = ET.parse(MJCF_PATH)
    root = tree.getroot()
    
    # Extract asset section
    asset_section = root.find('asset')
    if asset_section is not None:
        # Wrap in mujocoinclude
        include_wrapper = ET.Element('mujocoinclude')
        include_wrapper.append(asset_section)
        
        # Indent the XML properly
        ET.indent(include_wrapper, space='  ')
        
        asset_tree = ET.ElementTree(include_wrapper)
        asset_tree.write(ASSET_PATH, encoding='utf-8', xml_declaration=True)
        logger.info(f"Extracted assets to {ASSET_PATH}")
    
    # Extract worldbody section
    worldbody_section = root.find('worldbody')
    if worldbody_section is not None:
        # Wrap in mujocoinclude and body element with freejoint to allow movement
        include_wrapper = ET.Element('mujocoinclude')
        wrapper = ET.SubElement(include_wrapper, 'body', attrib={'name': 'bittle', 'pos': '0 0 0.5'})
        freejoint = ET.SubElement(wrapper, 'freejoint')
        
        # Add all worldbody children to the wrapper
        for child in worldbody_section:
            wrapper.append(child)
        
        # Indent the XML properly
        ET.indent(include_wrapper, space='  ')
        
        body_tree = ET.ElementTree(include_wrapper)
        body_tree.write(BODY_PATH, encoding='utf-8', xml_declaration=True)
        logger.info(f"Extracted body to {BODY_PATH}")