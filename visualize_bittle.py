"""Launch the MuJoCo viewer with the Bittle scene model."""

import mujoco
import mujoco.viewer
from pathlib import Path

XML_PATH = Path(__file__).parent / "assets" / "descriptions" / "bittle" / "mjcf" / "bittle_scene.xml"

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if home_key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    mujoco.mj_forward(model, data)

mujoco.viewer.launch(model, data)
