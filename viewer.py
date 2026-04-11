import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("assets/descriptions/bittle/mjcf/bittle_scene.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0)
mujoco.viewer.launch(m, d)
