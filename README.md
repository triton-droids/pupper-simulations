# pupper-simulations

Official repo for the Triton Pupper Simulations team

Bug:
If you're trying to convert the URDF to MJCF and run into this issue:

model = mujoco.MjModel.from_xml_path(URDF_PATH)
ValueError: Error: error 'inertia must have positive eigenvalues' in alternative for principal axes
Element name 'f_1', id 74

The thing that worked for me was uninstalling Mujoco 3.3.4, reinstalling eariler versions (3.2.7, 3.3.1, 3.3.3) and then reinstalling Mujoco 3.3.4 as the primary library.
