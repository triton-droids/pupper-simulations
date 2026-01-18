#@title Import packages for plotting and creating graphics
import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

# Graphics and plotting.
import cv2
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

#@title Import MuJoCo, MJX, and Brax
from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict


import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
from orbax import checkpoint as ocp

import mujoco
#from mujoco import mjx

#from brax import base
#from brax import envs
#from brax import math
#from brax.base import Base, Motion, Transform
#from brax.base import State as PipelineState
#from brax.envs.base import Env, PipelineEnv, State
#from brax.mjx.base import State as MjxState
#from brax.training.agents.ppo import train as ppo
#from brax.training.agents.ppo import networks as ppo_networks
#from brax.io import html, mjcf, model

# Make model, data, and renderer
mj_model = mujoco.MjModel.from_xml_path("..\\assets\\bittle\\mjcf\\bittle.xml")
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

frames = []
mujoco.mj_resetData(mj_model, mj_data)
while mj_data.time < duration:
  mujoco.mj_step(mj_model, mj_data)
  if len(frames) < mj_data.time * framerate:
    renderer.update_scene(mj_data, scene_option=scene_option)
    pixels = renderer.render()
    frames.append(pixels)

# Save video using OpenCV
def save_video_opencv(frames, output_path, fps=60):
    """Save video frames to file using OpenCV."""
    if not frames:
        print("No frames to save")
        return

    height, width, channels = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Failed to open video writer for {output_path}")
        return

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Video saved to {output_path}")

# Save video to file
save_video_opencv(frames, "simulation_output.mp4", fps=framerate)