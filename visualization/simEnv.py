#@title Import packages for plotting and creating graphics
import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

# Graphics and plotting.
import mediapy as media
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
import mediapy as media
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

# Simulate and display video.
media.show_video(frames, fps=framerate)