# pupper-simulations

Official repository for the Triton Pupper Simulations team - a robotics simulation project focused on quadruped locomotion using MuJoCo and Brax.

## Overview

This project provides simulation and reinforcement learning environments for the Bittle quadruped robot. It includes tools for visualization, training locomotion policies, and converting between different model formats (URDF, MJCF).

## Features

- MuJoCo-based physics simulation for quadruped robots
- Brax reinforcement learning environment for locomotion training
- Interactive 3D visualization using MuJoCo viewer
- Model conversion utilities (URDF to MJCF)
- Configurable reward functions for training
- JAX-accelerated training with GPU support

## Installation

### Prerequisites

- Python 3.12 (or compatible version)
- CUDA 12 support (for GPU-accelerated training)

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd pupper-simulations
```

2. Create and activate a virtual environment:

```bash
python -m venv simulations-env
source simulations-env/bin/activate  # On Windows: simulations-env\Scripts\activate
```

3. Install basic dependencies:

```bash
pip install -r requirements.txt
```

4. Install training dependencies (optional, for reinforcement learning):

```bash
pip install -r training_requirements.txt
```

## Directory Structure

```
pupper-simulations/
├── assets/                      # Robot models and media files
│   ├── descriptions/           # Robot description files
│   │   └── bittle/            # Bittle quadruped robot
│   │       ├── meshes/        # STL mesh files
│   │       ├── mjcf/          # MuJoCo model files
│   │       ├── urdf/          # URDF robot descriptions
│   │       └── xacro/         # Xacro source files
│   └── media/                 # Media assets
│
├── locomotion/                 # RL training environment
│   ├── bittle_env.py          # Brax environment for Bittle
│   ├── training_helpers.py    # Training utilities
│   ├── training.ipynb         # Training notebook
│   ├── env_test.py            # Environment testing
│   ├── bittle_adapted.xml     # Adapted model configuration
│   └── bittle_adapted_scene.xml  # Scene configuration
│
├── visualization/              # Visualization and simulation tools
│   ├── main.py                # Main visualization script
│   ├── model_converter.py     # URDF to MJCF converter
│   ├── simEnv.py              # Simulation environment
│   ├── constants.py           # Configuration constants
│   └── init.py                # Initialization utilities
│
├── requirements.txt            # Core dependencies
├── training_requirements.txt   # Training-specific dependencies
└── README.md                  # This file
```

## Usage

### Visualization

To launch the interactive MuJoCo viewer:

```bash
cd visualization
python main.py
```

This will:

1. Convert URDF to MJCF if necessary
2. Load the Bittle robot model
3. Launch the interactive 3D viewer

### Training

To train a locomotion policy:

```bash
cd locomotion
jupyter notebook training.ipynb
```

Or use the Python environment directly:

```python
from bittle_env import BittleEnv, get_config

# Create environment
config = get_config()
env = BittleEnv(config=config)

# Train using Brax PPO or other RL algorithms
```

### Model Conversion

The project includes utilities to convert URDF models to MuJoCo's MJCF format:

```python
from visualization.model_converter import convert_to_MJCF

convert_to_MJCF(urdf_path, mjcf_path, assets_path, body_path)
```

## Dependencies

### Core Dependencies

- `numpy==2.3.4` - Numerical computing
- `mujoco==3.3.4` - Physics simulation engine
- `xacrodoc==1.3.0` - URDF/Xacro processing

### Training Dependencies

- `jax[cuda12]==0.8.0` - GPU-accelerated numerical computing
- `brax==0.13.0` - Reinforcement learning library
- `ml_collections==1.1.0` - Configuration management
- `mediapy==1.2.4` - Media processing

## Known Issues

### URDF to MJCF Conversion Error

If you encounter this error during URDF to MJCF conversion:

```
model = mujoco.MjModel.from_xml_path(URDF_PATH)
ValueError: Error: error 'inertia must have positive eigenvalues' in alternative for principal axes
Element name 'f_1', id 74
```

**Solution:** Uninstall MuJoCo 3.3.4, install earlier versions (3.2.7, 3.3.1, or 3.3.3), then reinstall MuJoCo 3.3.4 as the primary library:

```bash
pip uninstall mujoco
pip install mujoco==3.2.7  # Or 3.3.1, 3.3.3
pip install mujoco==3.3.4
```

## Contributing

This is the official repository for the Triton Pupper Simulations team. For contributions, please follow the team's development guidelines.

## License

See the repository license file for details.
